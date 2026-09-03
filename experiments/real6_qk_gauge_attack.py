from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outotesti.gauge import (
    joint_qk_head_gauge,
    max_head_score_relative_error,
)
from outotesti.geometry import (
    four_point_gaps_for_quartets,
    sampled_quartets,
    spectrum_randomized_matrix,
)
from outotesti.metrics import channel_distance


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model


def qk_by_layer(model):
    out = {}
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2:
            continue
        if ".q_proj.weight" in name:
            layer = int(name.split(".h.")[1].split(".")[0])
            out.setdefault(layer, {})["Q"] = (name, parameter)
        elif ".k_proj.weight" in name:
            layer = int(name.split(".h.")[1].split(".")[0])
            out.setdefault(layer, {})["K"] = (name, parameter)
    return out


def p95_gap(W: np.ndarray, quartets: np.ndarray) -> float:
    gaps = four_point_gaps_for_quartets(channel_distance(W), quartets)
    return float(np.quantile(gaps, 0.95))


def baseline_null(
    W: np.ndarray,
    *,
    controls: int,
    quartets: np.ndarray,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    vals = np.empty(controls, dtype=float)
    for i in range(controls):
        W0 = spectrum_randomized_matrix(W, rng)
        vals[i] = p95_gap(W0, quartets)

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if controls > 1 else 0.0
    q05 = float(np.quantile(vals, 0.05))
    return {
        "null_values": vals,
        "mean": mean,
        "std": std,
        "q05": q05,
        "median": float(np.median(vals)),
    }


def z_from_gap(gap: float, null: dict) -> float:
    return float((null["mean"] - gap) / max(null["std"], 1e-12))


def one_layer(
    Wq: np.ndarray,
    Wk: np.ndarray,
    *,
    num_heads: int,
    gauges: int,
    controls: int,
    quartets_count: int,
    seed: int,
) -> dict:
    qs = sampled_quartets(Wq.shape[0], quartets_count, seed=seed + 7)

    q_null = baseline_null(
        Wq, controls=controls, quartets=qs, seed=seed + 11
    )
    k_null = baseline_null(
        Wk, controls=controls, quartets=qs, seed=seed + 13
    )

    q_gap0 = p95_gap(Wq, qs)
    k_gap0 = p95_gap(Wk, qs)
    q_z0 = z_from_gap(q_gap0, q_null)
    k_z0 = z_from_gap(k_gap0, k_null)

    gauge_rows = []
    max_m_error = 0.0
    q_zs = []
    k_zs = []
    q_sig = []
    k_sig = []

    for g in range(gauges):
        q2, k2, _ = joint_qk_head_gauge(
            Wq,
            Wk,
            num_heads=num_heads,
            rng=np.random.default_rng(seed + 1000 + g),
        )
        merr = max_head_score_relative_error(
            Wq, Wk, q2, k2, num_heads=num_heads
        )
        max_m_error = max(max_m_error, merr)

        q_gap = p95_gap(q2, qs)
        k_gap = p95_gap(k2, qs)
        q_z = z_from_gap(q_gap, q_null)
        k_z = z_from_gap(k_gap, k_null)

        q_is_sig = bool(q_gap <= q_null["q05"])
        k_is_sig = bool(k_gap <= k_null["q05"])

        q_zs.append(q_z)
        k_zs.append(k_z)
        q_sig.append(q_is_sig)
        k_sig.append(k_is_sig)

        gauge_rows.append(
            {
                "gauge": int(g),
                "Q_gap": q_gap,
                "K_gap": k_gap,
                "Q_z": q_z,
                "K_z": k_z,
                "Q_significant_vs_same_spectrum": q_is_sig,
                "K_significant_vs_same_spectrum": k_is_sig,
                "max_head_score_relative_error": merr,
            }
        )

    return {
        "baseline": {
            "Q_gap": q_gap0,
            "K_gap": k_gap0,
            "Q_z": q_z0,
            "K_z": k_z0,
            "Q_significant_vs_same_spectrum": bool(q_gap0 <= q_null["q05"]),
            "K_significant_vs_same_spectrum": bool(k_gap0 <= k_null["q05"]),
        },
        "gauge": {
            "count": int(gauges),
            "Q_median_z": float(np.median(q_zs)),
            "K_median_z": float(np.median(k_zs)),
            "Q_median_abs_z_shift": float(np.median(np.abs(np.asarray(q_zs) - q_z0))),
            "K_median_abs_z_shift": float(np.median(np.abs(np.asarray(k_zs) - k_z0))),
            "Q_fraction_significant": float(np.mean(q_sig)),
            "K_fraction_significant": float(np.mean(k_sig)),
            "max_head_score_relative_error": float(max_m_error),
            "rows": gauge_rows,
        },
        "null": {
            "Q_mean": q_null["mean"],
            "Q_std": q_null["std"],
            "Q_q05": q_null["q05"],
            "K_mean": k_null["mean"],
            "K_std": k_null["std"],
            "K_q05": k_null["q05"],
            "controls": int(controls),
        },
    }


def mutate_model_with_gauge(model, *, seed: int):
    import torch

    layers = qk_by_layer(model)
    num_heads = int(
        getattr(model.config, "num_heads", getattr(model.config, "num_attention_heads"))
    )
    originals = []

    for layer in sorted(layers):
        qname, qparam = layers[layer]["Q"]
        kname, kparam = layers[layer]["K"]
        Wq = qparam.detach().float().cpu().numpy().astype(float)
        Wk = kparam.detach().float().cpu().numpy().astype(float)
        q2, k2, _ = joint_qk_head_gauge(
            Wq,
            Wk,
            num_heads=num_heads,
            rng=np.random.default_rng(seed + layer * 100),
        )
        originals.append((qparam, qparam.detach().clone()))
        originals.append((kparam, kparam.detach().clone()))
        with torch.no_grad():
            qparam.copy_(torch.as_tensor(q2, dtype=qparam.dtype, device=qparam.device))
            kparam.copy_(torch.as_tensor(k2, dtype=kparam.dtype, device=kparam.device))

    return originals


def restore_model(originals):
    import torch
    with torch.no_grad():
        for param, value in originals:
            param.copy_(value)


def logit_invariance(model, *, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    vocab = int(model.config.vocab_size)
    ids = torch.randint(
        0, vocab, (1, 24), dtype=torch.long, device=next(model.parameters()).device
    )
    with torch.no_grad():
        ref = model(input_ids=ids).logits.float()

    originals = mutate_model_with_gauge(model, seed=seed + 1)
    try:
        with torch.no_grad():
            out = model(input_ids=ids).logits.float()
    finally:
        restore_model(originals)

    diff = out - ref
    return {
        "max_abs_logit_error": float(diff.abs().max().item()),
        "relative_logit_error": float(
            torch.linalg.vector_norm(diff)
            / torch.clamp_min(torch.linalg.vector_norm(ref), 1e-12)
        ),
        "argmax_token_agreement": float(
            (out.argmax(dim=-1) == ref.argmax(dim=-1)).float().mean().item()
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--gauges", type=int, default=16)
    ap.add_argument("--controls", type=int, default=64)
    ap.add_argument("--quartets", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "real6" / "audit.json",
    )
    args = ap.parse_args()

    model = load_model(args.model)
    layers = qk_by_layer(model)
    num_heads = int(
        getattr(model.config, "num_heads", getattr(model.config, "num_attention_heads"))
    )

    detail = {}
    baseline_z = []
    gauge_median_z = []
    baseline_sig = []
    gauge_sig_fraction = []
    abs_shifts = []
    max_m_error = 0.0

    for layer in sorted(layers):
        _, qparam = layers[layer]["Q"]
        _, kparam = layers[layer]["K"]
        Wq = qparam.detach().float().cpu().numpy().astype(float)
        Wk = kparam.detach().float().cpu().numpy().astype(float)

        result = one_layer(
            Wq,
            Wk,
            num_heads=num_heads,
            gauges=args.gauges,
            controls=args.controls,
            quartets_count=args.quartets,
            seed=args.seed + layer * 10000,
        )
        detail[layer] = result

        for family in ("Q", "K"):
            baseline_z.append(result["baseline"][f"{family}_z"])
            gauge_median_z.append(result["gauge"][f"{family}_median_z"])
            baseline_sig.append(
                result["baseline"][f"{family}_significant_vs_same_spectrum"]
            )
            gauge_sig_fraction.append(
                result["gauge"][f"{family}_fraction_significant"]
            )
            abs_shifts.append(
                result["gauge"][f"{family}_median_abs_z_shift"]
            )

        max_m_error = max(
            max_m_error,
            result["gauge"]["max_head_score_relative_error"],
        )

        print(
            f"layer {layer} "
            f"Q z {result['baseline']['Q_z']:+.2f} -> "
            f"{result['gauge']['Q_median_z']:+.2f}; "
            f"K z {result['baseline']['K_z']:+.2f} -> "
            f"{result['gauge']['K_median_z']:+.2f}; "
            f"Merr={result['gauge']['max_head_score_relative_error']:.2e}"
        )

    baseline_z = np.asarray(baseline_z, dtype=float)
    gauge_median_z = np.asarray(gauge_median_z, dtype=float)
    baseline_sig = np.asarray(baseline_sig, dtype=bool)
    gauge_sig_fraction = np.asarray(gauge_sig_fraction, dtype=float)
    abs_shifts = np.asarray(abs_shifts, dtype=float)

    logits = logit_invariance(model, seed=args.seed + 999999)

    aggregate = {
        "raw_matrix_count": int(len(baseline_z)),
        "baseline_median_z": float(np.median(baseline_z)),
        "gauge_median_z": float(np.median(gauge_median_z)),
        "median_baseline_minus_gauge_z": float(
            np.median(baseline_z - gauge_median_z)
        ),
        "median_absolute_z_shift": float(np.median(abs_shifts)),
        "baseline_significant_count": int(np.sum(baseline_sig)),
        "median_gauge_significant_fraction": float(
            np.median(gauge_sig_fraction)
        ),
        "fraction_matrices_baseline_z_above_gauge_median": float(
            np.mean(baseline_z > gauge_median_z)
        ),
        "max_head_score_relative_error": float(max_m_error),
        "logit_invariance": logits,
    }

    invariant = (
        aggregate["max_head_score_relative_error"] < 1e-12
        and logits["relative_logit_error"] < 1e-5
        and logits["argmax_token_agreement"] == 1.0
    )
    gauge_sensitive = (
        aggregate["median_absolute_z_shift"] >= 1.0
        and aggregate[
            "fraction_matrices_baseline_z_above_gauge_median"
        ] >= 0.75
        and aggregate["median_gauge_significant_fraction"] <= 0.5
    )

    if invariant and gauge_sensitive:
        classification = "RAW_QK_TREE_SIGNAL_IS_GAUGE_DEPENDENT"
    elif invariant:
        classification = "RAW_QK_TREE_SIGNAL_SURVIVES_RANDOM_ORTHOGONAL_GAUGES"
    else:
        classification = "GAUGE_IMPLEMENTATION_INVARIANCE_FAILED"

    summary = {
        "experiment": "REAL6",
        "model": args.model,
        "protocol": {
            "gauges_per_layer": int(args.gauges),
            "exact_spectrum_controls_per_matrix": int(args.controls),
            "quartets": int(args.quartets),
            "gauge": (
                "independent orthogonal rotation per attention head, applied "
                "identically to Q and K output coordinates"
            ),
            "functional_invariant": (
                "W_Q,h^T W_K,h is unchanged under the joint orthogonal gauge"
            ),
            "predeclared_gauge_sensitive": (
                "median abs z shift >=1, baseline z above gauge median in >=75% "
                "of 16 Q/K matrices, median gauge significant fraction <=.5"
            ),
            "predeclared_invariance": (
                "max head score relative error <1e-12, model logit relative "
                "error <1e-5, argmax agreement 1.0"
            ),
        },
        "aggregate": aggregate,
        "classification": classification,
        "layers": detail,
        "stopping_line": (
            "If gauge-dependent, raw Q/K tree geometry is not a property of the "
            "learned attention function. Together with REAL5, this closes the "
            "TinyStories hidden-tree hypothesis."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("REAL6 aggregate")
    print(f"baseline median z:                     {aggregate['baseline_median_z']:+.3f}")
    print(f"gauge median z:                        {aggregate['gauge_median_z']:+.3f}")
    print(f"median absolute z shift:               {aggregate['median_absolute_z_shift']:.3f}")
    print(f"baseline significant:                  {aggregate['baseline_significant_count']} / 16")
    print(f"median gauge significant fraction:     {aggregate['median_gauge_significant_fraction']:.3f}")
    print(f"baseline z > gauge median:             {aggregate['fraction_matrices_baseline_z_above_gauge_median']:.3f}")
    print(f"max head score relative error:         {aggregate['max_head_score_relative_error']:.3e}")
    print(f"logit relative error:                  {logits['relative_logit_error']:.3e}")
    print(f"logit argmax agreement:                {logits['argmax_token_agreement']:.3f}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
