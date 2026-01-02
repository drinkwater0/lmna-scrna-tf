#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def roc_auc_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Tie-aware ROC AUC via Mann–Whitney U.
    y_true must be 0/1.
    """
    y = y_true.astype(int)
    s = scores.astype(float)

    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(s, method="average")  # 1..N, average ties
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def load_auc_from_arrays(run_dir: Path) -> Tuple[Optional[float], Optional[int]]:
    y_path = run_dir / "y_test.npy"
    p_path = run_dir / "p_test.npy"
    if not (y_path.exists() and p_path.exists()):
        return None, None

    y = np.load(y_path).astype(int).reshape(-1)
    p = np.load(p_path).astype(float).reshape(-1)
    if y.shape[0] != p.shape[0]:
        raise RuntimeError(f"[{run_dir.name}] y_test and p_test length mismatch: {y.shape} vs {p.shape}")

    auc = roc_auc_manual(y, p)
    return auc, int(y.shape[0])


def pick_metric(metrics: Dict[str, Any], keys: list[str]) -> Optional[float]:
    for k in keys:
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                return None
    return None


def infer_model_letter(cfg: Optional[Dict[str, Any]]) -> str:
    if not cfg:
        return "?"
    m = cfg.get("model", None)
    if isinstance(m, str):
        return m
    # fallback: sometimes stored differently
    return "?"


def config_fingerprint(cfg: Optional[Dict[str, Any]]) -> str:
    if cfg is None:
        return "missing"
    # Keep only the knobs that should match for a fair comparison
    keep = ["seed", "batch_size", "epochs", "lr", "width", "depth", "dropout", "model", "loss_w_day"]
    sub = {k: cfg.get(k, None) for k in keep}
    return json.dumps(sub, sort_keys=True)


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    cfg = load_json(run_dir / "config.json")
    metrics = load_json(run_dir / "test_metrics.json") or {}

    auc_from_arrays, n_test = load_auc_from_arrays(run_dir)

    # Model A uses keys: loss/acc/auc
    # Model B uses keys like: genotype_loss, timepoint_loss, genotype_acc, genotype_auc, timepoint_acc, loss, etc.
    model_letter = infer_model_letter(cfg)

    auc_metric = pick_metric(metrics, ["auc", "genotype_auc"])
    acc_metric = pick_metric(metrics, ["acc", "genotype_acc"])
    loss_metric = pick_metric(metrics, ["loss"])
    tp_acc = pick_metric(metrics, ["timepoint_acc"])
    tp_loss = pick_metric(metrics, ["timepoint_loss"])

    out = {
        "run": run_dir.name,
        "model": model_letter,
        "data_dir": (cfg or {}).get("data_dir", None),
        "auc_reported": auc_metric,
        "auc_from_arrays": auc_from_arrays,
        "acc_reported": acc_metric,
        "loss_reported": loss_metric,
        "timepoint_acc": tp_acc,
        "timepoint_loss": tp_loss,
        "n_test_cells": n_test,
        "config_fp": config_fingerprint(cfg),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="outputs/runs", help="Folder containing run subdirs.")
    ap.add_argument("--leak_subdir", type=str, default="hvg from all samples", help="Subfolder inside runs_root with leaky runs.")
    ap.add_argument("--out_csv", type=str, default="outputs/leakage_comparison.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    leak_root = runs_root / args.leak_subdir

    if not runs_root.exists():
        raise SystemExit(f"runs_root not found: {runs_root}")
    if not leak_root.exists():
        raise SystemExit(f"leak_root not found: {leak_root}")

    # All "clean" run dirs = direct children of runs_root, excluding the leak_root itself
    clean_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name != leak_root.name])

    rows = []
    config_mismatches = []

    for clean in clean_dirs:
        leak = leak_root / clean.name
        if not leak.exists() or not leak.is_dir():
            continue  # no pair

        a = summarize_run(clean)
        b = summarize_run(leak)

        # choose AUC to compare: prefer recomputed from arrays, else reported
        a_auc = a["auc_from_arrays"] if a["auc_from_arrays"] is not None else a["auc_reported"]
        b_auc = b["auc_from_arrays"] if b["auc_from_arrays"] is not None else b["auc_reported"]

        a_acc = a["acc_reported"]
        b_acc = b["acc_reported"]

        # config sanity
        if a["config_fp"] != b["config_fp"]:
            config_mismatches.append((clean.name, clean / "config.json", leak / "config.json"))

        rows.append({
            "run": clean.name,
            "model": a["model"],
            "n_test_cells": a["n_test_cells"],
            "clean_auc": a_auc,
            "leak_auc": b_auc,
            "delta_auc_clean_minus_leak": (a_auc - b_auc) if (a_auc is not None and b_auc is not None) else None,
            "clean_acc": a_acc,
            "leak_acc": b_acc,
            "delta_acc_clean_minus_leak": (a_acc - b_acc) if (a_acc is not None and b_acc is not None) else None,
            "clean_loss": a["loss_reported"],
            "leak_loss": b["loss_reported"],
            "clean_timepoint_acc": a["timepoint_acc"],
            "leak_timepoint_acc": b["timepoint_acc"],
            "clean_data_dir": a["data_dir"],
            "leak_data_dir": b["data_dir"],
        })

    if not rows:
        raise SystemExit("No paired runs found. Check your folder names.")

    df = pd.DataFrame(rows)

    # Sort: biggest absolute AUC change first
    if "delta_auc_clean_minus_leak" in df.columns:
        df["abs_delta_auc"] = df["delta_auc_clean_minus_leak"].abs()
        df = df.sort_values(["abs_delta_auc", "run"], ascending=[False, True]).drop(columns=["abs_delta_auc"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Pretty print
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)

    print("\n=== Leakage vs No-leak comparison (paired runs) ===")
    print(df.to_string(index=False))

    if config_mismatches:
        print("\n⚠️ Config mismatches detected (you may be comparing different training settings):")
        for run, c1, c2 in config_mismatches:
            print(f" - {run}:")
            print(f"    clean: {c1}")
            print(f"    leak : {c2}")

    print(f"\nSaved CSV: {out_csv}")


if __name__ == "__main__":
    main()
