#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata


def load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def roc_auc_manual(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = y_true.astype(int).reshape(-1)
    s = scores.astype(float).reshape(-1)
    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s, method="average")
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pick_metric(metrics: dict[str, Any], keys: list[str]) -> Optional[float]:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except Exception:
                return None
    return None


def load_auc(run_dir: Path) -> Optional[float]:
    metrics = load_json(run_dir / "test_metrics.json") or {}
    auc = pick_metric(metrics, ["auc", "genotype_auc", "genotype_auc_1", "genotype_genotype_auc"])
    if auc is not None:
        return auc

    y_path = run_dir / "y_test.npy"
    p_path = run_dir / "p_test.npy"
    if y_path.exists() and p_path.exists():
        y = np.load(y_path).astype(int).reshape(-1)
        p = np.load(p_path).astype(float).reshape(-1)
        if y.shape[0] == p.shape[0]:
            return roc_auc_manual(y, p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Model A vs Model B AUC across split schemes.")
    ap.add_argument("--runs_root", default="outputs/runs")
    ap.add_argument("--out_png", default="figures/auc_by_split.png")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--no_delta", action="store_true")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    split_defs = [
        ("original", "orig_A", "orig_B"),
        ("day-matched (test=0, val=16)", "dm_t0_v16_A", "dm_t0_v16_B"),
        ("day-matched (test=16, val=0)", "dm_t16_v0_A", "dm_t16_v0_B"),
        ("day-matched (test=19, val=0)", "dm_t19_v0_A", "dm_t19_v0_B"),
    ]

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for split_label, run_a, run_b in split_defs:
        dir_a = runs_root / run_a
        dir_b = runs_root / run_b
        if not dir_a.exists():
            missing.append(run_a)
            continue
        if not dir_b.exists():
            missing.append(run_b)
            continue

        auc_a = load_auc(dir_a)
        auc_b = load_auc(dir_b)
        rows.append({"split": split_label, "model": "A", "run": run_a, "auc": auc_a})
        rows.append({"split": split_label, "model": "B", "run": run_b, "auc": auc_b})

    if not rows:
        raise SystemExit("No runs found. Check --runs_root or run names.")

    df = pd.DataFrame(rows)

    # Plot
    splits = [s for s, _, _ in split_defs if s in df["split"].unique()]
    x = np.arange(len(splits))
    width = 0.36

    auc_a = [df[(df["split"] == s) & (df["model"] == "A")]["auc"].iloc[0] for s in splits]
    auc_b = [df[(df["split"] == s) & (df["model"] == "B")]["auc"].iloc[0] for s in splits]

    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width / 2, auc_a, width, label="Model A", color="#2C7FB8")
    plt.bar(x + width / 2, auc_b, width, label="Model B", color="#7FCDBB")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Test AUC (genotype)")
    plt.xticks(x, splits, rotation=10, ha="right")
    plt.title("AUC comparison across split schemes")
    plt.legend()

    if not args.no_delta:
        for i, (a, b) in enumerate(zip(auc_a, auc_b)):
            if a is None or b is None:
                continue
            delta = b - a
            if a is None or b is None:
                continue
            delta = b - a
            y = min(0.94, max(a, b) + 0.03)
            plt.text(i, y, f"Δ {delta:+.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_png = Path(args.out_png)
    if not out_png.is_absolute():
        out_png = (Path.cwd() / out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    if args.out_csv:
        out_csv = Path(args.out_csv)
        if not out_csv.is_absolute():
            out_csv = (Path.cwd() / out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved plot: {out_png}")
    if args.out_csv:
        print(f"Saved CSV: {out_csv}")
    if missing:
        print("Missing runs:", ", ".join(missing))


if __name__ == "__main__":
    main()
