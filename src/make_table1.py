#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
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


def infer_split_label(data_dir: Optional[str]) -> str:
    if not data_dir:
        return ""
    name = Path(data_dir).name
    mapping = {
        "processed_original": "original",
        "processed_dm_t0_v16": "day-matched (test=0, val=16)",
        "processed_dm_t16_v0": "day-matched (test=16, val=0)",
        "processed_dm_t19_v0": "day-matched (test=19, val=0)",
    }
    return mapping.get(name, name)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    cfg = load_json(run_dir / "config.json") or {}
    metrics = load_json(run_dir / "test_metrics.json") or {}

    model = str(cfg.get("model", "?")).upper()
    data_dir = cfg.get("data_dir")

    y_path = run_dir / "y_test.npy"
    p_path = run_dir / "p_test.npy"
    n_test = None
    auc_from_arrays = None
    if y_path.exists():
        y = np.load(y_path).astype(int).reshape(-1)
        n_test = int(y.shape[0])
        if p_path.exists():
            p = np.load(p_path).astype(float).reshape(-1)
            if p.shape[0] == y.shape[0]:
                auc_from_arrays = roc_auc_manual(y, p)

    if model == "A":
        geno_loss = pick_metric(metrics, ["loss"])
        geno_acc = pick_metric(metrics, ["acc", "binary_accuracy"])
        geno_auc = pick_metric(metrics, ["auc"])
        combined_loss = geno_loss
    else:
        geno_loss = pick_metric(metrics, ["genotype_loss", "genotype_loss_1", "genotype_genotype_loss"])
        if geno_loss is None:
            geno_loss = pick_metric(metrics, ["loss"])
        geno_acc = pick_metric(metrics, [
            "genotype_acc",
            "genotype_binary_accuracy",
            "genotype_acc_1",
            "genotype_binary_accuracy_1",
            "genotype_genotype_acc",
        ])
        geno_auc = pick_metric(metrics, ["genotype_auc", "genotype_auc_1", "genotype_genotype_auc"])
        combined_loss = pick_metric(metrics, ["loss"])

    if geno_auc is None:
        geno_auc = auc_from_arrays

    return {
        "run": run_dir.name,
        "split": infer_split_label(data_dir),
        "model": model,
        "n_test_cells": n_test,
        "test_auc": geno_auc,
        "test_acc": geno_acc,
        "test_genotype_loss": geno_loss,
        "test_combined_loss": combined_loss,
        "data_dir": data_dir,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize default split runs for Table 1.")
    ap.add_argument("--runs_root", default="outputs/runs", help="Directory containing run subfolders.")
    ap.add_argument("--runs", nargs="*", default=[
        "orig_A", "orig_B",
        "dm_t0_v16_A", "dm_t0_v16_B",
        "dm_t16_v0_A", "dm_t16_v0_B",
        "dm_t19_v0_A", "dm_t19_v0_B",
    ])
    ap.add_argument("--out_csv", default="outputs/table1_default.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    rows: list[dict[str, Any]] = []
    missing = []

    for name in args.runs:
        run_dir = runs_root / name
        if not run_dir.exists():
            missing.append(name)
            continue
        rows.append(summarize_run(run_dir))

    if not rows:
        raise SystemExit("No run directories found. Check --runs_root and --runs.")

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = (Path.cwd() / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {out_csv}")
    if missing:
        print("Missing runs:", ", ".join(missing))


if __name__ == "__main__":
    main()
