# src/sweep.py
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("\n>>", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(d: dict[str, Any], keys: list[str]) -> Optional[float]:
    """Return first float value found for any key in keys."""
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return None


def extract_test_metrics(model: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize metric keys across Model A/B.
    Model A writes: {"loss":..., "acc":..., "auc":...}
    Model B writes many keys from Keras: includes loss + genotype_* + timepoint_*
    """
    model = model.upper()
    out = {"test_loss": None, "test_acc": None, "test_auc": None, "test_day_acc": None}

    if model == "A":
        out["test_loss"] = safe_get(metrics, ["loss"])
        out["test_acc"]  = safe_get(metrics, ["acc", "binary_accuracy"])
        out["test_auc"]  = safe_get(metrics, ["auc"])
        return out

    # Model B
    out["test_loss"] = safe_get(metrics, ["loss"])

    out["test_acc"] = safe_get(metrics, [
        "genotype_acc",
        "genotype_binary_accuracy",
        "genotype_acc_1",
        "genotype_binary_accuracy_1",
    ])

    out["test_auc"] = safe_get(metrics, [
        "genotype_auc",
        "genotype_auc_1",
    ])

    out["test_day_acc"] = safe_get(metrics, [
        "timepoint_acc",
        "timepoint_sparse_categorical_accuracy",
        "timepoint_acc_1",
        "timepoint_sparse_categorical_accuracy_1",
    ])

    return out


def extract_val_summary(model: str, history: dict[str, Any]) -> dict[str, Any]:
    """
    From history.json, compute:
    - epochs_trained: length of training history
    - best_epoch: epoch index (1-based) where the chosen validation metric is best
    - best_val_auc: for Model A -> val_auc; for Model B -> val_genotype_auc
    """
    model = model.upper()
    out = {"epochs_trained": None, "best_epoch": None, "best_val_auc": None}

    # epochs trained
    # history keys vary but "loss" always present
    if "loss" in history and isinstance(history["loss"], list):
        out["epochs_trained"] = int(len(history["loss"]))

    if model == "A":
        key = "val_auc"
    else:
        key = "val_genotype_auc"

    vals = history.get(key)
    if isinstance(vals, list) and len(vals) > 0:
        best_i = max(range(len(vals)), key=lambda i: vals[i] if vals[i] is not None else -1)
        out["best_epoch"] = int(best_i + 1)  # 1-based
        out["best_val_auc"] = float(vals[best_i])

    return out


def main():
    ap = argparse.ArgumentParser(description="Run a small hyperparameter sweep and write results.csv")
    ap.add_argument("--model", choices=["A", "B", "both"], default="A")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--base_out", default="outputs/runs")
    ap.add_argument("--rerun", action="store_true", help="Re-run even if outputs exist")
    ap.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_py = repo_root / "src" / "train.py"
    data_dir = repo_root / args.data_dir
    base_out = repo_root / args.base_out
    base_out.mkdir(parents=True, exist_ok=True)

    # ---- Define sweep ----
    sweep_A = [
        # baseline: single-layer (depth=1, dropout=0.0)
        dict(name="A_baseline_d1_e30_b256", model="A", epochs=30, batch_size=256, depth=1, width=512, dropout=0.0, lr=1e-3),
        # core MLP
        dict(name="A_mlp_d2_e10_b256",      model="A", epochs=10, batch_size=256, depth=2, width=512, dropout=0.3, lr=1e-3),
        dict(name="A_mlp_d2_e30_b256",      model="A", epochs=30, batch_size=256, depth=2, width=512, dropout=0.3, lr=1e-3),
        dict(name="A_mlp_d2_e30_b512",      model="A", epochs=30, batch_size=512, depth=2, width=512, dropout=0.3, lr=1e-3),
        dict(name="A_mlp_d3_e30_b256",      model="A", epochs=30, batch_size=256, depth=3, width=512, dropout=0.3, lr=1e-3),
        dict(name="A_mlp_d3_e50_b512",      model="A", epochs=50, batch_size=512, depth=3, width=512, dropout=0.3, lr=1e-3),
        dict(name="A_mlp_d2_e10_b128",      model="A", epochs=10, batch_size=128, depth=2, width=512, dropout=0.3, lr=1e-3),
    ]

    sweep_B = [
        dict(name="B_mlp_d2_e30_b256_a03", model="B", epochs=30, batch_size=256, depth=2, width=512, dropout=0.3, lr=1e-3, loss_w_day=0.3),
        dict(name="B_mlp_d2_e30_b256_a01", model="B", epochs=30, batch_size=256, depth=2, width=512, dropout=0.3, lr=1e-3, loss_w_day=0.1),
        dict(name="B_mlp_d2_e30_b256_a05", model="B", epochs=30, batch_size=256, depth=2, width=512, dropout=0.3, lr=1e-3, loss_w_day=0.5),
    ]

    if args.model == "A":
        sweep = sweep_A
    elif args.model == "B":
        sweep = sweep_B
    else:
        sweep = sweep_A + sweep_B

    results: list[dict[str, Any]] = []

    for cfg in sweep:
        run_dir = base_out / cfg["name"]
        metrics_path = run_dir / "test_metrics.json"
        history_path = run_dir / "history.json"
        config_path  = run_dir / "config.json"

        need_run = args.rerun or not (metrics_path.exists() and history_path.exists() and config_path.exists())

        if need_run:
            cmd = [
                sys.executable, str(train_py),
                "--data_dir", str(data_dir),
                "--out_dir", str(run_dir),
                "--model", cfg["model"],
                "--epochs", str(cfg["epochs"]),
                "--batch_size", str(cfg["batch_size"]),
                "--lr", str(cfg["lr"]),
                "--width", str(cfg["width"]),
                "--depth", str(cfg["depth"]),
                "--dropout", str(cfg["dropout"]),
            ]
            if cfg["model"].upper() == "B":
                cmd += ["--loss_w_day", str(cfg.get("loss_w_day", 0.3))]
            run_cmd(cmd, dry_run=args.dry_run)
        else:
            print(f"Skipping (already done): {run_dir}")

        # Collect outputs if present
        if metrics_path.exists() and history_path.exists() and config_path.exists():
            conf = read_json(config_path)
            mets = read_json(metrics_path)
            hist = read_json(history_path)

            model = str(conf.get("model", cfg["model"])).upper()
            test_norm = extract_test_metrics(model, mets)
            val_sum = extract_val_summary(model, hist)

            row = {
                "run": run_dir.name,
                "model": model,
                "epochs_requested": conf.get("epochs", cfg["epochs"]),
                "epochs_trained": val_sum["epochs_trained"],
                "best_epoch": val_sum["best_epoch"],
                "best_val_auc": val_sum["best_val_auc"],
                "batch_size": conf.get("batch_size", cfg["batch_size"]),
                "lr": conf.get("lr", cfg["lr"]),
                "width": conf.get("width", cfg["width"]),
                "depth": conf.get("depth", cfg["depth"]),
                "dropout": conf.get("dropout", cfg["dropout"]),
                "loss_w_day": conf.get("loss_w_day", cfg.get("loss_w_day", "")),
                **test_norm,
            }
            results.append(row)
        else:
            print(f"WARNING: missing outputs for {run_dir} (train may have failed or dry_run)")

    # Write results.csv
    out_csv = base_out / "results.csv"
    fieldnames = [
        "run", "model",
        "epochs_requested", "epochs_trained", "best_epoch", "best_val_auc",
        "batch_size", "lr", "width", "depth", "dropout", "loss_w_day",
        "test_loss", "test_acc", "test_auc", "test_day_acc",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nWrote: {out_csv}")

    # Show best by validation AUC (correct selection criterion)
    def key_best(r: dict[str, Any]):
        v = r.get("best_val_auc")
        return (v is not None, v if v is not None else -1.0)

    if results:
        best = max(results, key=key_best)
        print("Best by best_val_auc:", best["run"], "best_val_auc=", best["best_val_auc"], "test_auc=", best["test_auc"])


if __name__ == "__main__":
    main()

