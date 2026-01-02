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


def _to_float(val: Any) -> Optional[float]:
    try:
        if val is None or val == "":
            return None
        return float(val)
    except Exception:
        return None


def _pick_best_a_run(results_csv: Path) -> Optional[str]:
    if not results_csv.exists():
        return None
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if str(r.get("model", "")).upper() == "A"]
    if not rows:
        return None

    def score(row: dict[str, Any]) -> tuple[bool, float]:
        for key in ("best_val_auc", "test_auc", "test_acc"):
            v = _to_float(row.get(key))
            if v is not None:
                return True, v
        return False, -1.0

    best = max(rows, key=score)
    return best.get("run")


def _load_best_a_config(base_out: Path, run_name: str) -> Optional[dict[str, Any]]:
    if not run_name:
        return None
    cfg_path = base_out / run_name / "config.json"
    if not cfg_path.exists():
        return None
    return read_json(cfg_path)


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
    ap.add_argument("--out_csv", default="results.csv", help="Output CSV filename or path.")
    ap.add_argument("--best_a_run", default="", help="Name of the Model A run to seed Model B.")
    ap.add_argument("--best_a_csv", default="", help="CSV to pick the best Model A run (defaults to outputs/runs/results.csv).")
    ap.add_argument("--rerun", action="store_true", help="Re-run even if outputs exist")
    ap.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_py = repo_root / "src" / "train.py"
    data_dir = repo_root / args.data_dir
    base_out = repo_root / args.base_out
    base_out.mkdir(parents=True, exist_ok=True)

    # ---- Define sweep ----
    base_cfg = dict(
        epochs=20,
        batch_size=128,
        lr=1e-3,
        width=512,
        depth=1,
        dropout=0.0,
        widths="",
        arch="mlp",
        activation="relu",
        optimizer="adam",
        loss_geno="binary_crossentropy",
        loss_day="sparse_categorical_crossentropy",
        conv_filters="32,64",
        kernel_size=5,
        pool_size=2,
        momentum=0.9,
        loss_w_day=0.3,
    )

    sweep_base = [
        # single-layer network (depth=1, dropout=0)
        dict(name="d1_e20_b128_relu_adam", depth=1),
        # multi-layer network (depth=2)
        dict(name="d2_e20_b128_relu_adam", depth=2),
        # change epochs (20 vs 50)
        dict(name="d1_e50_b128_relu_adam", epochs=50),
        # change batch size (128 vs 512)
        dict(name="d1_e20_b512_relu_adam", batch_size=512),
        # change activation (relu vs tanh)
        dict(name="d1_e20_b128_tanh_adam", activation="tanh"),
        # change optimizer (adam vs sgd)
        dict(name="d1_e20_b128_relu_sgd", optimizer="sgd"),
    ]

    sweep_A = [
        {**base_cfg, **{k: v for k, v in cfg.items() if k != "name"}, "model": "A", "name": f"A_{cfg['name']}"}
        for cfg in sweep_base
    ]

    # Model B: use best Model A config and vary only loss_w_day.
    default_best_csv = Path(args.best_a_csv) if args.best_a_csv else (base_out / "results.csv")
    best_a_run = args.best_a_run or _pick_best_a_run(default_best_csv)
    best_a_cfg = _load_best_a_config(base_out, best_a_run) if best_a_run else None
    if best_a_run and best_a_cfg is None:
        print(f"WARNING: best_a_run '{best_a_run}' has no config.json; using base defaults for Model B.")
    if not best_a_run:
        print("WARNING: No best Model A run found; using base defaults for Model B.")

    b_base_cfg = dict(base_cfg)
    if best_a_cfg:
        for key in (
            "epochs", "batch_size", "lr", "width", "depth", "dropout",
            "widths", "arch", "activation", "optimizer", "loss_geno", "loss_day",
            "conv_filters", "kernel_size", "pool_size", "momentum",
        ):
            if key in best_a_cfg:
                b_base_cfg[key] = best_a_cfg[key]

    b_prefix = f"B_from_{best_a_run}" if best_a_run else "B_base"
    sweep_B = []
    for lw in (0.1, 0.3, 0.5):
        name = f"{b_prefix}_a{int(lw * 100):02d}"
        sweep_B.append({**b_base_cfg, "model": "B", "loss_w_day": lw, "name": name})

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
                "--arch", str(cfg.get("arch", "mlp")),
                "--activation", str(cfg.get("activation", "relu")),
                "--optimizer", str(cfg.get("optimizer", "adam")),
                "--loss_geno", str(cfg.get("loss_geno", "binary_crossentropy")),
                "--loss_day", str(cfg.get("loss_day", "sparse_categorical_crossentropy")),
                "--momentum", str(cfg.get("momentum", 0.9)),
            ]
            if cfg.get("widths"):
                cmd += ["--widths", str(cfg["widths"])]
            if cfg.get("arch", "mlp") == "cnn":
                cmd += [
                    "--conv_filters", str(cfg.get("conv_filters", "32,64")),
                    "--kernel_size", str(cfg.get("kernel_size", 5)),
                    "--pool_size", str(cfg.get("pool_size", 2)),
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
            arch = str(conf.get("arch", cfg.get("arch", "mlp")))
            optimizer = str(conf.get("optimizer", cfg.get("optimizer", "adam")))
            is_multitask = (model == "B")
            is_cnn = (arch.lower() == "cnn")

            loss_day_val = conf.get("loss_day", cfg.get("loss_day", "")) if is_multitask else ""
            loss_w_day_val = conf.get("loss_w_day", cfg.get("loss_w_day", "")) if is_multitask else ""
            test_day_acc_val = test_norm.get("test_day_acc") if is_multitask else ""
            conv_filters_val = conf.get("conv_filters", cfg.get("conv_filters", "")) if is_cnn else ""
            kernel_size_val = conf.get("kernel_size", cfg.get("kernel_size", "")) if is_cnn else ""
            pool_size_val = conf.get("pool_size", cfg.get("pool_size", "")) if is_cnn else ""
            momentum_val = conf.get("momentum", cfg.get("momentum", "")) if optimizer.lower() == "sgd" else ""

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
                "widths": conf.get("widths", cfg.get("widths", "")),
                "arch": arch,
                "activation": conf.get("activation", cfg.get("activation", "relu")),
                "optimizer": optimizer,
                "loss_geno": conf.get("loss_geno", cfg.get("loss_geno", "binary_crossentropy")),
                "loss_day": loss_day_val,
                "conv_filters": conv_filters_val,
                "kernel_size": kernel_size_val,
                "pool_size": pool_size_val,
                "momentum": momentum_val,
                "loss_w_day": loss_w_day_val,
                "test_loss": test_norm.get("test_loss"),
                "test_acc": test_norm.get("test_acc"),
                "test_auc": test_norm.get("test_auc"),
                "test_day_acc": test_day_acc_val,
            }
            results.append(row)
        else:
            print(f"WARNING: missing outputs for {run_dir} (train may have failed or dry_run)")

    # Write results.csv
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = base_out / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run", "model",
        "epochs_requested", "epochs_trained", "best_epoch", "best_val_auc",
        "batch_size", "lr", "width", "depth", "dropout", "loss_w_day",
        "widths", "arch", "activation", "optimizer", "loss_geno", "loss_day",
        "conv_filters", "kernel_size", "pool_size", "momentum",
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
