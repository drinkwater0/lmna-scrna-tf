# src/evaluate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required for evaluate.py (roc_curve, auc, confusion_matrix). "
        "Install with: pip install scikit-learn"
    ) from e


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _pick_history_keys(history: dict):
    """
    Model A keys typically: loss, acc, auc, val_loss, val_acc, val_auc
    Model B keys typically: loss, genotype_loss, genotype_acc, genotype_auc, val_genotype_*
    We plot PRIMARY TASK curves (genotype if available, else overall).
    """
    if "genotype_loss" in history and "val_genotype_loss" in history:
        loss_k, val_loss_k = "genotype_loss", "val_genotype_loss"
    else:
        loss_k, val_loss_k = "loss", "val_loss" if "val_loss" in history else None

    if "genotype_acc" in history and "val_genotype_acc" in history:
        acc_k, val_acc_k = "genotype_acc", "val_genotype_acc"
    elif "acc" in history and "val_acc" in history:
        acc_k, val_acc_k = "acc", "val_acc"
    else:
        # fallback: try BinaryAccuracy default name if user changed it
        # (rare, but safe)
        candidates = [k for k in history.keys() if k.endswith("acc")]
        val_candidates = [k for k in history.keys() if k.startswith("val_") and k.endswith("acc")]
        acc_k = candidates[0] if candidates else None
        val_acc_k = val_candidates[0] if val_candidates else None

    return loss_k, val_loss_k, acc_k, val_acc_k


def plot_loss(history: dict, out_path: Path):
    loss_k, val_loss_k, _, _ = _pick_history_keys(history)
    if loss_k is None:
        print("No loss key found in history; skipping loss plot.")
        return

    epochs = np.arange(1, len(history[loss_k]) + 1)
    plt.figure()
    plt.plot(epochs, history[loss_k], label=f"train ({loss_k})")
    if val_loss_k and val_loss_k in history:
        plt.plot(epochs, history[val_loss_k], label=f"val ({val_loss_k})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_accuracy(history: dict, out_path: Path):
    _, _, acc_k, val_acc_k = _pick_history_keys(history)
    if acc_k is None:
        print("No accuracy key found in history; skipping accuracy plot.")
        return

    epochs = np.arange(1, len(history[acc_k]) + 1)
    plt.figure()
    plt.plot(epochs, history[acc_k], label=f"train ({acc_k})")
    if val_acc_k and val_acc_k in history:
        plt.plot(epochs, history[val_acc_k], label=f"val ({val_acc_k})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (genotype)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return float(roc_auc)


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_path: Path):
    y_pred = (y_prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure()
    plt.imshow(cm)  # default colormap; no custom colors
    plt.title(f"Confusion matrix (threshold={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Control(0)", "Patient(1)"])
    plt.yticks([0, 1], ["Control(0)", "Patient(1)"])

    # annotate counts
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return cm


def main(run_dir: str, threshold: float, invert_prob: bool):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    history_path = run_dir / "history.json"
    y_path = run_dir / "y_test.npy"
    p_path = run_dir / "p_test.npy"

    for p in [history_path, y_path, p_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    history = _load_json(history_path)
    y_true = np.load(y_path).astype(np.int64)
    y_prob = np.load(p_path).astype(np.float32).reshape(-1)
    if invert_prob == True:
        y_prob = 1.0 - y_prob
    
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) training curves
    plot_loss(history, plots_dir / "loss_curve.png")
    plot_accuracy(history, plots_dir / "accuracy_curve.png")

    # 2) test ROC + confusion matrix
    roc_auc = plot_roc(y_true, y_prob, plots_dir / "roc_curve.png")
    cm = plot_confusion(y_true, y_prob, threshold, plots_dir / "confusion_matrix.png")

    # Save a small metrics summary too
    summary = {
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_true)),
        "pos_rate_test": float(y_true.mean()),
    }
    (run_dir / "plots_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved plots to:", plots_dir)
    print("Saved summary to:", run_dir / "plots_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. outputs/runs/model_A")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--invert_prob", action="store_true", help="If set, use 1 - p.")
    args = ap.parse_args()
    main(args.run_dir, args.threshold, args.invert_prob)
