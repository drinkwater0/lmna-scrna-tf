# src/train.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.sparse import load_npz


class CSRSequence(keras.utils.Sequence):
    """
    Keras Sequence that serves dense batches from a SciPy CSR matrix.
    Keeps RAM safe: densifies ONLY the batch.
    """
    def __init__(
        self,
        X_csr,
        y_geno: np.ndarray,
        y_day: np.ndarray | None,
        indices: np.ndarray,
        batch_size: int = 256,
        shuffle: bool = True,
        multitask: bool = False,
    ):
        self.X = X_csr
        self.y_geno = y_geno
        self.y_day = y_day
        self.indices = indices.astype(np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.multitask = bool(multitask)
        self._order = np.arange(len(self.indices), dtype=np.int64)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._order)

    def __getitem__(self, i: int):
        start = i * self.batch_size
        end = min((i + 1) * self.batch_size, len(self.indices))
        batch_pos = self._order[start:end]
        batch_idx = self.indices[batch_pos]

        Xb = self.X[batch_idx].toarray().astype(np.float32)
        ygeno = self.y_geno[batch_idx].astype(np.float32)

        if not self.multitask:
            return Xb, ygeno

        assert self.y_day is not None
        yday = self.y_day[batch_idx].astype(np.int64)
        return Xb, {"genotype": ygeno, "timepoint": yday}


def build_model(n_features: int, width: int, depth: int, dropout: float, multitask: bool):
    inp = keras.Input(shape=(n_features,), name="expr")
    x = inp
    for _ in range(depth):
        x = layers.Dense(width, use_bias=False)(x)
        x = layers.BatchNormalization()(x)          # OK to keep; consider LayerNorm if you want
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)

    shared = layers.Dense(max(width // 2, 32), activation="relu", name="shared")(x)

    out_geno = layers.Dense(1, activation="sigmoid", name="genotype")(shared)

    if not multitask:
        return keras.Model(inp, out_geno, name="model_A_genotype")

    out_day = layers.Dense(7, activation="softmax", name="timepoint")(shared)
    return keras.Model(inp, [out_geno, out_day], name="model_B_multitask")


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    seed: int
    batch_size: int
    epochs: int
    lr: float
    width: int
    depth: int
    dropout: float
    model: str
    loss_w_day: float


def main(cfg: TrainConfig):
    tf.keras.utils.set_random_seed(cfg.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    X_csr = load_npz(data_dir / "X_csr.npz")
    y_geno = np.load(data_dir / "y_geno.npy")
    y_day  = np.load(data_dir / "y_day.npy")
    split  = np.load(data_dir / "split.npy", allow_pickle=True).astype(str)

    tr_idx = np.where(split == "train")[0]
    va_idx = np.where(split == "val")[0]
    te_idx = np.where(split == "test")[0]

    multitask = (cfg.model.upper() == "B")
    n_features = X_csr.shape[1]

    train_seq = CSRSequence(X_csr, y_geno, y_day, tr_idx, cfg.batch_size, shuffle=True,  multitask=multitask)
    val_seq   = CSRSequence(X_csr, y_geno, y_day, va_idx, cfg.batch_size, shuffle=False, multitask=multitask)
    test_seq  = CSRSequence(X_csr, y_geno, y_day, te_idx, cfg.batch_size, shuffle=False, multitask=multitask)

    # ---- Build & compile ----
    model = build_model(n_features, cfg.width, cfg.depth, cfg.dropout, multitask)

    if not multitask:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                     keras.metrics.AUC(name="auc")],
        )
        monitor = "val_auc"
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
            loss={
                "genotype": keras.losses.BinaryCrossentropy(),
                "timepoint": keras.losses.SparseCategoricalCrossentropy(),
            },
            loss_weights={"genotype": 1.0, "timepoint": float(cfg.loss_w_day)},
            metrics={
                "genotype": [keras.metrics.BinaryAccuracy(name="acc"),
                             keras.metrics.AUC(name="auc")],
                "timepoint": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
            },
        )
        monitor = "val_genotype_auc"

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "model.keras"),
        monitor=monitor,
        mode="max",
        save_best_only=True,
    )
    early = keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode="max",
        patience=8,
        restore_best_weights=True,
    )

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=cfg.epochs,
        callbacks=[ckpt, early],
        verbose=2,
    )

    # ---- Evaluate on test ----
    best = keras.models.load_model(out_dir / "model.keras")

    if not multitask:
        test_metrics = best.evaluate(test_seq, verbose=0)
        # metrics order: [loss, acc, auc]
        metrics_dict = {"loss": float(test_metrics[0]), "acc": float(test_metrics[1]), "auc": float(test_metrics[2])}

        # predictions for plots
        y_true = y_geno[te_idx].astype(np.int64)
        y_prob = best.predict(test_seq, verbose=0).reshape(-1)
        np.save(out_dir / "y_test.npy", y_true)
        np.save(out_dir / "p_test.npy", y_prob)

    else:
        test_metrics = best.evaluate(test_seq, verbose=0)
        # Keras returns many metrics; store raw list + names
        metrics_dict = dict(zip(best.metrics_names, map(float, test_metrics)))

        # predictions
        y_true = y_geno[te_idx].astype(np.int64)
        preds = best.predict(test_seq, verbose=0)
        # preds[0] genotype prob (N,1), preds[1] day probs (N,7)
        y_prob = preds[0].reshape(-1)
        day_prob = preds[1]
        np.save(out_dir / "y_test.npy", y_true)
        np.save(out_dir / "p_test.npy", y_prob)
        np.save(out_dir / "p_day_test.npy", day_prob)
        np.save(out_dir / "y_day_test.npy", y_day[te_idx].astype(np.int64))

    # ---- Save ----
    (out_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

    print("Saved:", out_dir)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="outputs/runs/exp_001")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--model", choices=["A", "B"], default="A", help="A=genotype only, B=genotype + day")
    ap.add_argument("--loss_w_day", type=float, default=0.3, help="only for Model B")
    args = ap.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    main(parse_args())
 
