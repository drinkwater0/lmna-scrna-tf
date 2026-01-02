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


def _parse_int_list(text: str) -> list[int]:
    if text is None:
        return []
    items = [t.strip() for t in text.split(",") if t.strip()]
    return [int(t) for t in items]


def _build_optimizer(name: str, lr: float, momentum: float):
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")


def _build_loss(name: str):
    name = name.lower()
    if name in {"binary_crossentropy", "bce"}:
        return keras.losses.BinaryCrossentropy()
    if name in {"mean_squared_error", "mse"}:
        return keras.losses.MeanSquaredError()
    if name in {"sparse_categorical_crossentropy", "scc"}:
        return keras.losses.SparseCategoricalCrossentropy()
    raise ValueError(f"Unknown loss: {name}")


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


def _build_backbone(
    n_features: int,
    width: int,
    depth: int,
    dropout: float,
    activation: str,
    widths_list: list[int],
    arch: str,
    conv_filters: list[int],
    kernel_size: int,
    pool_size: int,
):
    arch = arch.lower()
    activation = activation.lower()
    inp = keras.Input(shape=(n_features,), name="expr")

    if arch == "mlp":
        layer_widths = widths_list if widths_list else [width] * depth
        if len(layer_widths) == 0:
            raise ValueError("MLP requires at least one hidden layer.")
        x = inp
        for w in layer_widths:
            x = layers.Dense(w, use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.Dropout(dropout)(x)
        shared_units = max(int(layer_widths[-1] // 2), 32)
        shared = layers.Dense(shared_units, activation=activation, name="shared")(x)
        return inp, shared

    if arch == "cnn":
        filters = conv_filters if conv_filters else [32, 64]
        if len(filters) == 0:
            raise ValueError("CNN requires at least one conv layer.")
        # Treat gene expression vector as a 1D signal for Conv1D.
        x = layers.Reshape((n_features, 1), name="reshape_for_conv")(inp)
        for f in filters:
            x = layers.Conv1D(f, kernel_size, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.MaxPooling1D(pool_size=pool_size)(x)
            x = layers.Dropout(dropout)(x)
        x = layers.GlobalMaxPooling1D()(x)
        shared_units = max(int(width // 2), 32)
        shared = layers.Dense(shared_units, activation=activation, name="shared")(x)
        return inp, shared

    raise ValueError(f"Unknown architecture: {arch}")


def build_model(
    n_features: int,
    width: int,
    depth: int,
    dropout: float,
    multitask: bool,
    arch: str,
    activation: str,
    widths_list: list[int],
    conv_filters: list[int],
    kernel_size: int,
    pool_size: int,
):
    inp, shared = _build_backbone(
        n_features,
        width,
        depth,
        dropout,
        activation,
        widths_list,
        arch,
        conv_filters,
        kernel_size,
        pool_size,
    )

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
    widths: str
    arch: str
    activation: str
    optimizer: str
    loss_geno: str
    loss_day: str
    conv_filters: str
    kernel_size: int
    pool_size: int
    momentum: float
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

    widths_list = _parse_int_list(cfg.widths)
    conv_filters = _parse_int_list(cfg.conv_filters)

    train_seq = CSRSequence(X_csr, y_geno, y_day, tr_idx, cfg.batch_size, shuffle=True,  multitask=multitask)
    val_seq   = CSRSequence(X_csr, y_geno, y_day, va_idx, cfg.batch_size, shuffle=False, multitask=multitask)
    test_seq  = CSRSequence(X_csr, y_geno, y_day, te_idx, cfg.batch_size, shuffle=False, multitask=multitask)

    # ---- Build & compile ----
    model = build_model(
        n_features,
        cfg.width,
        cfg.depth,
        cfg.dropout,
        multitask,
        cfg.arch,
        cfg.activation,
        widths_list,
        conv_filters,
        cfg.kernel_size,
        cfg.pool_size,
    )

    optimizer = _build_optimizer(cfg.optimizer, cfg.lr, cfg.momentum)
    loss_geno = _build_loss(cfg.loss_geno)
    loss_day = _build_loss(cfg.loss_day)

    if not multitask:
        model.compile(
            optimizer=optimizer,
            loss=loss_geno,
            metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                     keras.metrics.AUC(name="auc")],
        )
        monitor = "val_auc"
    else:
        model.compile(
            optimizer=optimizer,
            loss={
                "genotype": loss_geno,
                "timepoint": loss_day,
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
    ap.add_argument("--widths", default="", help="Comma-separated widths per MLP layer (overrides --width/--depth).")
    ap.add_argument("--arch", choices=["mlp", "cnn"], default="mlp")
    ap.add_argument("--activation", choices=["relu", "tanh", "elu", "selu"], default="relu")
    ap.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam")
    ap.add_argument("--loss_geno", choices=["binary_crossentropy", "mean_squared_error"], default="binary_crossentropy")
    ap.add_argument("--loss_day", choices=["sparse_categorical_crossentropy"], default="sparse_categorical_crossentropy")
    ap.add_argument("--conv_filters", default="32,64", help="Comma-separated Conv1D filters (CNN only).")
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--pool_size", type=int, default=2)
    ap.add_argument("--momentum", type=float, default=0.9, help="Only used for SGD.")
    ap.add_argument("--model", choices=["A", "B"], default="A", help="A=genotype only, B=genotype + day")
    ap.add_argument("--loss_w_day", type=float, default=0.3, help="only for Model B")
    args = ap.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    main(parse_args())
 
