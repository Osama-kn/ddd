"""Training and evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR, IMG_SIZE, TIMESTEPS
from .data import build_sequences, list_samples, load_images, split_indices


def _sequence_batches(images, seq_indices, labels, batch_size, shuffle, seed=42):
    """Keras data source that gathers frame sequences lazily from a uint8 array."""
    from tensorflow import keras

    num_classes = len(CLASS_NAMES)

    class SequenceBatches(keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.order = np.arange(len(seq_indices))
            self.rng = np.random.default_rng(seed)
            if shuffle:
                self.rng.shuffle(self.order)

        def __len__(self):
            return math.ceil(len(self.order) / batch_size)

        def __getitem__(self, i):
            batch = self.order[i * batch_size : (i + 1) * batch_size]
            x = images[seq_indices[batch]].astype(np.float32) / 255.0
            y = np.eye(num_classes, dtype=np.float32)[labels[batch]]
            return x, y

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.order)

    return SequenceBatches()


def _save_history_plot(history: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ("accuracy", "loss"), strict=False):
        ax.plot(history[metric], label=f"train {metric}")
        ax.plot(history[f"val_{metric}"], label=f"val {metric}")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_confusion_matrix(y_true, y_pred, path: Path) -> list[list[int]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm.tolist()


def train(
    data_dir: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    epochs: int = 15,
    batch_size: int = 32,
    timesteps: int = TIMESTEPS,
    img_size: int = IMG_SIZE,
    stride: int = 2,
    split: str = "subject",
    val_fraction: float = 0.2,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train the CNN-LSTM, save the best model and write an evaluation report."""
    import tensorflow as tf
    from sklearn.metrics import classification_report
    from tensorflow import keras

    from .model import build_cnn_lstm

    tf.keras.utils.set_random_seed(seed)
    model_path, report_dir = Path(model_path), Path(report_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(data_dir)
    print(f"Found {len(samples)} images, {len({s.subject for s in samples})} subjects.")
    seq_idx, labels, groups = build_sequences(samples, timesteps, stride)
    train_pos, val_pos = split_indices(labels, groups, val_fraction, split, seed)

    train_subjects, val_subjects = set(groups[train_pos]), set(groups[val_pos])
    print(
        f"{len(seq_idx)} sequences of {timesteps} frames | split='{split}': "
        f"{len(train_pos)} train / {len(val_pos)} val | "
        f"{len(train_subjects)} train subjects / {len(val_subjects)} val subjects"
    )

    images = load_images(samples, img_size)
    train_data = _sequence_batches(images, seq_idx[train_pos], labels[train_pos], batch_size, True, seed)
    val_data = _sequence_batches(images, seq_idx[val_pos], labels[val_pos], batch_size, False)

    # Balance the loss if one class dominates the training split.
    counts = np.bincount(labels[train_pos], minlength=len(CLASS_NAMES))
    class_weight = {i: len(train_pos) / (len(counts) * c) for i, c in enumerate(counts) if c}

    model = build_cnn_lstm(timesteps, img_size, learning_rate=learning_rate)
    model.summary()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ],
    ).history
    model.save(model_path)

    probs = model.predict(val_data, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = labels[val_pos]
    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    _save_history_plot(history, report_dir / "training_curves.png")
    cm = _save_confusion_matrix(y_true, y_pred, report_dir / "confusion_matrix.png")

    metrics = {
        "split": split,
        "timesteps": timesteps,
        "img_size": img_size,
        "train_sequences": int(len(train_pos)),
        "val_sequences": int(len(val_pos)),
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "val_accuracy": float(report["accuracy"]),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "confusion_matrix": cm,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        classification_report(
            y_true, y_pred, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES, zero_division=0
        )
    )
    print(f"Model saved to {model_path}; report written to {report_dir}/")
    return metrics
