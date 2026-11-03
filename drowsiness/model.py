"""CNN-LSTM architecture.

A small CNN is applied to every frame (``TimeDistributed``) to extract eye and
face features; an LSTM then reads the sequence of per-frame features, so the
prediction depends on how the face changes over time, not on a single frame.
"""

from __future__ import annotations

from .config import CLASS_NAMES, IMG_SIZE, TIMESTEPS


def build_cnn_lstm(
    timesteps: int = TIMESTEPS,
    img_size: int = IMG_SIZE,
    num_classes: int = len(CLASS_NAMES),
    learning_rate: float = 1e-3,
):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(timesteps, img_size, img_size, 3), name="frames")

    x = inputs
    # 64x64 -> 32 -> 16 -> 8 -> 4: a 4x4x128 feature map per frame keeps
    # spatial layout (where the eyes are) while keeping the LSTM small.
    for filters in (32, 64, 128, 128):
        x = layers.TimeDistributed(layers.Conv2D(filters, 3, padding="same", activation="relu"))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)

    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)
    x = layers.LSTM(64, name="temporal")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="drowsiness")(x)

    model = keras.Model(inputs, outputs, name="ddd_cnn_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model(path):
    """Load a trained model (``.keras`` or the legacy Colab ``.h5``)."""
    from tensorflow import keras

    return keras.models.load_model(path, compile=False)


def input_spec(model) -> tuple[int, int]:
    """Read ``(timesteps, img_size)`` from a loaded model's input shape."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, timesteps, height, _, _ = shape
    return int(timesteps), int(height)
