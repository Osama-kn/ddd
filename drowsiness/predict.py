"""Offline prediction on still images or a folder of consecutive frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, IMAGE_EXTENSIONS
from .data import parse_name


def _load(path: Path, size: int) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def predict_paths(model_path: str, inputs: list[str]) -> list[tuple[str, str, float]]:
    """Classify face images.

    * An image file is treated as a static clip (the frame is repeated to fill
      the sequence).
    * A folder is treated as consecutive frames: it is sorted by frame number
      and classified with a sliding window; one result per window is returned.

    Returns ``(name, label, drowsy_probability)`` tuples.
    """
    from .model import input_spec, load_model

    model = load_model(model_path)
    timesteps, size = input_spec(model)
    results = []
    for item in map(Path, inputs):
        if item.is_dir():
            files = sorted(
                (f for f in item.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS),
                key=lambda f: (parse_name(f.stem), f.name),
            )
            if len(files) < timesteps:
                raise ValueError(f"'{item}' needs at least {timesteps} frames")
            frames = np.stack([_load(f, size) for f in files])
            windows = np.stack([frames[i : i + timesteps] for i in range(len(files) - timesteps + 1)])
            names = [f"{item.name}/{files[i + timesteps - 1].name}" for i in range(len(windows))]
        else:
            frame = _load(item, size)
            windows = np.repeat(frame[None, None], timesteps, axis=1)
            names = [item.name]

        probs = model.predict(windows, verbose=0)
        for name, p in zip(names, probs, strict=False):
            results.append((name, CLASS_NAMES[int(p.argmax())], float(p[1])))
    return results
