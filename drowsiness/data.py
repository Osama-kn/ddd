"""Dataset discovery, loading, sequence building and splitting.

The Driver Drowsiness Dataset (DDD) ships as two folders of face crops
("Drowsy" and "Non Drowsy"). File names look like ``A0001.png``: an alphabetic
prefix followed by a running frame number. This module treats the prefix
(case-insensitive) as the subject / source-video id and the number as the
frame position, which lets us:

* build *temporal* sequences from consecutive frames of the same subject,
  instead of stacking unrelated random images, and
* split train/validation **by subject**, so the model is evaluated on people
  it has never seen (a random image-level split leaks near-identical frames
  of the same person into both sets and inflates accuracy).

This module deliberately does not import TensorFlow, so it can be tested
quickly without it.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CLASS_FOLDERS, IMAGE_EXTENSIONS, IMG_SIZE

_NAME_RE = re.compile(r"^([A-Za-z]+)[_\-]?(\d+)")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    subject: str
    frame: int


def parse_name(stem: str) -> tuple[str, int]:
    """Split a file stem like ``ZA0123`` into ``("za", 123)``.

    Unrecognised names fall back to ``(stem, 0)`` so every file is still used.
    """
    match = _NAME_RE.match(stem)
    if not match:
        return stem.lower(), 0
    return match.group(1).lower(), int(match.group(2))


def find_dataset_root(path: str | os.PathLike) -> Path:
    """Return the folder that directly contains the class folders.

    Works whether you point at the unzip folder or the inner
    ``Driver Drowsiness Dataset (DDD)`` folder.
    """
    path = Path(path)
    wanted = {name.lower() for name in CLASS_FOLDERS}
    for current, dirs, _ in os.walk(path):
        if wanted.issubset({d.lower() for d in dirs}):
            return Path(current)
    raise FileNotFoundError(
        f"Could not find the class folders {CLASS_FOLDERS} under '{path}'. "
        "Download the dataset first (see README)."
    )


def list_samples(root: str | os.PathLike) -> list[Sample]:
    """List every image with its label, subject id and frame number."""
    root = find_dataset_root(root)
    folders = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    samples: list[Sample] = []
    for label, folder_name in enumerate(CLASS_FOLDERS):
        folder = folders[folder_name.lower()]
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                subject, frame = parse_name(file.stem)
                samples.append(Sample(file, label, subject, frame))
    if not samples:
        raise FileNotFoundError(f"No images found under '{root}'.")
    return samples


def preprocess_face(image_rgb: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize an RGB face crop to the model's input size (uint8)."""
    import cv2

    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)


def _read_rgb(path: Path, size: int) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return preprocess_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size)


def load_images(samples: list[Sample], size: int = IMG_SIZE, workers: int = 8) -> np.ndarray:
    """Load all images as a ``uint8`` array of shape ``(N, size, size, 3)``.

    Kept as uint8 (about 0.5 GB for the full dataset at 64x64) and only
    scaled to [0, 1] per batch, which uses a quarter of the memory of float32.
    """
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = list(
            tqdm(
                pool.map(lambda s: _read_rgb(s.path, size), samples),
                total=len(samples),
                desc="Loading images",
                unit="img",
            )
        )
    return np.stack(images).astype(np.uint8)


def build_sequences(
    samples: list[Sample], timesteps: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows of consecutive frames from the same subject/class.

    Returns
    -------
    indices : (num_sequences, timesteps) int array of indices into ``samples``
    labels : (num_sequences,) int array
    groups : (num_sequences,) array of subject ids, used for splitting
    """
    if timesteps < 1 or stride < 1:
        raise ValueError("timesteps and stride must be >= 1")

    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        buckets[(sample.subject, sample.label)].append(i)

    indices, labels, groups = [], [], []
    for (subject, label), members in sorted(buckets.items()):
        members.sort(key=lambda i: (samples[i].frame, samples[i].path.name))
        for start in range(0, len(members) - timesteps + 1, stride):
            indices.append(members[start : start + timesteps])
            labels.append(label)
            groups.append(subject)

    if not indices:
        raise ValueError(
            f"No sequence of {timesteps} frames could be built. "
            "Use a smaller --timesteps or check the file names."
        )
    return np.asarray(indices, dtype=np.int64), np.asarray(labels), np.asarray(groups)


def split_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    val_fraction: float = 0.2,
    strategy: str = "subject",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train / validation positions.

    ``strategy="subject"`` keeps every subject entirely in one split (honest
    estimate of performance on new drivers). ``strategy="random"`` reproduces
    the original notebook's image-level split, for comparison.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    positions = np.arange(len(labels))
    if strategy == "random":
        return train_test_split(positions, test_size=val_fraction, stratify=labels, random_state=seed)
    if strategy != "subject":
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    if len(np.unique(groups)) < 2:
        raise ValueError("Subject split needs at least two subjects.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_pos, val_pos = next(splitter.split(positions, labels, groups))
    return train_pos, val_pos
