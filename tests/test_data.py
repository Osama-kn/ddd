from pathlib import Path

import cv2
import numpy as np
import pytest

from drowsiness.data import (
    build_sequences,
    find_dataset_root,
    list_samples,
    parse_name,
    split_indices,
)


@pytest.mark.parametrize(
    "stem, expected",
    [("A0001", ("a", 1)), ("za0123", ("za", 123)), ("ZC_07", ("zc", 7)), ("weird", ("weird", 0))],
)
def test_parse_name(stem, expected):
    assert parse_name(stem) == expected


def _make_dataset(root: Path, subjects=("A", "B", "C", "D"), frames=6) -> Path:
    inner = root / "Driver Drowsiness Dataset (DDD)"
    for folder, case in (("Drowsy", str.upper), ("Non Drowsy", str.lower)):
        (inner / folder).mkdir(parents=True)
        for s in subjects:
            for i in range(frames):
                img = np.full((20, 20, 3), i * 10, dtype=np.uint8)
                cv2.imwrite(str(inner / folder / f"{case(s)}{i:04d}.png"), img)
    return inner


def test_find_root_and_labels(tmp_path):
    inner = _make_dataset(tmp_path)
    assert find_dataset_root(tmp_path) == inner
    samples = list_samples(tmp_path)
    assert len(samples) == 4 * 6 * 2
    drowsy = [s for s in samples if s.path.parent.name == "Drowsy"]
    assert {s.label for s in drowsy} == {1}
    # Upper/lower-case prefixes of the same letter map to one subject.
    assert {s.subject for s in samples} == {"a", "b", "c", "d"}


def test_sequences_are_consecutive_and_pure(tmp_path):
    _make_dataset(tmp_path)
    samples = list_samples(tmp_path)
    idx, labels, groups = build_sequences(samples, timesteps=4, stride=1)
    # 6 frames -> 3 windows per (subject, class)
    assert len(idx) == 4 * 2 * 3
    for window, label, group in zip(idx, labels, groups, strict=False):
        members = [samples[i] for i in window]
        assert {m.subject for m in members} == {group}
        assert {m.label for m in members} == {label}
        frames = [m.frame for m in members]
        assert frames == sorted(frames) and frames[-1] - frames[0] == 3


def test_subject_split_has_no_overlap(tmp_path):
    _make_dataset(tmp_path)
    _, labels, groups = build_sequences(list_samples(tmp_path), timesteps=4)
    train_pos, val_pos = split_indices(labels, groups, 0.25, "subject")
    assert set(groups[train_pos]).isdisjoint(groups[val_pos])
    assert len(train_pos) + len(val_pos) == len(labels)


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_dataset_root(tmp_path)
