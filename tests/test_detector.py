import numpy as np

from drowsiness.detector import DrowsinessDetector, draw_overlay

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
FACE = (100, 60, 120, 120)


class FakeModel:
    """Returns a fixed drowsiness probability and records the input shape."""

    def __init__(self, drowsy_prob):
        self.drowsy_prob = drowsy_prob
        self.shapes = []

    def __call__(self, batch):
        self.shapes.append(batch.shape)
        return np.array([[1 - self.drowsy_prob, self.drowsy_prob]])


def make(model, **kw):
    return DrowsinessDetector(model, timesteps=4, img_size=64, face_detector=lambda f: FACE, **kw)


def test_buffers_until_enough_frames():
    model = FakeModel(0.1)
    det = make(model)
    results = [det.process(FRAME, now=i * 0.1) for i in range(4)]
    assert [r.smoothed_prob is None for r in results] == [True, True, True, False]
    assert model.shapes[0] == (1, 4, 64, 64, 3)
    assert results[-1].label == "Alert" and not results[-1].alarm


def test_alarm_only_after_sustained_drowsiness():
    det = make(FakeModel(0.9), threshold=0.6, alarm_after=1.0)
    t, results = 0.0, []
    for _ in range(20):
        results.append(det.process(FRAME, now=t))
        t += 0.1
    drowsy = [r for r in results if r.label == "Drowsy"]
    assert drowsy, "should detect drowsiness"
    first_drowsy = results.index(drowsy[0])
    assert not any(r.alarm for r in results[: first_drowsy + 9])  # < 1 s
    assert results[-1].alarm


def test_no_face_resets_state():
    det = DrowsinessDetector(FakeModel(0.9), timesteps=2, img_size=32, face_detector=lambda f: None)
    r = det.process(FRAME, now=0)
    assert r.face is None and r.label == "No face detected" and len(det.frames) == 0


def test_overlay_draws_without_error():
    det = make(FakeModel(0.9), alarm_after=0)
    for i in range(5):
        r = det.process(FRAME.copy(), now=i)
    out = draw_overlay(FRAME.copy(), r, fps=25.0)
    assert out.shape == FRAME.shape and out.any()
