"""Real-time drowsiness detection on a webcam or video stream.

Pipeline per frame:
    face detection (OpenCV Haar cascade) -> crop + resize -> rolling buffer of
    the last N face crops -> CNN-LSTM -> drowsiness probability ->
    moving-average smoothing -> alarm if the driver stays drowsy for too long.

The detection logic (``DrowsinessDetector``) is independent of the camera and
of TensorFlow: it takes any ``predict_fn`` that maps a batch of sequences to
class probabilities, which keeps it easy to test.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import CLASS_NAMES, DROWSY_INDEX, IMG_SIZE, TIMESTEPS

PredictFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class FrameResult:
    face: tuple[int, int, int, int] | None  # x, y, w, h
    drowsy_prob: float | None  # latest raw probability
    smoothed_prob: float | None  # moving average
    label: str
    alarm: bool


class DrowsinessDetector:
    def __init__(
        self,
        predict_fn: PredictFn,
        timesteps: int = TIMESTEPS,
        img_size: int = IMG_SIZE,
        smoothing: int = 8,
        threshold: float = 0.6,
        alarm_after: float = 1.5,
        face_detector: Callable[[np.ndarray], tuple | None] | None = None,
    ):
        self.predict_fn = predict_fn
        self.timesteps = timesteps
        self.img_size = img_size
        self.threshold = threshold
        self.alarm_after = alarm_after
        self.frames: deque = deque(maxlen=timesteps)
        self.history: deque = deque(maxlen=smoothing)
        self._drowsy_since: float | None = None
        self._face_detector = face_detector or HaarFaceDetector()

    def reset(self) -> None:
        self.frames.clear()
        self.history.clear()
        self._drowsy_since = None

    def process(self, frame_bgr: np.ndarray, now: float | None = None) -> FrameResult:
        """Process one BGR frame. ``now`` (seconds) can be injected for testing."""
        import cv2

        now = time.monotonic() if now is None else now
        face = self._face_detector(frame_bgr)
        if face is None:
            self.reset()
            return FrameResult(None, None, None, "No face detected", False)

        x, y, w, h = face
        crop = cv2.cvtColor(frame_bgr[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        self.frames.append(crop.astype(np.float32) / 255.0)

        if len(self.frames) < self.timesteps:
            return FrameResult(face, None, None, "Collecting frames...", False)

        probs = np.asarray(self.predict_fn(np.stack(self.frames)[None, ...]))[0]
        drowsy_prob = float(probs[DROWSY_INDEX])
        self.history.append(drowsy_prob)
        smoothed = float(np.mean(self.history))

        is_drowsy = smoothed >= self.threshold
        if is_drowsy:
            self._drowsy_since = self._drowsy_since if self._drowsy_since is not None else now
        else:
            self._drowsy_since = None
        alarm = is_drowsy and (now - self._drowsy_since) >= self.alarm_after

        label = CLASS_NAMES[DROWSY_INDEX] if is_drowsy else CLASS_NAMES[1 - DROWSY_INDEX]
        return FrameResult(face, drowsy_prob, smoothed, label, alarm)


class HaarFaceDetector:
    """Largest-face detector using OpenCV's bundled Haar cascade."""

    def __init__(self, min_size: int = 80):
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise OSError(
                f"Could not load the Haar face cascade from {path}. "
                "Install OpenCV 4.x: pip install 'opencv-python>=4.8,<5'"
            )
        self.min_size = min_size

    def __call__(self, frame_bgr: np.ndarray):
        import cv2

        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_size, self.min_size)
        )
        if len(faces) == 0:
            return None
        return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


class Alarm:
    """Non-blocking, rate-limited audible alert (macOS / Windows / Linux)."""

    def __init__(self, cooldown: float = 1.0, enabled: bool = True):
        self.cooldown = cooldown
        self.enabled = enabled
        self._last = 0.0

    def trigger(self) -> None:
        if not self.enabled or time.monotonic() - self._last < self.cooldown:
            return
        self._last = time.monotonic()
        threading.Thread(target=self._beep, daemon=True).start()

    @staticmethod
    def _beep() -> None:
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"], check=False)
            elif system == "Windows":
                import winsound

                winsound.Beep(1500, 400)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass


def draw_overlay(frame: np.ndarray, result: FrameResult, fps: float) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if result.face is not None:
        x, y, fw, fh = result.face
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 160, 0), 2)

    if result.smoothed_prob is None:
        color = (0, 220, 255)
    elif result.label == CLASS_NAMES[DROWSY_INDEX]:
        color = (0, 0, 255)
    else:
        color = (0, 200, 0)

    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, result.label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if result.smoothed_prob is not None:
        bar = int((w - 24) * result.smoothed_prob)
        cv2.rectangle(frame, (12, 44), (w - 12, 58), (60, 60, 60), -1)
        cv2.rectangle(frame, (12, 44), (12 + bar, 58), color, -1)
        cv2.putText(
            frame,
            f"drowsiness {result.smoothed_prob:.0%}",
            (w - 230, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
        )
    cv2.putText(
        frame, f"{fps:4.1f} FPS", (w - 110, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1
    )

    if result.alarm:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
        text = "WAKE UP!"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2, 4)
        cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)
    return frame


def run(
    model_path: str,
    source: str = "0",
    threshold: float = 0.6,
    alarm_after: float = 1.5,
    smoothing: int = 8,
    sound: bool = True,
    display: bool = True,
    output: str | None = None,
) -> None:
    """Run the detector on a webcam index (e.g. ``"0"``) or a video file."""
    import cv2

    from .model import input_spec, load_model

    model = load_model(model_path)
    timesteps, img_size = input_spec(model)
    print(f"Loaded {model_path} (timesteps={timesteps}, img_size={img_size})")

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return model(batch, training=False).numpy()

    detector = DrowsinessDetector(predict_fn, timesteps, img_size, smoothing, threshold, alarm_after)
    alarm = Alarm(enabled=sound)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise OSError(f"Cannot open video source '{source}'")

    writer = None
    fps, last = 0.0, time.monotonic()
    print("Running. Press 'q' to quit." if display else "Running headless...")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.process(frame)
            if result.alarm:
                alarm.trigger()

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            draw_overlay(frame, result, fps)

            if output:
                if writer is None:
                    h, w = frame.shape[:2]
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(frame)
            if display:
                cv2.imshow("Driver Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Annotated video saved to {output}")
        if display:
            cv2.destroyAllWindows()
