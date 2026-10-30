import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from utils import Hand

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "hand_landmarker.task"


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading hand model to {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    def __init__(self, max_hands: int = 2):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_ms = 0

    def detect(self, frame_bgr: np.ndarray) -> list:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

        stamp = max(int(time.monotonic() * 1000), self.last_ms + 1)
        self.last_ms = stamp
        result = self.detector.detect_for_video(mp_image, stamp)

        h, w = frame_bgr.shape[:2]
        hands = []
        for points in result.hand_landmarks:
            norm = np.array([[p.x, p.y] for p in points], dtype=np.float32)
            lm = np.array([[p.x * w, p.y * h, p.z * w] for p in points], dtype=np.float32)
            hands.append(Hand(lm, norm))
        return hands
