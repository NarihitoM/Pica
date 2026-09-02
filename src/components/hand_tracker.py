from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "hand_landmarker.task")


class HandTracker:
    """Wraps MediaPipe's HandLandmarker (Tasks API): frame in -> list of (21, 3) landmark arrays out."""

    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.4, model_path: str = MODEL_PATH):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.4,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_index = 0

    def process(self, frame_bgr) -> list[np.ndarray]:
        """Returns raw landmarks per hand -- real screen-space coordinates, unmirrored,
        so cursor control always matches actual hand movement direction."""
        rgb = frame_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._frame_index)
        self._frame_index += 1
        return [
            np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            for hand in result.hand_landmarks
        ]

    def process_with_handedness(self, frame_bgr) -> list[tuple[np.ndarray, str]]:
        """Same as process(), but also returns each hand's handedness label
        ('Left'/'Right') for callers that need to normalize for classification."""
        rgb = frame_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._frame_index)
        self._frame_index += 1
        return [
            (
                np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32),
                handedness[0].category_name,
            )
            for hand, handedness in zip(result.hand_landmarks, result.handedness)
        ]

    def close(self):
        self._landmarker.close()


def demo():
    tracker = HandTracker()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = tracker.process(blank)
    assert landmarks == [] or landmarks[0].shape == (21, 3)
    tracker.close()
    print("hand_tracker demo OK")


if __name__ == "__main__":
    demo()
