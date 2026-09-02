from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "selfie_segmenter.tflite")


class BackgroundBlurrer:
    """Blurs everything MediaPipe's selfie segmenter doesn't classify as the person."""

    def __init__(self, model_path: str = MODEL_PATH):
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._frame_index = 0

    def apply(self, frame_bgr):
        rgb = frame_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment_for_video(mp_image, self._frame_index)
        self._frame_index += 1

        mask = np.squeeze(result.category_mask.numpy_view()) == 0  # 0 = background
        blurred = cv2.GaussianBlur(frame_bgr, (55, 55), 0)
        frame_bgr[mask] = blurred[mask]
        return frame_bgr

    def close(self):
        self._segmenter.close()


def demo():
    blurrer = BackgroundBlurrer()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out = blurrer.apply(blank)
    assert out.shape == (480, 640, 3)
    blurrer.close()
    print("background_blur demo OK")


if __name__ == "__main__":
    demo()
