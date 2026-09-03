import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def is_dark(frame: np.ndarray, threshold: int = 60) -> bool:
    """A frame this dim usually means MediaPipe is about to lose the hand entirely."""
    return float(frame.mean()) < threshold


def enhance(frame: np.ndarray) -> np.ndarray:
    """Lift a dim frame so the hand still has edges to track.

    Equalizes lightness only, in LAB, so colour stays put -- a plain gain on BGR
    washes skin tone out and MediaPipe does worse, not better.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)


def apply(frame: np.ndarray, mode: str = "auto", threshold: int = 60) -> np.ndarray:
    """mode: 'auto' only when the frame is dark, 'on' always, 'off' never."""
    if mode == "off":
        return frame
    if mode == "auto" and not is_dark(frame, threshold):
        return frame
    return enhance(frame)


def demo():
    dark = np.full((120, 160, 3), 20, dtype=np.uint8)
    dark[40:80, 60:100] = 45

    assert is_dark(dark)
    assert not is_dark(np.full((120, 160, 3), 200, dtype=np.uint8))

    lifted = apply(dark)
    assert lifted.mean() > dark.mean(), "night mode must brighten a dark frame"
    assert lifted.shape == dark.shape and lifted.dtype == dark.dtype

    bright = np.full((120, 160, 3), 200, dtype=np.uint8)
    assert np.array_equal(apply(bright), bright), "a well lit frame must pass through untouched"
    assert np.array_equal(apply(dark, mode="off"), dark), "off must never touch the frame"
    assert not np.array_equal(apply(bright, mode="on"), bright), "on must apply regardless"
    print("night_mode demo OK")


if __name__ == "__main__":
    demo()
