from pathlib import Path

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_LOGO_HEIGHT = 40
_logo = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets" / "logo.jpg"))
if _logo is not None:
    scale = _LOGO_HEIGHT / _logo.shape[0]
    _logo = cv2.resize(_logo, (int(_logo.shape[1] * scale), _LOGO_HEIGHT))

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


def draw_landmarks(frame, landmarks: np.ndarray):
    h, w = frame.shape[:2]
    points = [(int(x * w), int(y * h)) for x, y, _ in landmarks]

    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 128, 255), -1)


def blur_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for x, y, w, h in _face_cascade.detectMultiScale(gray, 1.1, 5):
        frame[y:y + h, x:x + w] = cv2.GaussianBlur(frame[y:y + h, x:x + w], (51, 51), 0)


def draw_logo(frame, x: int = 10, y: int = 10) -> int:
    """Draws the logo top-left, returns the x where following overlays should start."""
    if _logo is None:
        return x
    h, w = _logo.shape[:2]
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return x
    frame[y:y + h, x:x + w] = _logo
    return x + w + 12


def draw_status(frame, gesture: str | None, confidence: float | None):
    text = "no hand" if gesture is None else f"{gesture} ({confidence:.2f})"
    x = draw_logo(frame)
    cv2.putText(frame, text, (x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def demo():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    lm = np.random.rand(21, 3).astype(np.float32)
    draw_landmarks(frame, lm)
    draw_status(frame, "open_palm", 0.9)
    draw_status(frame, None, None)
    print("visualizer demo OK")


if __name__ == "__main__":
    demo()
