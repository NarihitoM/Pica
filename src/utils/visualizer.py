import cv2
import numpy as np

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


def draw_status(frame, gesture: str | None, confidence: float | None):
    text = "no hand" if gesture is None else f"{gesture} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def demo():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    lm = np.random.rand(21, 3).astype(np.float32)
    draw_landmarks(frame, lm)
    draw_status(frame, "open_palm", 0.9)
    draw_status(frame, None, None)
    print("visualizer demo OK")


if __name__ == "__main__":
    demo()
