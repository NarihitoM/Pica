import cv2
import numpy as np

from pica.components.hand_tracker import HandTracker
from pica.utils.paths import annotations_dir
from pica.utils.visualizer import draw_landmarks

_GREEN = (120, 255, 120)
_AMBER = (80, 200, 255)
_WHITE = (255, 255, 255)
_DIM = (180, 180, 180)


def draw_guide(frame, label: str, collected: int, total: int, hand_visible: bool,
               position: tuple[int, int] | None = None, next_label: str | None = None):
    """Tells the user which pose to hold and whether it is actually being recorded --
    without this the window is just a webcam feed with no hint of what to do."""
    height, width = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 96), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 40), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    step = f"[{position[0]}/{position[1]}]  " if position else ""
    cv2.putText(frame, f"{step}HOLD THIS GESTURE:  {label.replace('_', ' ').upper()}", (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2)

    pose = label.replace("_", " ")
    if hand_visible:
        status, color = f"recording... {collected}/{total}", _GREEN
    else:
        status, color = f"show {pose} to start", _AMBER
    cv2.putText(frame, status, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    bar_width = width - 32
    filled = int(bar_width * collected / total) if total else 0
    cv2.rectangle(frame, (16, 76), (16 + bar_width, 86), (70, 70, 70), -1)
    if filled:
        cv2.rectangle(frame, (16, 76), (16 + filled, 86), _GREEN, -1)

    footer = "move your hand around a little  |  press q to stop"
    if next_label:
        footer = f"next: {next_label.replace('_', ' ')}  |  {footer}"
    cv2.putText(frame, footer, (16, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _DIM, 1)


def collect(label: str, num_samples: int = 200, camera_id: int = 0, append: bool = True,
            position: tuple[int, int] | None = None, next_label: str | None = None) -> str:
    """Record landmark samples for one gesture. Hold the pose steady, 'q' stops early."""
    tracker = HandTracker()
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError(f"could not open camera {camera_id}")

    print(f"recording '{label}' -- hold the gesture, press q in the window to stop")
    samples: list[np.ndarray] = []
    try:
        while len(samples) < num_samples:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)
            if hands:
                samples.append(hands[0])
                draw_landmarks(frame, hands[0])

            draw_guide(frame, label, len(samples), num_samples, bool(hands), position, next_label)
            cv2.imshow("collect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    return save(label, np.array(samples, dtype=np.float32), append=append)


def save(label: str, samples: np.ndarray, append: bool = True, target_dir=None) -> str:
    """Appends by default -- re-recording a gesture adds to it unless append is False."""
    target_dir = target_dir or annotations_dir()
    out_path = target_dir / f"{label}.npy"
    if append and out_path.exists():
        samples = np.concatenate([np.load(out_path), samples], axis=0)

    np.save(out_path, samples)
    print(f"saved {samples.shape[0]} samples to {out_path}")
    return str(out_path)


def demo():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        batch = np.random.rand(5, 21, 3).astype(np.float32)

        save("open_palm", batch, target_dir=target)
        assert np.load(target / "open_palm.npy").shape == (5, 21, 3)

        save("open_palm", batch, target_dir=target)
        assert np.load(target / "open_palm.npy").shape == (10, 21, 3), "append must grow the file"

        save("open_palm", batch, append=False, target_dir=target)
        assert np.load(target / "open_palm.npy").shape == (5, 21, 3), "append=False must overwrite"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_guide(frame, "open_palm", 0, 200, hand_visible=False)
    assert frame.any(), "guide must draw something when no hand is visible"

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_guide(blank, "open_palm", 200, 200, hand_visible=True)
    assert not np.array_equal(frame, blank), "the guide must change once samples are recorded"

    stepped = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_guide(stepped, "open_palm", 0, 200, False, position=(2, 8), next_label="close_palm")
    assert not np.array_equal(frame, stepped), "step counter and next gesture must be drawn"
    print("collect demo OK")


if __name__ == "__main__":
    demo()
