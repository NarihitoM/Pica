import cv2
import numpy as np

from pica.components.hand_tracker import HandTracker
from pica.utils.paths import annotations_dir


def collect(label: str, num_samples: int = 200, camera_id: int = 0, append: bool = True) -> str:
    """Record landmark samples for one gesture. Hold the pose steady, 'q' stops early."""
    tracker = HandTracker()
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError(f"could not open camera {camera_id}")

    samples = []
    try:
        while len(samples) < num_samples:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)
            if hands:
                samples.append(hands[0])
                cv2.putText(frame, f"{label}: {len(samples)}/{num_samples}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
    print("collect demo OK")


if __name__ == "__main__":
    demo()
