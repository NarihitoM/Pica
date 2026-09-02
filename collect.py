import argparse
import os

import cv2
import numpy as np

from src.components.hand_tracker import HandTracker

DATA_DIR = "data/annotations"


def collect(label: str, num_samples: int = 150, camera_id: int = 0):
    os.makedirs(DATA_DIR, exist_ok=True)
    tracker = HandTracker()
    cap = cv2.VideoCapture(camera_id)
    samples = []

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
        cv2.imshow("collect (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()

    out_path = f"{DATA_DIR}/{label}.npy"
    arr = np.array(samples, dtype=np.float32)
    if os.path.exists(out_path):
        arr = np.concatenate([np.load(out_path), arr], axis=0)
    np.save(out_path, arr)
    print(f"saved {arr.shape[0]} total samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--camera-id", type=int, default=0)
    args = parser.parse_args()
    collect(args.label, args.num_samples, args.camera_id)
