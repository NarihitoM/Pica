import numpy as np


def normalize(landmarks: np.ndarray) -> np.ndarray:
    """Translate to wrist origin, scale by hand size. (21, 3) -> (21, 3)."""
    wrist = landmarks[0]
    centered = landmarks - wrist
    scale = np.linalg.norm(centered).clip(min=1e-6)
    return centered / scale


def flatten(landmarks: np.ndarray) -> np.ndarray:
    """(21, 3) -> (63,) for model input."""
    return landmarks.reshape(-1)


def demo():
    lm = np.random.rand(21, 3).astype(np.float32)
    n = normalize(lm)
    assert n.shape == (21, 3)
    assert np.allclose(n[0], 0, atol=1e-5)
    flat = flatten(n)
    assert flat.shape == (63,)
    print("landmarks demo OK")


if __name__ == "__main__":
    demo()
