from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pica.utils.landmarks import flatten, normalize
from pica.utils.paths import model_path

INPUT_DIM = 63


class GestureNet(nn.Module):
    """Small MLP -- 63 landmark coordinates in, one score per trained gesture out.
    Retrained by `narihito-pica train` whenever gestures are added or re-recorded."""

    def __init__(self, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class GestureClassifier:
    """Loads a trained GestureNet and predicts gesture name + confidence from landmarks."""

    def __init__(self, path: str | Path | None = None):
        checkpoint = torch.load(path or model_path(), map_location="cpu", weights_only=True)
        self.labels: list[str] = checkpoint["labels"]
        self.model = GestureNet(num_classes=len(self.labels))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        flat = flatten(normalize(landmarks))
        logits = self.model(torch.from_numpy(flat).float().unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs))
        return self.labels[idx], float(probs[idx])


def demo():
    import tempfile

    labels = ["open_palm", "close_palm"]
    model = GestureNet(num_classes=len(labels))
    tmp_path = Path(tempfile.gettempdir()) / "_pica_demo_classifier.pth"
    torch.save({"labels": labels, "state_dict": model.state_dict()}, tmp_path)

    clf = GestureClassifier(path=tmp_path)
    dummy = np.random.rand(21, 3).astype(np.float32)
    name, confidence = clf.predict(dummy)
    assert name in labels
    assert 0.0 <= confidence <= 1.0

    tmp_path.unlink()
    print("classifier demo OK")


if __name__ == "__main__":
    demo()
