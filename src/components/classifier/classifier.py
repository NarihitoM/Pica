from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils.landmarks import flatten, normalize

INPUT_DIM = 63
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[3] / "models" / "gesture_classifier.pth")


class GestureNet(nn.Module):
    """Small MLP -- 63 landmark coordinates in, one score per trained gesture out.
    Retrained by notebooks/training_model.ipynb whenever gestures are added or re-recorded."""

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

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.labels: list[str] = checkpoint["labels"]
        self.model = GestureNet(num_classes=len(self.labels))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        x = flatten(normalize(landmarks))
        x = torch.from_numpy(x).float().unsqueeze(0)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs))
        return self.labels[idx], float(probs[idx])


def demo():
    labels = ["open_palm", "close_palm"]
    model = GestureNet(num_classes=len(labels))
    tmp_path = Path("models/_demo_classifier.pth")
    torch.save({"labels": labels, "state_dict": model.state_dict()}, tmp_path)

    clf = GestureClassifier(model_path=str(tmp_path))
    dummy = np.random.rand(21, 3).astype(np.float32)
    name, confidence = clf.predict(dummy)
    assert name in labels
    assert 0.0 <= confidence <= 1.0

    tmp_path.unlink()
    print("classifier demo OK")


if __name__ == "__main__":
    demo()
