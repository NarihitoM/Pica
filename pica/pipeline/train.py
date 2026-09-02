import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pica.components.classifier import GestureNet
from pica.utils.landmarks import flatten, normalize
from pica.utils.paths import annotations_dir, model_path


def load_dataset(source_dir=None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Every annotations/<label>.npy becomes one class, ordered by name."""
    source_dir = source_dir or annotations_dir()
    labels = sorted(p.stem for p in source_dir.glob("*.npy"))
    if not labels:
        raise FileNotFoundError(f"no recordings in {source_dir} -- run 'narihito-pica collect <gesture>' first")

    X, y = [], []
    for index, label in enumerate(labels):
        for landmarks in np.load(source_dir / f"{label}.npy"):
            X.append(flatten(normalize(landmarks)))
            y.append(index)

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64), labels


def train(epochs: int = 30, batch_size: int = 32, lr: float = 1e-3, out_path=None) -> str:
    X, y, labels = load_dataset()
    print(f"labels: {labels}")
    print(f"samples: {X.shape[0]}")

    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=batch_size, shuffle=True)
    model = GestureNet(num_classes=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch}: loss={total / len(loader):.4f}")

    out_path = out_path or model_path()
    torch.save({"labels": labels, "state_dict": model.state_dict()}, out_path)
    print(f"saved {out_path}")
    return str(out_path)


def demo():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        for label in ("open_palm", "close_palm"):
            np.save(source / f"{label}.npy", np.random.rand(8, 21, 3).astype(np.float32))

        X, y, labels = load_dataset(source)
        assert labels == ["close_palm", "open_palm"], "labels must be sorted so class order is stable"
        assert X.shape == (16, 63)
        assert set(y.tolist()) == {0, 1}

        try:
            load_dataset(Path(tmp) / "empty")
            raise AssertionError("expected FileNotFoundError for an empty annotations dir")
        except FileNotFoundError:
            pass
    print("train demo OK")


if __name__ == "__main__":
    demo()
