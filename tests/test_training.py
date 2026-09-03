import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pica.components.classifier import GestureClassifier, GestureNet
from pica.pipeline.train import load_dataset


class TestLoadDataset(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.source = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def record(self, label: str, count: int):
        rng = np.random.default_rng(abs(hash(label)) % 2**32)
        np.save(self.source / f"{label}.npy", rng.random((count, 21, 3)).astype(np.float32))

    def test_flattens_every_sample_to_the_model_input_width(self):
        self.record("open_palm", 4)
        self.record("close_palm", 6)
        X, y, labels = load_dataset(self.source)
        self.assertEqual(X.shape, (10, 63))
        self.assertEqual(y.shape, (10,))

    def test_labels_are_sorted_so_class_indices_stay_stable(self):
        self.record("open_palm", 2)
        self.record("close_palm", 2)
        self.record("one_finger_up", 2)
        _, _, labels = load_dataset(self.source)
        self.assertEqual(labels, ["close_palm", "one_finger_up", "open_palm"])

    def test_each_sample_carries_its_own_label_index(self):
        self.record("close_palm", 3)
        self.record("open_palm", 2)
        _, y, labels = load_dataset(self.source)
        self.assertEqual(int((y == labels.index("close_palm")).sum()), 3)
        self.assertEqual(int((y == labels.index("open_palm")).sum()), 2)

    def test_says_what_to_do_when_nothing_has_been_recorded(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_dataset(self.source)
        self.assertIn("collect", str(caught.exception))


class TestGestureClassifier(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.labels = ["close_palm", "open_palm"]
        self.checkpoint = Path(self._tmp.name) / "gesture_classifier.pth"
        model = GestureNet(num_classes=len(self.labels))
        torch.save({"labels": self.labels, "state_dict": model.state_dict()}, self.checkpoint)

    def test_survives_a_save_load_round_trip(self):
        classifier = GestureClassifier(path=self.checkpoint)
        self.assertEqual(classifier.labels, self.labels)

    def test_predicts_a_known_gesture_with_a_usable_confidence(self):
        classifier = GestureClassifier(path=self.checkpoint)
        name, confidence = classifier.predict(np.random.default_rng(0).random((21, 3)).astype(np.float32))
        self.assertIn(name, self.labels)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_a_trained_model_actually_learns_its_gestures(self):
        rng = np.random.default_rng(1)
        poses = [rng.random((21, 3)).astype(np.float32) for _ in self.labels]
        X = np.stack([np.asarray(p).reshape(-1) for p in poses] * 20)
        y = np.array(list(range(len(poses))) * 20, dtype=np.int64)

        model = GestureNet(num_classes=len(self.labels))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(60):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(X)), torch.from_numpy(y))
            loss.backward()
            optimizer.step()

        predicted = model(torch.from_numpy(X)).argmax(dim=1).numpy()
        self.assertTrue((predicted == y).all(), "the net cannot separate two fixed poses")


if __name__ == "__main__":
    unittest.main()
