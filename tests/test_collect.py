import tempfile
import unittest
from pathlib import Path

import numpy as np

from pica.pipeline.collect import draw_guide, save
from tests import quiet


def batch(n: int = 5) -> np.ndarray:
    return np.random.default_rng(0).random((n, 21, 3)).astype(np.float32)


class TestSave(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.target = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_writes_a_npy_named_after_the_gesture(self):
        with quiet():
            out = save("open_palm", batch(), target_dir=self.target)
        self.assertEqual(Path(out), self.target / "open_palm.npy")
        self.assertEqual(np.load(out).shape, (5, 21, 3))

    def test_recording_again_adds_to_what_is_already_there(self):
        with quiet():
            save("open_palm", batch(), target_dir=self.target)
        with quiet():
            save("open_palm", batch(), target_dir=self.target)
        self.assertEqual(np.load(self.target / "open_palm.npy").shape, (10, 21, 3))

    def test_replace_throws_the_old_recording_away(self):
        with quiet():
            save("open_palm", batch(), target_dir=self.target)
        with quiet():
            save("open_palm", batch(3), append=False, target_dir=self.target)
        self.assertEqual(np.load(self.target / "open_palm.npy").shape, (3, 21, 3))


class TestDrawGuide(unittest.TestCase):
    def frame(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draws_something_before_a_hand_is_seen(self):
        frame = self.frame()
        draw_guide(frame, "open_palm", 0, 200, hand_visible=False)
        self.assertTrue(frame.any(), "an empty window would tell the user nothing")

    def test_looks_different_once_it_is_recording(self):
        waiting, recording = self.frame(), self.frame()
        draw_guide(waiting, "open_palm", 0, 200, hand_visible=False)
        draw_guide(recording, "open_palm", 120, 200, hand_visible=True)
        self.assertFalse(np.array_equal(waiting, recording))

    def test_progress_bar_grows_with_the_sample_count(self):
        early, late = self.frame(), self.frame()
        draw_guide(early, "open_palm", 10, 200, hand_visible=True)
        draw_guide(late, "open_palm", 190, 200, hand_visible=True)
        self.assertGreater(int(late[76:86].sum()), int(early[76:86].sum()))

    def test_loop_mode_adds_the_step_counter_and_next_gesture(self):
        single, looped = self.frame(), self.frame()
        draw_guide(single, "open_palm", 0, 200, False)
        draw_guide(looped, "open_palm", 0, 200, False, position=(1, 8), next_label="close_palm")
        self.assertFalse(np.array_equal(single, looped))

    def test_a_zero_sample_run_does_not_divide_by_zero(self):
        draw_guide(self.frame(), "open_palm", 0, 0, hand_visible=False)


if __name__ == "__main__":
    unittest.main()
