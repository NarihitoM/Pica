import unittest

import numpy as np

from pica.utils import night_mode


def frame(value: int) -> np.ndarray:
    out = np.full((120, 160, 3), value, dtype=np.uint8)
    out[40:80, 60:100] = min(value + 25, 255)
    return out


class TestIsDark(unittest.TestCase):
    def test_a_dim_frame_is_dark(self):
        self.assertTrue(night_mode.is_dark(frame(20)))

    def test_a_well_lit_frame_is_not(self):
        self.assertFalse(night_mode.is_dark(frame(200)))

    def test_the_threshold_is_adjustable(self):
        self.assertFalse(night_mode.is_dark(frame(20), threshold=5))


class TestApply(unittest.TestCase):
    def test_brightens_a_dark_frame(self):
        dark = frame(20)
        self.assertGreater(night_mode.apply(dark).mean(), dark.mean())

    def test_keeps_the_frame_usable_by_the_tracker(self):
        lifted = night_mode.apply(frame(20))
        self.assertEqual(lifted.shape, (120, 160, 3))
        self.assertEqual(lifted.dtype, np.uint8)

    def test_leaves_a_well_lit_frame_alone_on_auto(self):
        bright = frame(200)
        self.assertTrue(np.array_equal(night_mode.apply(bright), bright))

    def test_off_never_touches_the_frame(self):
        dark = frame(20)
        self.assertTrue(np.array_equal(night_mode.apply(dark, mode="off"), dark))

    def test_on_applies_even_when_the_room_is_lit(self):
        bright = frame(200)
        self.assertFalse(np.array_equal(night_mode.apply(bright, mode="on"), bright))


if __name__ == "__main__":
    unittest.main()
