import unittest

import numpy as np

from pica.utils.landmarks import flatten, normalize


class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.landmarks = np.random.default_rng(0).random((21, 3)).astype(np.float32)

    def test_wrist_moves_to_origin(self):
        self.assertTrue(np.allclose(normalize(self.landmarks)[0], 0, atol=1e-5))

    def test_shape_is_preserved(self):
        self.assertEqual(normalize(self.landmarks).shape, (21, 3))

    def test_same_pose_anywhere_in_frame_normalizes_the_same(self):
        moved = self.landmarks + np.array([0.3, -0.2, 0.1], dtype=np.float32)
        self.assertTrue(np.allclose(normalize(self.landmarks), normalize(moved), atol=1e-5))

    def test_same_pose_at_any_distance_normalizes_the_same(self):
        closer = self.landmarks * 3.0
        self.assertTrue(np.allclose(normalize(self.landmarks), normalize(closer), atol=1e-5))

    def test_a_hand_collapsed_to_one_point_does_not_divide_by_zero(self):
        result = normalize(np.zeros((21, 3), dtype=np.float32))
        self.assertTrue(np.isfinite(result).all())


class TestFlatten(unittest.TestCase):
    def test_produces_the_model_input_width(self):
        flat = flatten(np.zeros((21, 3), dtype=np.float32))
        self.assertEqual(flat.shape, (63,))

    def test_keeps_coordinates_in_landmark_order(self):
        landmarks = np.arange(63, dtype=np.float32).reshape(21, 3)
        self.assertTrue(np.array_equal(flatten(landmarks), np.arange(63, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
