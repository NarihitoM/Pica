import unittest

from pica.pipeline.run_app import combine


class CombineTestCase(unittest.TestCase):
    def test_one_hand_passes_through(self):
        self.assertEqual(combine([("open_palm", 0.9)]), ("open_palm", 0.9))

    def test_two_hands_on_one_pose_compose(self):
        self.assertEqual(
            combine([("open_palm", 0.9), ("open_palm", 0.7)]),
            ("two_hand_open_palm", 0.7),
            "the pair must carry the weaker confidence of the two hands",
        )

    def test_two_hands_disagreeing_fall_back_to_the_first(self):
        self.assertEqual(
            combine([("open_palm", 0.9), ("close_palm", 0.8)]),
            ("open_palm", 0.9),
        )

    def test_the_pair_is_not_the_cursor_gesture(self):
        cursor_gesture = "open_palm"
        paired, _ = combine([("open_palm", 0.9), ("open_palm", 0.9)])
        self.assertNotEqual(paired, cursor_gesture, "two open palms must not drive the cursor")


if __name__ == "__main__":
    unittest.main()
