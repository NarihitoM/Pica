import unittest
from unittest import mock

from pica import cli
from tests import quiet


class TestParser(unittest.TestCase):
    def setUp(self):
        self.parser = cli.build_parser()

    def test_bare_collect_loops_every_gesture_and_appends(self):
        args = self.parser.parse_args(["collect"])
        self.assertIsNone(args.gesture)
        self.assertFalse(args.replace)

    def test_collect_takes_a_single_gesture_and_a_sample_count(self):
        args = self.parser.parse_args(["collect", "open_palm", "--samples", "50", "--replace"])
        self.assertEqual(args.gesture, "open_palm")
        self.assertEqual(args.samples, 50)
        self.assertTrue(args.replace)

    def test_train_hyperparameters_are_overridable(self):
        args = self.parser.parse_args(["train", "--epochs", "5", "--batch-size", "8"])
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.batch_size, 8)

    def test_a_subcommand_is_required(self):
        with quiet(), self.assertRaises(SystemExit):
            self.parser.parse_args([])

    def test_an_unknown_subcommand_is_rejected(self):
        with quiet(), self.assertRaises(SystemExit):
            self.parser.parse_args(["fly"])


class TestGestureNames(unittest.TestCase):
    def test_lists_every_gesture_the_config_references_once(self):
        names = cli.gesture_names()
        self.assertIn("open_palm", names)
        self.assertEqual(len(names), len(set(names)), "a gesture must not be recorded twice in one loop")

    def test_the_cursor_gesture_comes_first(self):
        config = {
            "cursor": {"gesture": "open_palm"},
            "gestures": {"close_palm": "left_drag"},
            "scroll": {"up_gesture": "three_finger_up", "down_gesture": "open_palm"},
        }
        with mock.patch("pica.utils.config.load_config", return_value=config):
            self.assertEqual(cli.gesture_names(), ["open_palm", "close_palm", "three_finger_up"])


class TestRun(unittest.TestCase):
    def test_refuses_to_start_without_a_trained_model_and_says_how_to_train(self):
        with mock.patch.object(cli, "model_path") as path:
            path.return_value.exists.return_value = False
            with self.assertRaises(SystemExit) as caught:
                cli.cmd_run(mock.Mock())
        self.assertIn("collect", str(caught.exception))
        self.assertIn("train", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
