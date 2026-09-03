import unittest
from unittest import mock

import numpy as np

from pica.components.system_control import system_control as module

CONFIG = {
    "gestures": {"close_palm": "left_drag", "two_finger_up": "volume_up", "one_finger_up": "left_click"},
    "cursor": {"gesture": "open_palm", "landmark_index": 0, "smoothing": 0.0, "margin": 0.2},
    "scroll": {"up_gesture": "three_finger_up", "down_gesture": "three_finger_down", "amount": 100},
    "brightness": {"step": 20},
    "action_cooldown_seconds": 0.6,
}


def landmarks_at(x: float, y: float) -> np.ndarray:
    return np.full((21, 3), [x, y, 0.0], dtype=np.float32)


class ControlTestCase(unittest.TestCase):
    def setUp(self):
        self.gui = mock.patch.object(module, "pyautogui").start()
        self.gui.size.return_value = (1000, 800)
        mock.patch.object(module, "BrightnessControl").start()
        self.addCleanup(mock.patch.stopall)
        self.control = module.SystemControl(CONFIG)


class TestCursor(ControlTestCase):
    def moved_to(self) -> tuple[float, float]:
        args = self.gui.moveTo.call_args.args
        return args[0], args[1]

    def test_centre_of_the_frame_maps_to_the_centre_of_the_screen(self):
        self.control.handle("open_palm", landmarks_at(0.5, 0.5))
        x, y = self.moved_to()
        self.assertAlmostEqual(x, 500.0, places=3)
        self.assertAlmostEqual(y, 400.0, places=3)

    def test_the_margin_lets_you_reach_the_edges_without_leaving_frame(self):
        self.control.handle("open_palm", landmarks_at(0.2, 0.2))
        x, y = self.moved_to()
        self.assertAlmostEqual(x, 5.0, places=3)
        self.assertAlmostEqual(y, 5.0, places=3)

    def test_a_hand_outside_the_margin_is_clamped_on_screen(self):
        self.control.handle("open_palm", landmarks_at(1.0, 1.0))
        x, y = self.moved_to()
        self.assertLessEqual(x, 995.0)
        self.assertLessEqual(y, 795.0)

    def test_smoothing_lags_the_cursor_behind_the_hand(self):
        self.control.cursor_cfg = dict(CONFIG["cursor"], smoothing=0.8)
        self.control.handle("open_palm", landmarks_at(0.2, 0.2))
        self.control.handle("open_palm", landmarks_at(0.8, 0.8))
        x, _ = self.moved_to()
        self.assertLess(x, 995.0, "a smoothed cursor must not jump straight to the target")
        self.assertGreater(x, 5.0, "but it must still move toward it")


class TestDrag(ControlTestCase):
    def test_holding_the_gesture_presses_and_keeps_the_button_down(self):
        self.control.handle("close_palm", landmarks_at(0.5, 0.5))
        self.control.handle("close_palm", landmarks_at(0.6, 0.6))
        self.gui.mouseDown.assert_called_once()
        self.gui.mouseUp.assert_not_called()

    def test_the_window_follows_the_hand_while_dragging(self):
        self.control.handle("close_palm", landmarks_at(0.5, 0.5))
        self.assertTrue(self.gui.moveTo.called)

    def test_releasing_the_gesture_releases_the_button(self):
        self.control.handle("close_palm", landmarks_at(0.5, 0.5))
        self.control.handle("open_palm", landmarks_at(0.5, 0.5))
        self.gui.mouseUp.assert_called_once()

    def test_losing_the_hand_entirely_still_releases_the_button(self):
        self.control.handle("close_palm", landmarks_at(0.5, 0.5))
        self.control.handle(None, None)
        self.gui.mouseUp.assert_called_once()

    def test_closing_down_never_leaves_the_button_stuck(self):
        self.control.handle("close_palm", landmarks_at(0.5, 0.5))
        self.control.close()
        self.gui.mouseUp.assert_called_once()


class TestScroll(ControlTestCase):
    def test_scrolls_up_and_down_by_the_configured_amount(self):
        self.control.handle("three_finger_up", landmarks_at(0.5, 0.5))
        self.gui.scroll.assert_called_with(100)
        self.control.handle("three_finger_down", landmarks_at(0.5, 0.5))
        self.gui.scroll.assert_called_with(-100)

    def test_scrolling_is_not_rate_limited(self):
        for _ in range(3):
            self.control.handle("three_finger_up", landmarks_at(0.5, 0.5))
        self.assertEqual(self.gui.scroll.call_count, 3)


class TestCooldown(ControlTestCase):
    def test_a_held_gesture_fires_once_not_every_frame(self):
        for _ in range(5):
            self.control.handle("one_finger_up", landmarks_at(0.5, 0.5))
        self.gui.click.assert_called_once()

    def test_it_fires_again_once_the_cooldown_has_passed(self):
        self.control.handle("one_finger_up", landmarks_at(0.5, 0.5))
        self.control._last_fired["one_finger_up"] = 0.0
        self.control.handle("one_finger_up", landmarks_at(0.5, 0.5))
        self.assertEqual(self.gui.click.call_count, 2)

    def test_an_unmapped_gesture_does_nothing(self):
        self.control.handle("no_such_gesture", landmarks_at(0.5, 0.5))
        self.gui.click.assert_not_called()
        self.gui.press.assert_not_called()

    def test_media_actions_press_the_matching_key(self):
        self.control.handle("two_finger_up", landmarks_at(0.5, 0.5))
        self.gui.press.assert_called_once_with("volumeup")


if __name__ == "__main__":
    unittest.main()
