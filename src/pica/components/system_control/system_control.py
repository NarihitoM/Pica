import time

import numpy as np
import pyautogui

from pica.components.brightness import BrightnessControl

pyautogui.FAILSAFE = True

_MEDIA_ACTIONS = {
    "volume_up": "volumeup",
    "volume_down": "volumedown",
}


class SystemControl:
    """Maps gesture names -> pyautogui actions, using config for the mapping + cooldown."""

    def __init__(self, config: dict):
        self.gesture_actions: dict = config.get("gestures", {})
        self.cursor_cfg: dict = config.get("cursor", {})
        self.scroll_cfg: dict = config.get("scroll", {})
        self.brightness_cfg: dict = config.get("brightness", {})
        self.cooldown = config.get("action_cooldown_seconds", 0.6)

        self._last_fired: dict[str, float] = {}
        self._screen_w, self._screen_h = pyautogui.size()
        self._smoothed_pos: tuple[float, float] | None = None
        self._dragging = False
        self._hold_started: float | None = None
        self.drag_hold = config.get("drag_hold_seconds", 2.0)
        self.click_min_hold = config.get("click_min_hold_seconds", 0.1)
        self._brightness = BrightnessControl(step=self.brightness_cfg.get("step", 10))

    def close(self):
        self._hold_started = None
        self._end_drag()
        self._brightness.stop()

    def handle(self, gesture: str | None, landmarks: np.ndarray | None):
        if self.gesture_actions.get(gesture) == "left_drag":
            if self._hold_started is None:
                self._hold_started = time.time()
            if not self._dragging and time.time() - self._hold_started >= self.drag_hold:
                pyautogui.mouseDown()
                self._dragging = True
            if self._dragging and landmarks is not None:
                self._move_cursor(landmarks)
            return

        self._release_hold()

        if gesture is None or landmarks is None:
            return

        if gesture == self.cursor_cfg.get("gesture"):
            self._move_cursor(landmarks)
            return

        if gesture == self.scroll_cfg.get("up_gesture"):
            pyautogui.scroll(self.scroll_cfg.get("amount", 40))
            return
        if gesture == self.scroll_cfg.get("down_gesture"):
            pyautogui.scroll(-self.scroll_cfg.get("amount", 40))
            return

        action = self.gesture_actions.get(gesture)
        if action is None:
            return
        if not self._ready(gesture):
            return
        self._run_action(action)
        self._last_fired[gesture] = time.time()

    def _release_hold(self):
        """A short hold is a click, a long one was already a drag. Deciding on release is
        what lets the same gesture do both without a separate click gesture."""
        if self._hold_started is None:
            return

        held = time.time() - self._hold_started
        self._hold_started = None
        if self._dragging:
            self._end_drag()
        elif held >= self.click_min_hold:
            pyautogui.click()

    def _end_drag(self):
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

    def _ready(self, gesture: str) -> bool:
        last = self._last_fired.get(gesture, 0.0)
        return (time.time() - last) >= self.cooldown

    def _move_cursor(self, landmarks: np.ndarray):
        idx = self.cursor_cfg.get("landmark_index", 8)
        x_norm, y_norm = landmarks[idx][0], landmarks[idx][1]

        margin = self.cursor_cfg.get("margin", 0.2)
        x_mapped = (x_norm - margin) / (1 - 2 * margin)
        y_mapped = (y_norm - margin) / (1 - 2 * margin)
        x_mapped = min(max(x_mapped, 0.0), 1.0)
        y_mapped = min(max(y_mapped, 0.0), 1.0)

        pad = 5
        target = (
            min(max(x_mapped * self._screen_w, pad), self._screen_w - pad),
            min(max(y_mapped * self._screen_h, pad), self._screen_h - pad),
        )

        smoothing = self.cursor_cfg.get("smoothing", 0.3)
        if self._smoothed_pos is None:
            self._smoothed_pos = target
        else:
            sx, sy = self._smoothed_pos
            tx, ty = target
            self._smoothed_pos = (
                sx + (tx - sx) * (1 - smoothing),
                sy + (ty - sy) * (1 - smoothing),
            )

        pyautogui.moveTo(*self._smoothed_pos, _pause=False)

    def _run_action(self, action: str):
        if action.startswith("hotkey:"):
            pyautogui.hotkey(*action.removeprefix("hotkey:").split(","))
        elif action == "left_click":
            pyautogui.click()
        elif action == "brightness_up":
            self._brightness.up()
        elif action == "brightness_down":
            self._brightness.down()
        elif action in _MEDIA_ACTIONS:
            pyautogui.press(_MEDIA_ACTIONS[action])


def demo():
    cfg = {
        "gestures": {"close_palm": "left_drag"},
        "cursor": {"gesture": "open_palm", "landmark_index": 0, "smoothing": 0.3, "margin": 0.2},
        "brightness": {"step": 10},
        "action_cooldown_seconds": 1.0,
    }
    control = SystemControl(cfg)
    assert control._ready("close_palm") is True
    control._last_fired["close_palm"] = 0.0
    assert control._ready("close_palm") is True
    control._last_fired["close_palm"] = 1e18
    assert control._ready("close_palm") is False

    control._brightness._running = False
    control._brightness._level = 50
    control._run_action("brightness_up")
    assert control._brightness._level == 60
    control._run_action("brightness_down")
    assert control._brightness._level == 50

    pressed = []
    real_hotkey, pyautogui.hotkey = pyautogui.hotkey, lambda *keys: pressed.append(keys)
    try:
        control._run_action("hotkey:ctrl,win,right")
    finally:
        pyautogui.hotkey = real_hotkey
    assert pressed == [("ctrl", "win", "right")], pressed

    control.close()
    print("system_control demo OK")


if __name__ == "__main__":
    demo()
