import time

import numpy as np
import pyautogui

from src.components.brightness import BrightnessControl

pyautogui.FAILSAFE = True

_SCROLL_INTERVAL = 1 / 30
_SMOOTHING_REFERENCE_FPS = 60

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
        self._last_move = time.time()
        self._brightness = BrightnessControl(step=self.brightness_cfg.get("step", 10))

    def close(self):
        self._end_drag()
        self._brightness.stop()

    def handle(self, gesture: str | None, landmarks: np.ndarray | None):
        if self.gesture_actions.get(gesture) == "left_drag":
            if not self._dragging:
                pyautogui.mouseDown()
                self._dragging = True
            if landmarks is not None:
                self._move_cursor(landmarks)
            return

        self._end_drag()

        if gesture is None or landmarks is None:
            return

        if gesture == self.cursor_cfg.get("gesture"):
            self._move_cursor(landmarks)
            return

        scroll_up = gesture == self.scroll_cfg.get("up_gesture")
        scroll_down = gesture == self.scroll_cfg.get("down_gesture")
        if scroll_up or scroll_down:
            if time.time() - self._last_fired.get("scroll", 0.0) >= _SCROLL_INTERVAL:
                amount = self.scroll_cfg.get("amount", 40)
                pyautogui.scroll(amount if scroll_up else -amount)
                self._last_fired["scroll"] = time.time()
            return

        action = self.gesture_actions.get(gesture)
        if action is None:
            return
        if not self._ready(gesture):
            return
        self._run_action(action)
        self._last_fired[gesture] = time.time()

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

        now = time.time()
        dt = min(now - self._last_move, 0.1)
        self._last_move = now

        smoothing = self.cursor_cfg.get("smoothing", 0.3)
        if self._smoothed_pos is None or smoothing <= 0:
            self._smoothed_pos = target
        else:
            alpha = 1 - smoothing ** (dt * _SMOOTHING_REFERENCE_FPS)
            sx, sy = self._smoothed_pos
            tx, ty = target
            self._smoothed_pos = (sx + (tx - sx) * alpha, sy + (ty - sy) * alpha)

        pyautogui.moveTo(*self._smoothed_pos, _pause=False)

    def _run_action(self, action: str):
        if action == "left_click":
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

    corner = np.zeros((21, 3), dtype=np.float32)
    control.cursor_cfg["smoothing"] = 0.0
    control._smoothed_pos = (500.0, 500.0)
    control._last_move = time.time()
    control._move_cursor(corner)
    snapped = control._smoothed_pos

    control.cursor_cfg["smoothing"] = 0.5
    control._smoothed_pos = (500.0, 500.0)
    control._last_move = time.time() - 1 / 600
    control._move_cursor(corner)
    small_step = control._smoothed_pos

    control._smoothed_pos = (500.0, 500.0)
    control._last_move = time.time() - 1 / 20
    control._move_cursor(corner)
    big_step = control._smoothed_pos

    assert abs(small_step[0] - 500.0) < abs(big_step[0] - 500.0) < abs(snapped[0] - 500.0), (
        "a longer gap must move the cursor further, and smoothing 0 must snap all the way"
    )

    control._brightness._running = False
    control._brightness._level = 50
    control._run_action("brightness_up")
    assert control._brightness._level == 60
    control._run_action("brightness_down")
    assert control._brightness._level == 50

    control.close()
    print("system_control demo OK")


if __name__ == "__main__":
    demo()
