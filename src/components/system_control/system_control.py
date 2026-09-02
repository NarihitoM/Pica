import time

import numpy as np
import pyautogui

from src.components.brightness import BrightnessControl

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
        self._brightness = BrightnessControl(step=self.brightness_cfg.get("step", 5))

    def handle_two_hands(self, left: np.ndarray, right: np.ndarray) -> float:
        """Distance between the two wrists sets brightness -- further apart is brighter."""
        distance = float(np.linalg.norm(left[0][:2] - right[0][:2]))
        percent = self._brightness_percent(distance)
        self._brightness.set(percent)
        return percent

    def _brightness_percent(self, distance: float) -> float:
        near = self.brightness_cfg.get("min_distance", 0.15)
        far = self.brightness_cfg.get("max_distance", 0.6)
        ratio = (distance - near) / (far - near)
        return min(max(ratio, 0.0), 1.0) * 100

    def close(self):
        self._brightness.stop()

    def handle(self, gesture: str, landmarks: np.ndarray):
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
        if action == "left_click":
            pyautogui.click()
        elif action in _MEDIA_ACTIONS:
            pyautogui.press(_MEDIA_ACTIONS[action])


def demo():
    cfg = {
        "gestures": {"close_palm": "left_click"},
        "cursor": {"gesture": "open_palm", "landmark_index": 0, "smoothing": 0.3, "margin": 0.2},
        "brightness": {"min_distance": 0.15, "max_distance": 0.6},
        "action_cooldown_seconds": 1.0,
    }
    control = SystemControl(cfg)
    assert control._ready("close_palm") is True
    control._last_fired["close_palm"] = 0.0
    assert control._ready("close_palm") is True
    control._last_fired["close_palm"] = 1e18
    assert control._ready("close_palm") is False

    assert control._brightness_percent(0.10) == 0, "hands together -> dimmest"
    assert control._brightness_percent(0.90) == 100, "hands wide apart -> brightest"
    assert 45 < control._brightness_percent(0.375) < 55, "midpoint -> mid brightness"

    control.close()
    print("system_control demo OK")


if __name__ == "__main__":
    demo()
