import yaml

from pica.utils.paths import DEFAULT_CONFIG, config_path


def fill_missing(config: dict, defaults: dict) -> dict:
    """Adds keys a newer release introduced, leaving everything already set alone.

    Your config is copied once and never overwritten, which keeps your tuning safe but
    also means a gesture added in a later version would never show up. This closes that
    gap without touching a single value you chose.
    """
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict) and isinstance(config[key], dict):
            fill_missing(config[key], value)
    return config


def load_config(path: str | None = None) -> dict:
    """Single source of truth for gesture -> action mapping, camera id, and tuning."""
    with open(path or config_path(), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f) or {}
    return fill_missing(config, defaults)


def demo():
    cfg = load_config()
    assert "gestures" in cfg
    assert "camera_id" in cfg

    mine = {"camera_id": 2, "gestures": {"close_palm": "left_drag"}}
    merged = fill_missing(mine, {"camera_id": 0, "confidence_threshold": 0.75,
                                 "gestures": {"close_palm": "left_click", "four_finger_up": "hotkey:win"}})
    assert merged["camera_id"] == 2, "a value you set must win"
    assert merged["gestures"]["close_palm"] == "left_drag", "a nested value you set must win"
    assert merged["confidence_threshold"] == 0.75, "a missing key must be filled in"
    assert merged["gestures"]["four_finger_up"] == "hotkey:win", "a new gesture must be filled in"

    assert fill_missing({"gestures": None}, {"gestures": {"a": "b"}})["gestures"] is None, \
        "an explicit null must not be replaced by the default block"

    print("config demo OK")


if __name__ == "__main__":
    demo()
