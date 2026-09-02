import yaml

from pica.utils.paths import config_path


def load_config(path: str | None = None) -> dict:
    """Single source of truth for gesture -> action mapping, camera id, and tuning."""
    with open(path or config_path(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def demo():
    cfg = load_config()
    assert "gestures" in cfg
    assert "camera_id" in cfg
    print("config demo OK")


if __name__ == "__main__":
    demo()
