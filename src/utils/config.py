import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def demo():
    cfg = load_config()
    assert "gestures" in cfg
    assert "camera_id" in cfg
    print("config demo OK")


if __name__ == "__main__":
    demo()
