import os
import shutil
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_DIR / "default_config.yaml"
HAND_LANDMARKER = PACKAGE_DIR / "assets" / "hand_landmarker.task"
LOGO = PACKAGE_DIR / "assets" / "logo.jpg"


def home() -> Path:
    """Where a user's own config, recordings and trained model live.

    Installed via pip the package directory is read-only, so anything the user
    creates goes here instead. Override with PICA_HOME to keep several sets.
    """
    path = Path(os.environ.get("PICA_HOME") or Path.home() / ".pica")
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    """User config, seeded from the packaged default on first run."""
    path = home() / "config.yaml"
    if not path.exists():
        shutil.copy(DEFAULT_CONFIG, path)
    return path


def annotations_dir() -> Path:
    path = home() / "annotations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path() -> Path:
    return home() / "gesture_classifier.pth"


def demo():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["PICA_HOME"] = tmp
        assert home() == Path(tmp)
        assert not (Path(tmp) / "config.yaml").exists()
        seeded = config_path()
        assert seeded.exists(), "first call must seed the config from the packaged default"
        seeded.write_text("edited", encoding="utf-8")
        assert config_path().read_text(encoding="utf-8") == "edited", "must not overwrite an existing config"
        assert annotations_dir().is_dir()
    os.environ.pop("PICA_HOME", None)

    assert DEFAULT_CONFIG.exists(), DEFAULT_CONFIG
    assert HAND_LANDMARKER.exists(), HAND_LANDMARKER
    assert LOGO.exists(), LOGO
    print("paths demo OK")


if __name__ == "__main__":
    demo()
