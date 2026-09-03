import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pica.utils import paths


class TestHome(unittest.TestCase):
    def test_pica_home_overrides_the_default_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"PICA_HOME": tmp}):
                self.assertEqual(paths.home(), Path(tmp))

    def test_falls_back_to_a_dot_pica_folder_in_the_user_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            without_override = {k: v for k, v in os.environ.items() if k != "PICA_HOME"}
            with mock.patch.dict(os.environ, without_override, clear=True):
                with mock.patch.object(Path, "home", return_value=Path(tmp)):
                    self.assertEqual(paths.home(), Path(tmp) / ".pica")

    def test_creates_the_folder_if_it_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "pica"
            with mock.patch.dict(os.environ, {"PICA_HOME": str(target)}):
                self.assertTrue(paths.home().is_dir())


class TestConfig(unittest.TestCase):
    def test_seeds_the_packaged_default_on_first_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"PICA_HOME": tmp}):
                created = paths.config_path()
            self.assertTrue(created.exists())
            self.assertEqual(created.read_text(encoding="utf-8"),
                             paths.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_never_overwrites_a_config_the_user_has_edited(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"PICA_HOME": tmp}):
                paths.config_path().write_text("camera_id: 7\n", encoding="utf-8")
                self.assertEqual(paths.config_path().read_text(encoding="utf-8"), "camera_id: 7\n")


class TestUserData(unittest.TestCase):
    def test_recordings_and_model_live_under_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"PICA_HOME": tmp}):
                self.assertEqual(paths.annotations_dir(), Path(tmp) / "annotations")
                self.assertEqual(paths.model_path(), Path(tmp) / "gesture_classifier.pth")

    def test_packaged_assets_ship_with_the_install(self):
        self.assertTrue(paths.DEFAULT_CONFIG.exists())
        self.assertTrue(paths.HAND_LANDMARKER.exists())


if __name__ == "__main__":
    unittest.main()
