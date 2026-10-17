"""install/paths.py: the data directory and the layout beneath it."""

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import launcher, paths


class LayoutTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_a_custom_data_directory_holds_the_data(self):
        layout = paths.layout(self.root / "data")
        data = (self.root / "data").absolute()
        self.assertEqual(layout.data, data)
        self.assertEqual(layout.models, data / "models")
        self.assertEqual(layout.hub_cache, data / "huggingface")
        self.assertEqual(layout.temporary, data / "tmp")
        with mock.patch.dict(
            os.environ,
            {paths.DATA_VARIABLE: str(self.root / "env"), "HOME": str(self.root)},
        ):
            self.assertEqual(paths.layout().data, self.root / "env")
            self.assertEqual(paths.layout("~/tilde").data, self.root / "tilde")
            self.assertEqual(paths.custom_data_dir(), self.root / "env")
        with mock.patch.dict(os.environ, {paths.DATA_VARIABLE: ""}):
            self.assertIsNone(paths.custom_data_dir())

    def test_the_default_layout_is_the_program_data_directory(self):
        with mock.patch.dict(os.environ, {paths.DATA_VARIABLE: ""}):
            self.assertEqual(paths.layout(), paths.Layout(paths.DATA, paths.MODELS))

    def test_the_environment_of_a_custom_data_directory_gathers_the_caches(self):
        layout = paths.layout(self.root / "data")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher.data_environment(layout, custom=False), {})
            self.assertEqual(
                launcher.data_environment(layout, custom=True),
                {
                    paths.DATA_VARIABLE: str(layout.data),
                    "TMPDIR": str(layout.temporary),
                    "HF_HUB_CACHE": str(layout.hub_cache),
                },
            )
        # A Hub cache the user set explicitly stands.
        for variable in ("HF_HUB_CACHE", "HF_HOME"):
            with mock.patch.dict(os.environ, {variable: "/hf"}, clear=True):
                self.assertNotIn(
                    "HF_HUB_CACHE", launcher.data_environment(layout, custom=True)
                )

    def test_the_server_keeps_its_key_and_traces_under_the_data_directory(self):
        from server import crash_trace, thinking

        data = self.root / "data"
        # The trace directory is read at import; the default comes back after.
        self.addCleanup(importlib.reload, crash_trace)
        with mock.patch.dict(os.environ, {paths.DATA_VARIABLE: str(data)}):
            thinking.load_thinking_key()
            importlib.reload(crash_trace)
        self.assertTrue((data / "thinking.key").is_file())
        self.assertEqual(crash_trace.DEFAULT_TRACE_DIRECTORY, data / "logs" / "crash")


if __name__ == "__main__":
    unittest.main()
