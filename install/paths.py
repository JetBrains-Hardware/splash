"""Immutable program files and writable per-user data, for source or release.

SPLASH_DATA, or `splash serve --data-dir`, names one directory the data
Splash writes goes under: installed models (Layout), and unless set
otherwise the Hub cache, temporary files, the persistent cache, the Neural
Engine's calibration, the thinking key and crash traces
(launcher.data_environment). Without it, a release writes under
~/Library/Application Support/Splash and a source checkout under itself.
The locks of the program's own installation and of its ports stay in
RUNTIME, whatever the data directory: an upgrade replaces the program every
data directory runs, and a port is the machine's.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT / "release.json").is_file()
# The frozen server build_server.sh builds: one executable, no interpreter.
FROZEN = bool(getattr(sys, "frozen", False))
DATA = Path.home() / "Library/Application Support/Splash" if PACKAGED else ROOT
MODELS = DATA / "models" if PACKAGED else ROOT / "install/models"
RUNTIME = DATA / "runtime" if PACKAGED else ROOT / "build/runtime"
PYTHON = ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
BINARY = ROOT / ("engine/splash" if PACKAGED else "build/splash")
DATA_VARIABLE = "SPLASH_DATA"


def program(subcommand, *arguments):
    """The command that runs one of Splash's Python programs: PYTHON with
    these arguments (interpreter options, then a script or -m module), or in
    the frozen server, which has no interpreter, this executable's
    subcommand of that name (dev/tools/frozen_server.py)."""
    if FROZEN:
        return [sys.executable, subcommand]
    return [str(PYTHON), *arguments]


@dataclass(frozen=True)
class Layout:
    """The writable directories of one data directory."""

    data: Path
    models: Path

    @property
    def hub_cache(self) -> Path:
        """Where Hub downloads go unless HF_HUB_CACHE or HF_HOME says."""
        return self.data / "huggingface"

    @property
    def temporary(self) -> Path:
        """TMPDIR of every process serving from a custom data directory: the
        SSD cache's unlinked files and the installer's scratch."""
        return self.data / "tmp"


def custom_data_dir():
    """The data directory SPLASH_DATA names, or None."""
    value = os.environ.get(DATA_VARIABLE)
    return Path(value).expanduser().absolute() if value else None


def layout(data=None) -> Layout:
    """The layout of data, a custom data directory, or without one of
    SPLASH_DATA's, or else the default: a release's per-user directory, or
    the source checkout's own folders."""
    if data is None:
        data = custom_data_dir()
    if data is None:
        return Layout(DATA, MODELS)
    data = Path(data).expanduser().absolute()
    return Layout(data, data / "models")
