"""Keep runtime assets outside cloud-managed Documents directories."""

import os
from pathlib import Path

LIVE_HOME = Path(
    os.environ.get("STS_LIVE_HOME", Path.home() / "Library/Application Support/STS")
).expanduser()
ASSETS = LIVE_HOME / "models"
