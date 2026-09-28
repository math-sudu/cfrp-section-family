"""Paths for the self-contained manuscript research code and data."""
from pathlib import Path
import sys

TUST_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = TUST_ROOT / "src"
RESULTS_DIR = TUST_ROOT / "results/analysis"


def ensure_local_importable() -> Path:
    """Make the bundled numerical modules available to script entry points."""
    source = str(SOURCE_DIR)
    if source not in sys.path:
        sys.path.insert(0, source)
    return TUST_ROOT
