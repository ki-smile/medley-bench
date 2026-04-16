"""Bundled benchmark data for medley-bench."""
from pathlib import Path

DATA_DIR = Path(__file__).parent


def get_default_data_dir(track: str = "metacognition", version: str = "v1.0") -> Path:
    """Return the path to the bundled dataset for a given track and version."""
    return DATA_DIR / track / version
