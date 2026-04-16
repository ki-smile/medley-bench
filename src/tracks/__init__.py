"""MEDLEY-BENCH cognitive ability tracks.

Each track implements a benchmark for one DeepMind cognitive ability.
Tracks are self-contained: prompts, scoring, generation, and analysis.

Available tracks:
    metacognition (v1.0) - Behavioral metacognition under social pressure

Versioning:
    Three independent version numbers ensure result comparability:
    - Library version (medley-bench==X.Y.Z): code changes, bug fixes, new features
    - Dataset version (per track): frozen instance data
    - Scoring version (per track): measures, weights, aggregation

    Results are comparable only when BOTH dataset_version AND scoring_version match.
    Library version changes alone (bug fixes, new providers) do not affect comparability.
"""

import importlib
from typing import Any


AVAILABLE_TRACKS = {
    "metacognition": {
        "module": "src.tracks.metacognition",
        "description": "Behavioral metacognition under social pressure",
        "deepmind_ability": "Metacognition",
        "instances": 130,
        "domains": 5,
    },
}


def get_track_info(track_name: str) -> dict[str, Any]:
    """Get track metadata including version info.

    Returns:
        dict with keys: module, description, deepmind_ability, instances, domains,
        track_version, dataset_version, scoring_version
    """
    if track_name not in AVAILABLE_TRACKS:
        available = ", ".join(AVAILABLE_TRACKS.keys())
        raise ValueError(f"Unknown track '{track_name}'. Available: {available}")

    info = dict(AVAILABLE_TRACKS[track_name])
    track_mod = importlib.import_module(info["module"])

    info["track_version"] = getattr(track_mod, "__version__", "unknown")
    info["dataset_version"] = getattr(track_mod, "DATASET_VERSION", "unknown")
    info["scoring_version"] = getattr(track_mod, "SCORING_VERSION", "unknown")
    return info


def build_result_metadata(track_name: str, model_id: str) -> dict:
    """Build metadata dict to stamp on every result file.

    This metadata ensures any result can be traced back to the exact
    library version, dataset version, and scoring version used.
    """
    import src
    info = get_track_info(track_name)

    return {
        "medley_bench_version": src.__version__,
        "track": track_name,
        "track_version": info["track_version"],
        "dataset_version": info["dataset_version"],
        "scoring_version": info["scoring_version"],
        "model": model_id,
        "comparable_key": f"{track_name}/d{info['dataset_version']}/s{info['scoring_version']}",
    }
