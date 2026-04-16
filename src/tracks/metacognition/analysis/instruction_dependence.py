"""Analysis: instruction dependence across minimal-instruction probes.

Compares behavior under full vs minimal prompts to measure how much
metacognitive behavior is driven by the rubric vs genuinely internalized.
"""
from __future__ import annotations

import numpy as np


def analyze_instruction_dependence(instance_results: list[dict]) -> dict:
    """Analyze instruction dependence across minimal-instruction probes.

    Only includes instances that have instruction_dependence_gap computed.
    """
    gaps = []
    for r in instance_results:
        gap = r.get("computed", {}).get("instruction_dependence_gap")
        if gap is not None:
            gaps.append(gap)

    if not gaps:
        return {"n_probes": 0, "message": "No minimal-instruction probes found"}

    arr = np.array(gaps)
    return {
        "n_probes": len(gaps),
        "mean_gap": round(float(arr.mean()), 4),
        "median_gap": round(float(np.median(arr)), 4),
        "std_gap": round(float(arr.std()), 4),
        "max_gap": round(float(arr.max()), 4),
        "pct_high_dependence": round(float((arr > 0.5).mean()), 4),
        "interpretation": (
            "High instruction dependence" if arr.mean() > 0.4
            else "Moderate instruction dependence" if arr.mean() > 0.2
            else "Low instruction dependence — metacognitive behavior appears internalized"
        ),
    }
