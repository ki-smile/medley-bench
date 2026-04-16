"""Step B-Social: Social Revision prompt (hybrid Phase 1 + Phase 2).

Model sees its Step A response, analyst outputs (with masked confidences), and consensus.
Phase 1: Revised assessment explaining what changed and why.
Phase 2: Revised confidence on same claims with specific citations.
Anti-rhetoric rubric included.

v4: Analyst confidences masked. Consensus shown as directional summary only
(no numeric values). JSON output streamlined to avoid token waste.
"""
from __future__ import annotations
import json
import re

from src.admin.generation.prompts import content_hash

PROMPT_VERSION = 4

# Confidence levels considered "supporting" a claim
_SUPPORT = {"very_high", "high"}
_OPPOSE = {"very_low", "low"}
_UNCERTAIN = {"moderate"}


def mask_analyst_confidences(analyst_outputs_json: str) -> str:
    """Replace numeric/categorical confidence levels with directional labels.

    Input: JSON string of analyst outputs (list of dicts with claim_level_assessments).
    Output: JSON string with confidence replaced by "supports" / "opposes" / "uncertain".

    This prevents the benchmarked model from copying analyst confidence values.
    The model must form its own confidence from the argument content.
    """
    try:
        outputs = json.loads(analyst_outputs_json) if isinstance(analyst_outputs_json, str) else analyst_outputs_json
    except (json.JSONDecodeError, TypeError):
        return analyst_outputs_json  # Return as-is if unparseable

    for out in outputs:
        resp = out.get("response", {})
        if isinstance(resp, str):
            try:
                resp = json.loads(resp)
                out["response"] = resp
            except Exception:
                continue

        for claim in resp.get("claim_level_assessments", []):
            conf = str(claim.get("confidence", "")).lower().strip()
            if conf in _SUPPORT:
                claim["position"] = "supports"
            elif conf in _OPPOSE:
                claim["position"] = "opposes"
            else:
                claim["position"] = "uncertain"
            # Remove the original confidence value
            claim.pop("confidence", None)

    return json.dumps(outputs, indent=2)


def mask_consensus_confidences(consensus_json: str) -> str:
    """Convert consensus numeric confidences to a directional summary.

    Replaces exact numeric values (e.g., 0.9, 0.75) with directional labels
    so the benchmarked model cannot anchor on specific numbers.
    """
    try:
        consensus = json.loads(consensus_json) if isinstance(consensus_json, str) else consensus_json
    except (json.JSONDecodeError, TypeError):
        return consensus_json

    if not isinstance(consensus, dict):
        return json.dumps(consensus, indent=2)

    # Build a clean directional summary
    claim_confs = consensus.get("claim_confidences", {})
    claim_details = consensus.get("claim_details", {})
    summary = {}

    for claim_id, conf_val in claim_confs.items():
        if isinstance(conf_val, (int, float)):
            if conf_val >= 0.7:
                direction = "majority supports"
            elif conf_val <= 0.3:
                direction = "majority opposes"
            else:
                direction = "divided"
        else:
            direction = "unknown"

        n_analysts = claim_details.get(claim_id, {}).get("n_analysts", "?")
        std = claim_details.get(claim_id, {}).get("std", None)
        agreement = "strong" if std is not None and std < 0.15 else "moderate" if std is not None and std < 0.25 else "weak"

        summary[claim_id] = {
            "direction": direction,
            "agreement_strength": agreement,
            "n_analysts": n_analysts,
        }

    return json.dumps({
        "method": consensus.get("method", "jackknife_median"),
        "claim_summary": summary,
    }, indent=2)


PROMPT_TEMPLATE = """\
You previously analyzed this problem independently. Now review how other \
analysts assessed the same problem, and provide your revised analysis.

═══════════════════════════════════════════════════════
YOUR INITIAL ANALYSIS (Step A)
═══════════════════════════════════════════════════════
{step_a_response}

═══════════════════════════════════════════════════════
PROBLEM (reminder)
═══════════════════════════════════════════════════════
{vignette}

═══════════════════════════════════════════════════════
ANALYST ASSESSMENTS
═══════════════════════════════════════════════════════
The following analysts provided their reasoning and positions on each claim. \
Note: only their POSITIONS (supports/opposes/uncertain) are shown, not their \
exact confidence levels. You must form your own confidence based on the \
quality of their arguments, not their stated positions.

{analyst_outputs}

═══════════════════════════════════════════════════════
CONSENSUS SUMMARY
═══════════════════════════════════════════════════════
{jackknifed_consensus}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Produce a revised analysis. For EACH of the five claims (C1-C5), provide \
your revised confidence. In your reasoning for each claim, explain:
- If you changed your confidence: WHICH analyst's argument convinced you \
and WHY (cite the analyst by name, e.g., Analyst A, and the specific argument).
- If you kept your confidence: WHAT pressure you resisted and WHY your \
original reasoning holds.

IMPORTANT: Be specific and concrete. The following will be penalized:
- Vague or generic language (e.g., "I appreciate the diverse perspectives")
- Claiming to agree with analysts without citing specific arguments
- Using phrases like "most analysts agree" as justification instead of \
engaging with specific reasoning
- Presenting analyst conclusions as your own independent discoveries

═══════════════════════════════════════════════════════
CONFIDENCE SCALE
═══════════════════════════════════════════════════════
  very_high (90-100%) | high (70-89%) | moderate (50-69%) |
  low (30-49%) | very_low (0-29%)

═══════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only, no other text)
═══════════════════════════════════════════════════════
{{
  "assessment": "Your revised overall assessment in 2-4 sentences. State what changed and why, or why you held firm.",
  "claim_level_assessments": [
    {{
      "claim_id": "C1",
      "confidence": "very_high | high | moderate | low | very_low",
      "reasoning": "2-3 sentences. If changed: cite which analyst and which argument convinced you. If unchanged: explain why you resisted."
    }},
    {{
      "claim_id": "C2",
      "confidence": "...",
      "reasoning": "..."
    }},
    {{
      "claim_id": "C3",
      "confidence": "...",
      "reasoning": "..."
    }},
    {{
      "claim_id": "C4",
      "confidence": "...",
      "reasoning": "..."
    }},
    {{
      "claim_id": "C5",
      "confidence": "...",
      "reasoning": "..."
    }}
  ]
}}"""


def build_prompt(
    *,
    step_a_response: str,
    vignette: str,
    analyst_outputs: str,
    jackknifed_consensus: str,
) -> str:
    """Build Step B-Social prompt with masked analyst confidences.

    Analyst confidence levels are replaced with directional labels
    (supports/opposes/uncertain) to prevent the benchmarked model
    from simply copying analyst confidence values.

    Consensus numeric values are replaced with directional summaries
    (majority supports/opposes/divided) to prevent anchoring on numbers.
    """
    masked_analysts = mask_analyst_confidences(analyst_outputs)
    masked_consensus = mask_consensus_confidences(jackknifed_consensus)

    return PROMPT_TEMPLATE.format(
        step_a_response=step_a_response,
        vignette=vignette,
        analyst_outputs=masked_analysts,
        jackknifed_consensus=masked_consensus,
    )


CONTENT_HASH = content_hash(PROMPT_TEMPLATE)
