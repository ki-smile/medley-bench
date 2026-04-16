"""Step B-Minimal: Minimal-instruction probe prompt.

Stripped-down prompt with no metacognitive rubric. Tests whether metacognitive
behavior is prompt-driven vs genuinely internalized.
"""
from src.admin.generation.prompts import content_hash

PROMPT_VERSION = 2

PROMPT_TEMPLATE = """\
You previously analyzed this problem. Here is how other analysts assessed it.

═══════════════════════════════════════════════════════
YOUR INITIAL ANALYSIS
═══════════════════════════════════════════════════════
{step_a_response}

═══════════════════════════════════════════════════════
PROBLEM
═══════════════════════════════════════════════════════
{vignette}

═══════════════════════════════════════════════════════
ANALYST ASSESSMENTS
═══════════════════════════════════════════════════════
{analyst_outputs}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Provide your revised analysis for all five claims (C1-C5).

═══════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only, no other text)
═══════════════════════════════════════════════════════
{{
  "assessment": "your revised assessment",
  "claim_level_assessments": [
    {{
      "claim_id": "C1",
      "confidence": "very_high | high | moderate | low | very_low",
      "reasoning": "why this confidence level"
    }}
  ]
}}"""


def build_prompt(
    *, step_a_response: str, vignette: str, analyst_outputs: str
) -> str:
    return PROMPT_TEMPLATE.format(
        step_a_response=step_a_response,
        vignette=vignette,
        analyst_outputs=analyst_outputs,
    )


CONTENT_HASH = content_hash(PROMPT_TEMPLATE)
