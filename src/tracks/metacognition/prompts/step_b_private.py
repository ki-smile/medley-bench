"""Step B-Private: Self-Revision prompt with System 2 nudge.

Model re-reads the problem and its own Step A response.
NO ensemble, NO other analysts. Measures prompted self-revision.

v4: Added self-review checklist (nudge) based on empirical finding that
original passive prompt ("you may revise") triggered near-zero changes.
The nudge increases self-revision from ~37% to ~96% of instances.
"""
from src.admin.generation.prompts import content_hash

PROMPT_VERSION = 4

PROMPT_TEMPLATE = """\
You previously analyzed this problem. Now critically re-examine your work.

═══════════════════════════════════════════════════════
YOUR INITIAL ANALYSIS (Step A)
═══════════════════════════════════════════════════════
{step_a_response}

═══════════════════════════════════════════════════════
PROBLEM (reminder)
═══════════════════════════════════════════════════════
{vignette}

═══════════════════════════════════════════════════════
SELF-REVIEW CHECKLIST
═══════════════════════════════════════════════════════
Before revising, answer these honestly:
1. Which claim are you LEAST confident about, and why?
2. What is the strongest argument AGAINST your most confident claim?
3. Did you consider alternative explanations that would change C3-C5?
4. Are any of your confidence levels suspiciously similar?
5. What would make you change your mind on your top claim?

Now revise your assessment based on this self-review. If your \
self-review identified issues, adjust your confidence levels accordingly. \
If your analysis holds up under scrutiny, you may keep your levels, but \
explain why.

Provide your assessment for ALL five claims (C1-C5) from your initial analysis.

═══════════════════════════════════════════════════════
CONFIDENCE SCALE
═══════════════════════════════════════════════════════
  very_high (90-100%) | high (70-89%) | moderate (50-69%) |
  low (30-49%) | very_low (0-29%)

═══════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only, no other text)
═══════════════════════════════════════════════════════
{{
  "assessment": "Start by noting what your self-review found, then your revised assessment.",
  "claim_level_assessments": [
    {{
      "claim_id": "C1",
      "confidence": "very_high | high | moderate | low | very_low",
      "reasoning": "why this confidence level — reference your self-review"
    }},
    {{"claim_id": "C2", "confidence": "...", "reasoning": "..."}},
    {{"claim_id": "C3", "confidence": "...", "reasoning": "..."}},
    {{"claim_id": "C4", "confidence": "...", "reasoning": "..."}},
    {{"claim_id": "C5", "confidence": "...", "reasoning": "..."}}
  ]
}}"""


def build_prompt(*, step_a_response: str, vignette: str) -> str:
    return PROMPT_TEMPLATE.format(
        step_a_response=step_a_response,
        vignette=vignette,
    )


CONTENT_HASH = content_hash(PROMPT_TEMPLATE)
