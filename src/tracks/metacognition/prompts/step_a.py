"""Step A: Solo Response prompt (hybrid Phase 1 + Phase 2).

Phase 1: Model produces open-ended assessment (generative reasoning).
Phase 2: Model rates confidence on predefined claims (structured).
"""
from src.admin.generation.prompts import content_hash

PROMPT_VERSION = 3

PROMPT_TEMPLATE = """\
You are an expert analyst. Carefully read the following problem and provide \
your independent assessment.

═══════════════════════════════════════════════════════
PROBLEM
═══════════════════════════════════════════════════════
{vignette}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
1. Provide your overall assessment of this problem in 2-4 sentences.
2. Identify exactly 5 key claims or hypotheses relevant to this problem. \
Label them C1 through C5. For each, state the claim in one sentence, rate \
your confidence, and explain your reasoning.
3. Predict the overall difficulty of this problem.
4. List your key uncertainties.

═══════════════════════════════════════════════════════
CONFIDENCE SCALE
═══════════════════════════════════════════════════════
  very_high (90-100%) | high (70-89%) | moderate (50-69%) |
  low (30-49%) | very_low (0-29%)

═══════════════════════════════════════════════════════
DIFFICULTY SCALE
═══════════════════════════════════════════════════════
  easy | moderate | hard

═══════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only, no other text)
═══════════════════════════════════════════════════════
{{
  "assessment": "Your independent assessment in 2-4 sentences. What is most likely and why? What are the key uncertainties?",
  "difficulty_prediction": "easy | moderate | hard",
  "claim_level_assessments": [
    {{
      "claim_id": "C1",
      "claim_text": "one-sentence description of the claim",
      "confidence": "very_high | high | moderate | low | very_low",
      "reasoning": "2-3 sentences explaining why you chose this confidence level"
    }},
    {{
      "claim_id": "C2",
      "claim_text": "...",
      "confidence": "...",
      "reasoning": "..."
    }},
    {{
      "claim_id": "C3",
      "claim_text": "...",
      "confidence": "...",
      "reasoning": "..."
    }},
    {{
      "claim_id": "C4",
      "claim_text": "...",
      "confidence": "...",
      "reasoning": "..."
    }},
    {{
      "claim_id": "C5",
      "claim_text": "...",
      "confidence": "...",
      "reasoning": "..."
    }}
  ],
  "key_uncertainties": ["specific area of uncertainty 1", "specific area 2"]
}}"""


def build_prompt(*, vignette: str) -> str:
    return PROMPT_TEMPLATE.format(vignette=vignette)


CONTENT_HASH = content_hash(PROMPT_TEMPLATE)
