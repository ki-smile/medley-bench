"""Response parsing utilities for MEDLEY-BENCH.

Handles JSON extraction from LLM responses and confidence mapping.
"""
from __future__ import annotations

import json
import re

CONFIDENCE_MAP: dict[str, float] = {
    "very_high": 0.95,
    "high": 0.80,
    "moderate": 0.55,
    "low": 0.35,
    "very_low": 0.15,
}

# Fuzzy aliases — models produce all sorts of variations
_CONFIDENCE_ALIASES: dict[str, str] = {
    # Truncated / partial
    "ve": "very_high", "ver": "very_high", "very": "very_high",
    "very high": "very_high", "veryhigh": "very_high", "v_high": "very_high",
    "very_hi": "very_high", "vhigh": "very_high", "vh": "very_high",
    "hi": "high", "h": "high",
    "mod": "moderate", "med": "moderate", "medium": "moderate", "m": "moderate",
    "lo": "low", "l": "low",
    "very low": "very_low", "verylow": "very_low", "very_lo": "very_low",
    "v_low": "very_low", "vlow": "very_low", "vl": "very_low",
    # With percentages
    "very_high (90-100%)": "very_high", "high (70-89%)": "high",
    "moderate (50-69%)": "moderate", "low (30-49%)": "low",
    "very_low (0-29%)": "very_low",
    # Capitalized variants
    "very_high": "very_high", "high": "high", "moderate": "moderate",
    "low": "low", "very_low": "very_low",
    # Other
    "not applicable": "moderate", "n/a": "moderate", "na": "moderate",
    "uncertain": "moderate", "unsure": "moderate",
}

NUMERIC_TO_LABEL: dict[float, str] = {v: k for k, v in CONFIDENCE_MAP.items()}


def conf_to_numeric(label) -> float:
    """Map a confidence label to its numeric value.

    Handles fuzzy/partial labels like "ve", "very high", "medium", "VH", etc.
    Also handles numeric values and dict/list wrappers.
    Returns 0.55 (moderate) for unrecognized labels.
    """
    # Handle non-string inputs
    if isinstance(label, (int, float)):
        return float(label)
    if isinstance(label, dict):
        # Some models return {"level": "high"} or {"confidence": "high"}
        for key in ("level", "confidence", "value", "rating"):
            if key in label:
                return conf_to_numeric(label[key])
        return 0.55
    if isinstance(label, list):
        return conf_to_numeric(label[0]) if label else 0.55
    if not isinstance(label, str):
        return 0.55

    cleaned = label.strip().lower().replace("-", "_")

    # Direct match
    if cleaned in CONFIDENCE_MAP:
        return CONFIDENCE_MAP[cleaned]

    # Alias match
    canonical = _CONFIDENCE_ALIASES.get(cleaned)
    if canonical:
        return CONFIDENCE_MAP[canonical]

    # Try stripping parenthetical content: "high (70-89%)" → "high"
    paren_stripped = re.sub(r"\s*\(.*?\)", "", cleaned).strip()
    if paren_stripped in CONFIDENCE_MAP:
        return CONFIDENCE_MAP[paren_stripped]
    canonical = _CONFIDENCE_ALIASES.get(paren_stripped)
    if canonical:
        return CONFIDENCE_MAP[canonical]

    # Try prefix matching: "very_h" → "very_high"
    for key in CONFIDENCE_MAP:
        if key.startswith(cleaned) and len(cleaned) >= 2:
            return CONFIDENCE_MAP[key]

    return 0.55


def numeric_to_conf(value: float) -> str:
    """Map a numeric confidence to its nearest label."""
    best_label = "moderate"
    best_dist = float("inf")
    for num, label in NUMERIC_TO_LABEL.items():
        dist = abs(num - value)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def get_claim_conf(
    response, *, claim_id: str, claim_text: str = ""
) -> float | None:
    """Extract numeric confidence for a specific claim from a step response.

    Searches claim_level_assessments by claim_id first, then falls back to
    claim_text substring matching. Handles list-wrapped responses.
    """
    if isinstance(response, list):
        # Some models return [{"claim_id":...}] directly
        response = {"claim_level_assessments": response}
    if not isinstance(response, dict):
        return None
    assessments = response.get("claim_level_assessments", [])
    if not assessments:
        return None

    # Try exact claim_id match
    for item in assessments:
        if item.get("claim_id") == claim_id:
            conf = item.get("confidence", "")
            if isinstance(conf, (int, float)):
                return float(conf)
            return conf_to_numeric(str(conf))

    # Fallback: claim_text substring match
    if claim_text:
        for item in assessments:
            item_text = item.get("claim_text", "")
            if claim_text.lower() in item_text.lower() or item_text.lower() in claim_text.lower():
                conf = item.get("confidence", "")
                if isinstance(conf, (int, float)):
                    return float(conf)
                return conf_to_numeric(str(conf))

    return None


def extract_claim_ids(response: dict) -> list[str]:
    """Get all claim IDs from a step response."""
    return [
        item["claim_id"]
        for item in response.get("claim_level_assessments", [])
        if "claim_id" in item
    ]


def parse_json_response(raw: str) -> dict:
    """Extract JSON from an LLM response.

    Handles:
    - Pure JSON responses
    - JSON wrapped in markdown code fences (```json ... ```)
    - JSON embedded in surrounding text
    - Partial/malformed JSON (returns error dict with raw text)
    """
    if not raw or not raw.strip():
        return {"_parse_error": True, "_raw": raw}

    text = raw.strip()

    # Strip extended thinking tags (DeepSeek-R1, Qwen3.5, etc.)
    # Models may emit <think>...</think> reasoning before the actual answer
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Also handle unclosed <think> tags (thinking truncated by token limit)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    if not text:
        return {"_parse_error": True, "_raw": raw}

    # Try direct parse first
    try:
        result = json.loads(text)
        # If model returns a list, wrap it
        if isinstance(result, list):
            return {"claim_level_assessments": result}
        return result
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return {"claim_level_assessments": result}
            return result
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group())
            if isinstance(result, list):
                return {"claim_level_assessments": result}
            return result
        except json.JSONDecodeError:
            pass

    # Try repairing truncated JSON (models sometimes cut off mid-response)
    repaired = _repair_truncated_json(text)
    if repaired:
        return repaired

    return {"_parse_error": True, "_raw": raw}


def _repair_truncated_json(text: str) -> dict | None:
    """Attempt to repair truncated JSON by closing open brackets/braces."""
    # Find the start of JSON
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:]

    # Count unclosed brackets/braces and try closing them
    for attempt in range(5):
        opens = fragment.count("[") - fragment.count("]")
        braces = fragment.count("{") - fragment.count("}")

        # Strip trailing partial content (incomplete strings, etc.)
        # Try trimming to the last complete element
        for trim in [
            # Try trimming after last complete JSON element
            lambda s: s[:s.rfind("}") + 1] if s.rfind("}") > 0 else s,
            lambda s: s,
        ]:
            trimmed = trim(fragment)
            # Close all open brackets/braces
            closes = "]" * max(0, trimmed.count("[") - trimmed.count("]"))
            closes += "}" * max(0, trimmed.count("{") - trimmed.count("}"))
            try:
                return json.loads(trimmed + closes)
            except json.JSONDecodeError:
                continue

        # Trim further — remove last incomplete element
        last_comma = fragment.rfind(",")
        if last_comma > 0:
            fragment = fragment[:last_comma]
        else:
            break

    return None
