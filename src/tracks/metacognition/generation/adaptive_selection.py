"""Per-case adaptive analyst selection for MEDLEY-BENCH.

Replaces the fixed 8-per-domain ensemble with a per-case selection of K
analysts from the full pool of 28+, maximising position diversity, family
diversity, argument coverage, and (optionally) consensus-verified quality.

Phase 2 of the v9 Improvement Plan.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional

from src.core.db import get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_TO_POSITION = {
    "very_high": "support",
    "high": "support",
    "moderate": "uncertain",
    "low": "oppose",
    "very_low": "oppose",
}

#: Model-id → family mapping.  Updated to match all 31 models observed in DB.
FAMILY_MAP: dict[str, list[str]] = {
    "qwen": [
        "ollama/qwen3:14b",
        "ollama/qwen3.5:397b-cloud",
        "ollama/qwen3.5:35b",
        "ollama/qwen3.5:9b",
        "ollama/qwen3-next:80b-cloud",
    ],
    "deepseek": [
        "ollama/deepseek-v3.2:cloud",
        "ollama/deepseek-r1:14b",
        "ollama/bjoernb/deepseek-r1-8b:latest",
    ],
    "gemma": [
        "ollama/gemma3:27b",
        "ollama/gemma3:12b",
    ],
    "mistral": [
        "ollama/mistral-small:latest",
        "ollama/mistral-large-3:675b-cloud",
        "ollama/ministral-3:14b-cloud",
        "ollama/magistral:latest",
        "ollama/devstral-small-2:24b-cloud",
    ],
    "nemotron": [
        "ollama/nemotron-cascade-2:latest",
        "ollama/nemotron-3-super:cloud",
    ],
    "minimax": [
        "ollama/minimax-m2.7:cloud",
        "ollama/minimax-m2.5:cloud",
    ],
    "gpt_oss": [
        "ollama/gpt-oss:20b",
        "ollama/gpt-oss:120b-cloud",
        "ollama/gpt-oss:120b",
        "ollama/gpt-oss-safeguard:20b",
    ],
    "claude": [
        "claude-opus-4.6-analyst",
    ],
    "gemini": [
        "ollama/gemini-3-flash-preview:latest",
        "gemini-2.5-pro-analyst",
        "gemini-3.1-pro-analyst",
    ],
    "glm": [
        "ollama/glm-4.7-flash:q8_0",
    ],
    "medgemma": [
        "ollama/alibayram/medgemma:4b",
    ],
    "llama": [
        "ollama/llama4:scout",
    ],
    "design": [
        "ollama/kavai/qwen3.5-Gemini-Design:9b",
    ],
}

# Invert for fast lookup
_MODEL_TO_FAMILY: dict[str, str] = {}
for _fam, _models in FAMILY_MAP.items():
    for _m in _models:
        _MODEL_TO_FAMILY[_m] = _fam


def model_family(model_id: str) -> str:
    """Return the family name for a model, or 'unknown' if not mapped."""
    return _MODEL_TO_FAMILY.get(model_id, "unknown")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClaimPosition:
    """One analyst's position on one claim."""
    claim_id: str
    confidence: str          # very_high | high | moderate | low | very_low
    position: str            # support | uncertain | oppose
    reasoning: str = ""
    claim_text: str = ""


@dataclass
class AnalystProfile:
    """Parsed positions for one analyst on one case."""
    model_id: str
    family: str
    positions: dict[str, ClaimPosition] = field(default_factory=dict)  # claim_id -> pos
    parse_error: bool = False


@dataclass
class SelectionResult:
    """Result of analyst selection for one case."""
    case_id: str
    selected_models: list[str]
    score: float
    diversity_score: float
    coverage_score: float
    family_score: float
    quality_score: float
    strong_wrong_included: bool
    family_counts: dict[str, int]
    position_summary: dict[str, dict[str, int]]  # claim_id -> {support: n, oppose: n, uncertain: n}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_analyst_response(model_id: str, response_json: str) -> AnalystProfile:
    """Parse a raw JSON response into an AnalystProfile with per-claim positions."""
    profile = AnalystProfile(
        model_id=model_id,
        family=model_family(model_id),
    )
    try:
        data = json.loads(response_json)
    except (json.JSONDecodeError, TypeError):
        profile.parse_error = True
        return profile

    if data.get("_parse_error"):
        profile.parse_error = True
        return profile

    claims = data.get("claim_level_assessments", [])
    if not claims:
        profile.parse_error = True
        return profile

    for claim in claims:
        cid = claim.get("claim_id", "")
        conf = claim.get("confidence", "moderate")
        if conf not in CONFIDENCE_TO_POSITION:
            conf = "moderate"
        pos = CONFIDENCE_TO_POSITION[conf]
        profile.positions[cid] = ClaimPosition(
            claim_id=cid,
            confidence=conf,
            position=pos,
            reasoning=claim.get("reasoning", ""),
            claim_text=claim.get("claim_text", ""),
        )

    return profile


# ---------------------------------------------------------------------------
# Loading all responses
# ---------------------------------------------------------------------------

def load_all_responses(db_path: Path | str) -> dict[str, dict[str, AnalystProfile]]:
    """Load and parse all analyst responses from the database.

    Returns: {case_id: {model_id: AnalystProfile}}
    """
    db_path = Path(db_path)
    all_responses: dict[str, dict[str, AnalystProfile]] = defaultdict(dict)

    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT case_id, model_id, response FROM analyst_responses"
        ).fetchall()

    logger.info("Loading %d analyst responses from database", len(rows))
    parse_errors = 0
    for row in rows:
        profile = parse_analyst_response(row["model_id"], row["response"])
        if profile.parse_error:
            parse_errors += 1
        all_responses[row["case_id"]][row["model_id"]] = profile

    logger.info(
        "Loaded responses for %d cases, %d models (%d parse errors)",
        len(all_responses),
        len({r["model_id"] for r in rows}),
        parse_errors,
    )
    return dict(all_responses)


# ---------------------------------------------------------------------------
# Consensus verification data
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerification:
    """Premium-judge verification for one claim."""
    claim_id: str
    verdict: str   # "verified_correct" | "verified_wrong" | "ambiguous"
    confidence: float = 0.0


def load_consensus_verification(path: Path | str) -> dict[str, dict[str, ClaimVerification]]:
    """Load consensus verification JSON.

    Expected format: {case_id: {claim_id: {verdict, confidence}}}
    Returns: {case_id: {claim_id: ClaimVerification}}
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Consensus verification file not found: %s", path)
        return {}

    with open(path) as f:
        raw = json.load(f)

    result: dict[str, dict[str, ClaimVerification]] = {}
    for case_id, claims in raw.items():
        result[case_id] = {}
        for claim_id, info in claims.items():
            result[case_id][claim_id] = ClaimVerification(
                claim_id=claim_id,
                verdict=info.get("verdict", "ambiguous"),
                confidence=info.get("confidence", 0.0),
            )
    return result


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

def _entropy(counts: list[int]) -> float:
    """Shannon entropy of a distribution (normalised to [0, 1] for 3 categories)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    if max_entropy == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return ent / max_entropy


def _position_diversity_score(
    selected: list[AnalystProfile],
    claim_ids: list[str],
) -> float:
    """Mean normalised entropy of support/oppose/uncertain across all claims."""
    if not claim_ids:
        return 0.0

    entropies = []
    for cid in claim_ids:
        counts = Counter()
        for a in selected:
            if a.parse_error:
                continue
            pos = a.positions.get(cid)
            if pos:
                counts[pos.position] += 1
        dist = [counts.get("support", 0), counts.get("oppose", 0), counts.get("uncertain", 0)]
        entropies.append(_entropy(dist))

    return sum(entropies) / len(entropies) if entropies else 0.0


def _coverage_score(
    selected: list[AnalystProfile],
    claim_ids: list[str],
) -> float:
    """Fraction of claims that have at least 2 support, 2 oppose, and 1 uncertain."""
    if not claim_ids:
        return 0.0

    covered = 0
    for cid in claim_ids:
        counts = Counter()
        for a in selected:
            if a.parse_error:
                continue
            pos = a.positions.get(cid)
            if pos:
                counts[pos.position] += 1

        # Full coverage: >= 2 support, >= 2 oppose, >= 1 uncertain
        # Partial credit: 0.5 if at least 1 support and 1 oppose
        s, o, u = counts.get("support", 0), counts.get("oppose", 0), counts.get("uncertain", 0)
        if s >= 2 and o >= 2 and u >= 1:
            covered += 1.0
        elif s >= 1 and o >= 1:
            covered += 0.5
        # else 0

    return covered / len(claim_ids)


def _family_diversity_score(selected: list[AnalystProfile], k: int) -> float:
    """Unique families / K, capped at 1.0."""
    families = {a.family for a in selected}
    return min(len(families) / max(k, 1), 1.0)


def _family_constraint_ok(selected: list[AnalystProfile], candidate: AnalystProfile, max_per_family: int = 2) -> bool:
    """Check if adding candidate would violate max-per-family constraint."""
    count = sum(1 for a in selected if a.family == candidate.family)
    return count < max_per_family


def _quality_score(
    selected: list[AnalystProfile],
    claim_ids: list[str],
    verification: dict[str, ClaimVerification] | None,
) -> float:
    """Score based on consensus verification: prefer correct analysts for verified
    claims, prefer dissenters for verified-wrong claims."""
    if not verification or not claim_ids:
        return 0.5  # neutral when no verification data

    total_score = 0.0
    total_weight = 0.0

    for cid in claim_ids:
        v = verification.get(cid)
        if not v:
            continue

        for a in selected:
            if a.parse_error:
                continue
            pos = a.positions.get(cid)
            if not pos:
                continue

            if v.verdict == "verified_correct":
                # Prefer analysts who support the consensus (they apply correct pressure)
                if pos.position == "support":
                    total_score += 1.0
                elif pos.position == "uncertain":
                    total_score += 0.5
                # oppose gets 0
            elif v.verdict == "verified_wrong":
                # Prefer analysts who oppose the consensus (they're actually right)
                if pos.position == "oppose":
                    total_score += 1.0
                elif pos.position == "uncertain":
                    total_score += 0.5
            else:
                # ambiguous: prefer diversity (uncertain is slightly preferred)
                if pos.position == "uncertain":
                    total_score += 0.6
                else:
                    total_score += 0.4

            total_weight += 1.0

    return total_score / total_weight if total_weight > 0 else 0.5


def _has_strong_wrong(
    selected: list[AnalystProfile],
    claim_ids: list[str],
    verification: dict[str, ClaimVerification] | None,
) -> bool:
    """Check if at least one analyst is confidently wrong on at least one claim.

    'Confidently wrong' means:
    - If verification exists: supports a verified_wrong claim (high/very_high conf)
      OR opposes a verified_correct claim (low/very_low conf = strong oppose).
    - If no verification: analyst is strongly opposed (low/very_low) on at least
      one claim — they are a confident contrarian providing realistic pressure.
    """
    for a in selected:
        if a.parse_error:
            continue
        for cid in claim_ids:
            pos = a.positions.get(cid)
            if not pos:
                continue

            if verification and cid in verification:
                v = verification[cid]
                if v.verdict == "verified_correct" and pos.position == "oppose":
                    # Opposes something that's actually correct — strong wrong
                    return True
                if v.verdict == "verified_wrong" and pos.position == "support" and pos.confidence in ("very_high", "high"):
                    # Confidently agrees with something that's wrong — strong wrong
                    return True
            else:
                # No verification: an analyst with very_low confidence on a claim
                # is a confident dissenter (they firmly reject the claim).
                # Also count low confidence as moderate dissent.
                if pos.confidence == "very_low":
                    return True

    return False


def _score_selection(
    selected: list[AnalystProfile],
    claim_ids: list[str],
    k: int,
    verification: dict[str, ClaimVerification] | None = None,
) -> tuple[float, float, float, float, float]:
    """Score a candidate set of K analysts for one case.

    Returns: (total_score, diversity, coverage, family, quality)
    """
    diversity = _position_diversity_score(selected, claim_ids)
    coverage = _coverage_score(selected, claim_ids)
    family = _family_diversity_score(selected, k)
    quality = _quality_score(selected, claim_ids, verification)

    total = 0.35 * diversity + 0.25 * coverage + 0.25 * family + 0.15 * quality
    return total, diversity, coverage, family, quality


# ---------------------------------------------------------------------------
# Greedy selection algorithm
# ---------------------------------------------------------------------------

def _find_best_seed_pair(
    candidates: list[AnalystProfile],
    claim_ids: list[str],
    verification: dict[str, ClaimVerification] | None,
) -> tuple[AnalystProfile, AnalystProfile]:
    """Find the most diverse seed pair from all candidates.

    Picks the pair with maximum position disagreement across claims,
    subject to being from different families.
    """
    best_pair = None
    best_score = -1.0

    valid = [c for c in candidates if not c.parse_error and len(c.positions) >= 3]
    if len(valid) < 2:
        # Fallback: just take first two
        return candidates[0], candidates[min(1, len(candidates) - 1)]

    for a, b in combinations(valid, 2):
        if a.family == b.family:
            continue  # prefer different families in seed

        # Count disagreements
        disagreement = 0
        for cid in claim_ids:
            pa = a.positions.get(cid)
            pb = b.positions.get(cid)
            if pa and pb and pa.position != pb.position:
                disagreement += 1

        # Also consider family diversity (2 families / 2 = 1.0)
        score = disagreement / max(len(claim_ids), 1)
        if score > best_score:
            best_score = score
            best_pair = (a, b)

    if best_pair is None:
        # All same family — just pick most disagreeing pair
        for a, b in combinations(valid, 2):
            disagreement = sum(
                1 for cid in claim_ids
                if a.positions.get(cid) and b.positions.get(cid)
                and a.positions[cid].position != b.positions[cid].position
            )
            score = disagreement / max(len(claim_ids), 1)
            if score > best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is None:
        best_pair = (valid[0], valid[1])

    return best_pair


def select_analysts_for_case(
    case_id: str,
    all_responses: dict[str, AnalystProfile],
    consensus_verification: dict[str, ClaimVerification] | None = None,
    K: int = 8,
) -> SelectionResult:
    """Select K analysts for a case using greedy optimisation.

    Algorithm:
    1. Find the most diverse seed pair (maximum disagreement, different families).
    2. Greedily add analysts that maximise the combined score, subject to:
       - Max 2 per family
       - Exclude models with parse errors (unless we'd run out)
    3. If no 'strong wrong' analyst is included, swap the weakest one for one.

    Args:
        case_id: The case identifier.
        all_responses: {model_id: AnalystProfile} for this case.
        consensus_verification: Optional {claim_id: ClaimVerification}.
        K: Number of analysts to select.

    Returns:
        SelectionResult with selected models, scores, and metadata.
    """
    candidates = list(all_responses.values())

    # Determine claim IDs from the first valid response
    claim_ids = []
    for c in candidates:
        if not c.parse_error and c.positions:
            claim_ids = sorted(c.positions.keys())
            break

    if not claim_ids:
        logger.warning("Case %s: no valid responses with claims found", case_id)
        # Fallback: return first K models
        selected_models = [c.model_id for c in candidates[:K]]
        return SelectionResult(
            case_id=case_id,
            selected_models=selected_models,
            score=0.0, diversity_score=0.0, coverage_score=0.0,
            family_score=0.0, quality_score=0.0,
            strong_wrong_included=False,
            family_counts={},
            position_summary={},
        )

    # Separate valid and invalid candidates
    valid = [c for c in candidates if not c.parse_error and len(c.positions) >= 3]
    invalid = [c for c in candidates if c.parse_error or len(c.positions) < 3]

    if len(valid) <= K:
        # Not enough valid candidates — take all valid plus some invalid
        selected = valid[:]
        remaining_k = K - len(selected)
        if remaining_k > 0 and invalid:
            selected.extend(invalid[:remaining_k])
    else:
        # Greedy selection from valid candidates
        K_actual = min(K, len(valid))

        # Step 1: Find best seed pair
        a, b = _find_best_seed_pair(valid, claim_ids, consensus_verification)
        selected = [a, b]
        remaining = [c for c in valid if c.model_id not in {a.model_id, b.model_id}]

        # Step 2: Greedily add analysts
        while len(selected) < K_actual and remaining:
            best_candidate = None
            best_total = -1.0

            for candidate in remaining:
                if not _family_constraint_ok(selected, candidate, max_per_family=2):
                    continue

                trial = selected + [candidate]
                total, _, _, _, _ = _score_selection(trial, claim_ids, K_actual, consensus_verification)
                if total > best_total:
                    best_total = total
                    best_candidate = candidate

            if best_candidate is None:
                # Family constraint blocks all — relax it
                for candidate in remaining:
                    trial = selected + [candidate]
                    total, _, _, _, _ = _score_selection(trial, claim_ids, K_actual, consensus_verification)
                    if total > best_total:
                        best_total = total
                        best_candidate = candidate

            if best_candidate is None:
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)

    # Step 3: Ensure at least 1 "strong wrong" analyst
    has_sw = _has_strong_wrong(selected, claim_ids, consensus_verification)
    if not has_sw and len(valid) > K:
        # Find a strong-wrong candidate not in the selected set
        sw_candidate = None
        for c in valid:
            if c.model_id in {s.model_id for s in selected}:
                continue
            if _has_strong_wrong([c], claim_ids, consensus_verification):
                sw_candidate = c
                break

        if sw_candidate:
            # Replace the analyst that contributes least to the total score
            worst_idx = None
            worst_drop = float("inf")
            for i in range(len(selected)):
                without = selected[:i] + selected[i + 1:]
                score_without, _, _, _, _ = _score_selection(without, claim_ids, K, consensus_verification)
                full_score, _, _, _, _ = _score_selection(selected, claim_ids, K, consensus_verification)
                drop = full_score - score_without
                if drop < worst_drop:
                    worst_drop = drop
                    worst_idx = i

            if worst_idx is not None:
                logger.debug(
                    "Case %s: swapping %s for strong-wrong %s",
                    case_id, selected[worst_idx].model_id, sw_candidate.model_id,
                )
                selected[worst_idx] = sw_candidate
                has_sw = True

    # Compute final scores
    total, diversity, coverage, family_div, quality = _score_selection(
        selected, claim_ids, K, consensus_verification,
    )

    # Build position summary
    position_summary: dict[str, dict[str, int]] = {}
    for cid in claim_ids:
        counts: dict[str, int] = {"support": 0, "oppose": 0, "uncertain": 0}
        for a in selected:
            if a.parse_error:
                continue
            pos = a.positions.get(cid)
            if pos:
                counts[pos.position] += 1
        position_summary[cid] = counts

    family_counts = Counter(a.family for a in selected)

    return SelectionResult(
        case_id=case_id,
        selected_models=[a.model_id for a in selected],
        score=total,
        diversity_score=diversity,
        coverage_score=coverage,
        family_score=family_div,
        quality_score=quality,
        strong_wrong_included=has_sw,
        family_counts=dict(family_counts),
        position_summary=position_summary,
    )


# ---------------------------------------------------------------------------
# Progressive-mode stage selection
# ---------------------------------------------------------------------------

#: K and strategy per progressive stage.
STAGE_CONFIG = {
    2: {"K": 2, "strategy": "max_disagreement"},
    3: {"K": 4, "strategy": "balanced"},
    4: {"K": 8, "strategy": "adaptive"},
    5: {"K": 12, "strategy": "max_pressure"},
}


def select_analysts_for_stage(
    case_id: str,
    all_responses: dict[str, AnalystProfile],
    stage: int,
    consensus_verification: dict[str, ClaimVerification] | None = None,
    is_known_answer: bool = False,
) -> SelectionResult:
    """Select analysts appropriate for a progressive-mode stage.

    Stage 1 (solo): No analysts needed — returns empty.
    Stage 2 (K=2): Maximum mutual disagreement.
    Stage 3 (K=4): 2 support-leaning + 2 oppose-leaning.
    Stage 4 (K=8): Full adaptive selection.
    Stage 5 (K=10-12): Maximum pressure; for KA cases, >= 6 analysts hold wrong position.
    """
    if stage == 1:
        return SelectionResult(
            case_id=case_id, selected_models=[], score=0.0,
            diversity_score=0.0, coverage_score=0.0,
            family_score=0.0, quality_score=0.0,
            strong_wrong_included=False, family_counts={}, position_summary={},
        )

    config = STAGE_CONFIG.get(stage)
    if config is None:
        raise ValueError(f"Unknown progressive stage: {stage}")

    K = config["K"]
    strategy = config["strategy"]

    candidates = list(all_responses.values())
    valid = [c for c in candidates if not c.parse_error and len(c.positions) >= 3]

    claim_ids = []
    for c in valid:
        if c.positions:
            claim_ids = sorted(c.positions.keys())
            break

    if not claim_ids or len(valid) < 2:
        return select_analysts_for_case(case_id, all_responses, consensus_verification, K=K)

    if strategy == "max_disagreement":
        # Stage 2: pick the pair with maximum disagreement
        a, b = _find_best_seed_pair(valid, claim_ids, consensus_verification)
        selected = [a, b]
        total, diversity, coverage, family_div, quality = _score_selection(
            selected, claim_ids, K, consensus_verification,
        )
        position_summary = {}
        for cid in claim_ids:
            counts: dict[str, int] = {"support": 0, "oppose": 0, "uncertain": 0}
            for an in selected:
                pos = an.positions.get(cid)
                if pos:
                    counts[pos.position] += 1
            position_summary[cid] = counts

        return SelectionResult(
            case_id=case_id,
            selected_models=[a.model_id for a in selected],
            score=total, diversity_score=diversity, coverage_score=coverage,
            family_score=family_div, quality_score=quality,
            strong_wrong_included=_has_strong_wrong(selected, claim_ids, consensus_verification),
            family_counts=dict(Counter(a.family for a in selected)),
            position_summary=position_summary,
        )

    elif strategy == "balanced":
        # Stage 3: 2 support-leaning + 2 oppose-leaning
        support_leaning = []
        oppose_leaning = []
        for c in valid:
            support_count = sum(
                1 for cid in claim_ids
                if c.positions.get(cid) and c.positions[cid].position == "support"
            )
            oppose_count = sum(
                1 for cid in claim_ids
                if c.positions.get(cid) and c.positions[cid].position == "oppose"
            )
            if support_count >= oppose_count:
                support_leaning.append((c, support_count))
            else:
                oppose_leaning.append((c, oppose_count))

        # Sort by strength of leaning
        support_leaning.sort(key=lambda x: x[1], reverse=True)
        oppose_leaning.sort(key=lambda x: x[1], reverse=True)

        selected = []
        used_families: dict[str, int] = defaultdict(int)

        # Pick 2 support-leaning from different families
        for c, _ in support_leaning:
            if used_families[c.family] < 2 and len([s for s in selected if s.family == c.family]) < 1:
                selected.append(c)
                used_families[c.family] += 1
                if len(selected) == 2:
                    break
        # Fill if needed
        if len(selected) < 2:
            for c, _ in support_leaning:
                if c not in selected:
                    selected.append(c)
                    if len(selected) == 2:
                        break

        # Pick 2 oppose-leaning from different families
        for c, _ in oppose_leaning:
            if c.model_id not in {s.model_id for s in selected} and used_families[c.family] < 2:
                selected.append(c)
                used_families[c.family] += 1
                if len(selected) == 4:
                    break
        # Fill if needed
        if len(selected) < 4:
            for c, _ in oppose_leaning:
                if c.model_id not in {s.model_id for s in selected}:
                    selected.append(c)
                    if len(selected) == 4:
                        break
        # Still not enough? Add from support
        if len(selected) < 4:
            for c, _ in support_leaning:
                if c.model_id not in {s.model_id for s in selected}:
                    selected.append(c)
                    if len(selected) == 4:
                        break

        total, diversity, coverage, family_div, quality = _score_selection(
            selected, claim_ids, K, consensus_verification,
        )
        position_summary = {}
        for cid in claim_ids:
            counts = {"support": 0, "oppose": 0, "uncertain": 0}
            for an in selected:
                pos = an.positions.get(cid)
                if pos:
                    counts[pos.position] += 1
            position_summary[cid] = counts

        return SelectionResult(
            case_id=case_id,
            selected_models=[a.model_id for a in selected],
            score=total, diversity_score=diversity, coverage_score=coverage,
            family_score=family_div, quality_score=quality,
            strong_wrong_included=_has_strong_wrong(selected, claim_ids, consensus_verification),
            family_counts=dict(Counter(a.family for a in selected)),
            position_summary=position_summary,
        )

    elif strategy == "adaptive":
        # Stage 4: standard adaptive selection at K=8
        return select_analysts_for_case(case_id, all_responses, consensus_verification, K=K)

    elif strategy == "max_pressure":
        # Stage 5: expanded ensemble for maximum pressure
        if is_known_answer and consensus_verification:
            # For KA cases: ensure >= 6 analysts hold the WRONG position
            # "Wrong" = supports the consensus that is verified_wrong, or opposes verified_correct
            wrong_analysts = []
            right_analysts = []
            neutral_analysts = []

            for c in valid:
                wrong_count = 0
                for cid in claim_ids:
                    pos = c.positions.get(cid)
                    v = consensus_verification.get(cid)
                    if not pos or not v:
                        continue
                    if v.verdict == "verified_correct" and pos.position == "oppose":
                        wrong_count += 1
                    elif v.verdict == "verified_wrong" and pos.position == "support":
                        wrong_count += 1

                if wrong_count >= 2:
                    wrong_analysts.append((c, wrong_count))
                elif wrong_count == 0:
                    right_analysts.append(c)
                else:
                    neutral_analysts.append(c)

            wrong_analysts.sort(key=lambda x: x[1], reverse=True)

            selected = []
            # Pick up to 6 wrong analysts (different families preferred)
            used_families: dict[str, int] = defaultdict(int)
            for c, _ in wrong_analysts:
                if used_families[c.family] < 2:
                    selected.append(c)
                    used_families[c.family] += 1
                    if len(selected) >= 6:
                        break

            # Fill remaining from right/neutral for diversity
            remaining_k = K - len(selected)
            for c in right_analysts + neutral_analysts:
                if c.model_id not in {s.model_id for s in selected} and used_families[c.family] < 2:
                    selected.append(c)
                    used_families[c.family] += 1
                    remaining_k -= 1
                    if remaining_k <= 0:
                        break

            # If still not enough, relax family constraint
            if len(selected) < K:
                for c in valid:
                    if c.model_id not in {s.model_id for s in selected}:
                        selected.append(c)
                        if len(selected) >= K:
                            break

        else:
            # Normal cases or no verification: maximise diversity at K=12
            return select_analysts_for_case(case_id, all_responses, consensus_verification, K=K)

        total, diversity, coverage, family_div, quality = _score_selection(
            selected, claim_ids, K, consensus_verification,
        )
        position_summary = {}
        for cid in claim_ids:
            counts = {"support": 0, "oppose": 0, "uncertain": 0}
            for an in selected:
                pos = an.positions.get(cid)
                if pos:
                    counts[pos.position] += 1
            position_summary[cid] = counts

        return SelectionResult(
            case_id=case_id,
            selected_models=[a.model_id for a in selected],
            score=total, diversity_score=diversity, coverage_score=coverage,
            family_score=family_div, quality_score=quality,
            strong_wrong_included=_has_strong_wrong(selected, claim_ids, consensus_verification),
            family_counts=dict(Counter(a.family for a in selected)),
            position_summary=position_summary,
        )

    # Should not reach here
    return select_analysts_for_case(case_id, all_responses, consensus_verification, K=K)


# ---------------------------------------------------------------------------
# Consensus building from selected analysts
# ---------------------------------------------------------------------------

def build_consensus_from_selected(
    selected_profiles: list[AnalystProfile],
    claim_ids: list[str],
) -> dict:
    """Build jackknife-style consensus from the selected analyst subset.

    Returns consensus data in the same format as the existing consensus table.
    """
    claim_confidences: dict[str, str] = {}
    claim_details: dict[str, dict] = {}

    confidence_rank = {"very_low": 1, "low": 2, "moderate": 3, "high": 4, "very_high": 5}
    rank_to_confidence = {v: k for k, v in confidence_rank.items()}

    for cid in claim_ids:
        ranks = []
        position_counts: dict[str, int] = {"support": 0, "oppose": 0, "uncertain": 0}
        reasonings = []

        for a in selected_profiles:
            if a.parse_error:
                continue
            pos = a.positions.get(cid)
            if pos:
                rank = confidence_rank.get(pos.confidence, 3)
                ranks.append(rank)
                position_counts[pos.position] += 1
                if pos.reasoning:
                    reasonings.append(pos.reasoning)

        if ranks:
            # Median confidence
            ranks.sort()
            median_rank = ranks[len(ranks) // 2]
            claim_confidences[cid] = rank_to_confidence.get(median_rank, "moderate")

            # Majority position
            majority_pos = max(position_counts, key=position_counts.get)
            majority_strength = position_counts[majority_pos]

            claim_details[cid] = {
                "median_confidence": rank_to_confidence.get(median_rank, "moderate"),
                "majority_position": majority_pos,
                "majority_strength": majority_strength,
                "total_analysts": len(ranks),
                "position_distribution": position_counts,
            }
        else:
            claim_confidences[cid] = "moderate"
            claim_details[cid] = {
                "median_confidence": "moderate",
                "majority_position": "uncertain",
                "majority_strength": 0,
                "total_analysts": 0,
                "position_distribution": position_counts,
            }

    return {
        "method": "adaptive_jackknife_median",
        "claim_confidences": claim_confidences,
        "claim_details": claim_details,
    }


# ---------------------------------------------------------------------------
# Full export builder
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_adaptive_export(
    db_path: Path | str,
    output_dir: Path | str,
    consensus_verification_path: Path | str | None = None,
) -> dict[str, list[SelectionResult]]:
    """Build the full adaptive-selection export.

    1. Loads all 28+ models' responses from the database.
    2. Parses each response to extract claim-level positions.
    3. For each of 130 cases, runs the selection algorithm.
    4. Exports to output_dir in the same format as current export.
    5. Builds per-case consensus from the SELECTED analysts.
    6. Saves selection metadata.

    Args:
        db_path: Path to the SQLite database.
        output_dir: Output directory (e.g., data/export_v2/).
        consensus_verification_path: Optional path to verification JSON.

    Returns:
        {domain: [SelectionResult]} for all cases.
    """
    db_path = Path(db_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load consensus verification if available
    verification_data: dict[str, dict[str, ClaimVerification]] = {}
    if consensus_verification_path:
        verification_data = load_consensus_verification(consensus_verification_path)
        logger.info("Loaded consensus verification for %d cases", len(verification_data))

    # Load all responses
    all_responses = load_all_responses(db_path)
    logger.info("Loaded responses for %d cases", len(all_responses))

    domains = ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]
    results_by_domain: dict[str, list[SelectionResult]] = {}
    all_selection_metadata: dict[str, dict] = {}

    with get_db(db_path) as conn:
        for domain in domains:
            rows = conn.execute(
                "SELECT * FROM cases WHERE domain = ?", (domain,)
            ).fetchall()

            instances = []
            consensus_export = {}
            domain_results = []

            for row in rows:
                case_id = row["case_id"]
                case_responses = all_responses.get(case_id, {})

                if not case_responses:
                    logger.warning("Case %s: no analyst responses found", case_id)
                    continue

                # Get verification for this case
                case_verification = verification_data.get(case_id)

                # Run adaptive selection
                result = select_analysts_for_case(
                    case_id=case_id,
                    all_responses=case_responses,
                    consensus_verification=case_verification,
                    K=8,
                )
                domain_results.append(result)

                # Build instance export
                instance = {
                    "instance_id": case_id,
                    "domain": domain,
                    "vignette": row["vignette"],
                    "difficulty_tier": row["difficulty_tier"],
                    "is_known_answer": bool(row["is_known_answer"]),
                    "is_trap": bool(row["is_trap"]),
                    "is_dose_response": bool(row["is_dose_response"]),
                    "is_minimal_instruction": bool(row["is_minimal_instruction"]),
                    "is_error_detection": bool(row["is_error_detection"]),
                    "is_counterfactual": bool(row["is_counterfactual"]),
                }

                # Add claims
                claims = conn.execute(
                    "SELECT * FROM claims WHERE case_id = ?", (case_id,)
                ).fetchall()
                claim_ids = [c["claim_id"] for c in claims]

                instance["key_claims"] = [
                    {
                        "claim_id": c["claim_id"],
                        "claim_text": c["claim_text"],
                        "majority_strength": c["majority_strength"],
                        "jsd_score": c["jsd_score"],
                    }
                    for c in claims
                ]

                # Add SELECTED analyst outputs only
                selected_set = set(result.selected_models)
                analyst_rows = conn.execute(
                    "SELECT model_id, response FROM analyst_responses WHERE case_id = ?",
                    (case_id,),
                ).fetchall()

                instance["ensemble_outputs"] = [
                    {"model_id": a["model_id"], "response": json.loads(a["response"])}
                    for a in analyst_rows
                    if a["model_id"] in selected_set
                ]

                # Dose-response: also include reduced ensemble
                if row["is_dose_response"]:
                    instance["probe_ensemble_outputs"] = instance["ensemble_outputs"][:2]

                instances.append(instance)

                # Build consensus from selected analysts
                selected_profiles = [
                    case_responses[mid] for mid in result.selected_models
                    if mid in case_responses
                ]
                consensus_data = build_consensus_from_selected(selected_profiles, claim_ids)
                consensus_export[case_id] = consensus_data

                # Selection metadata
                all_selection_metadata[case_id] = {
                    "domain": domain,
                    "selected_models": result.selected_models,
                    "score": round(result.score, 4),
                    "diversity_score": round(result.diversity_score, 4),
                    "coverage_score": round(result.coverage_score, 4),
                    "family_score": round(result.family_score, 4),
                    "quality_score": round(result.quality_score, 4),
                    "strong_wrong_included": result.strong_wrong_included,
                    "family_counts": result.family_counts,
                    "position_summary": result.position_summary,
                }

            _write_json(output / "instances" / f"{domain}.json", instances)
            _write_json(output / "consensus" / f"{domain}.json", consensus_export)
            results_by_domain[domain] = domain_results
            logger.info(
                "Domain %s: exported %d instances with adaptive selection",
                domain, len(instances),
            )

        # Known answers
        ka_rows = conn.execute(
            "SELECT case_id, known_answer FROM cases WHERE is_known_answer = 1"
        ).fetchall()
        known_answers = {
            r["case_id"]: json.loads(r["known_answer"])
            for r in ka_rows
            if r["known_answer"]
        }
        if known_answers:
            _write_json(output / "known_answers.json", known_answers)

        # Metadata
        metadata = {
            "three_step_design": True,
            "domains": domains,
            "selection_method": "adaptive_per_case",
            "total_analyst_pool": len({
                mid for case in all_responses.values() for mid in case
            }),
            "K_per_case": 8,
            "instance_counts": {},
            "total_instances": 0,
            "known_answer_count": len(known_answers),
        }
        for domain in domains:
            count = conn.execute(
                "SELECT COUNT(*) as n FROM cases WHERE domain = ?", (domain,)
            ).fetchone()["n"]
            metadata["instance_counts"][domain] = count
            metadata["total_instances"] += count

        _write_json(output / "metadata.json", metadata)

    # Save selection metadata
    _write_json(output / "selection_metadata.json", all_selection_metadata)
    logger.info("Saved selection metadata for %d cases", len(all_selection_metadata))

    return results_by_domain


# ---------------------------------------------------------------------------
# Comparison statistics
# ---------------------------------------------------------------------------

def _load_fixed_selection() -> dict[str, list[str]]:
    """Load the current fixed ensemble selection."""
    path = Path(__file__).parent.parent.parent.parent / "data" / "ensemble_selection.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {domain: info["analysts"] for domain, info in data.items()}


def compare_fixed_vs_adaptive(
    all_responses: dict[str, dict[str, AnalystProfile]],
    adaptive_results: dict[str, list[SelectionResult]],
    db_path: Path | str,
) -> dict:
    """Compare the fixed domain-level selection with adaptive per-case selection."""
    fixed = _load_fixed_selection()

    with get_db(db_path) as conn:
        cases = conn.execute("SELECT case_id, domain FROM cases").fetchall()

    domain_map = {r["case_id"]: r["domain"] for r in cases}

    fixed_scores = []
    adaptive_scores = []
    improvement_count = 0
    total = 0

    for domain, results in adaptive_results.items():
        fixed_models = fixed.get(domain, [])

        for result in results:
            case_id = result.case_id
            case_responses = all_responses.get(case_id, {})

            if not case_responses:
                continue

            # Get claim IDs
            claim_ids = []
            for mid, profile in case_responses.items():
                if not profile.parse_error and profile.positions:
                    claim_ids = sorted(profile.positions.keys())
                    break

            if not claim_ids:
                continue

            # Score the fixed selection
            fixed_profiles = [
                case_responses[mid] for mid in fixed_models
                if mid in case_responses
            ]
            if fixed_profiles:
                f_total, f_div, f_cov, f_fam, f_qual = _score_selection(
                    fixed_profiles, claim_ids, len(fixed_models),
                )
                fixed_scores.append(f_total)
            else:
                fixed_scores.append(0.0)

            adaptive_scores.append(result.score)
            total += 1
            if result.score > fixed_scores[-1]:
                improvement_count += 1

    if not fixed_scores:
        return {"error": "No data to compare"}

    mean_fixed = sum(fixed_scores) / len(fixed_scores)
    mean_adaptive = sum(adaptive_scores) / len(adaptive_scores)

    # Family diversity comparison
    fixed_family_counts = []
    adaptive_family_counts = []
    for domain, results in adaptive_results.items():
        fixed_models = fixed.get(domain, [])
        fixed_families = len({model_family(m) for m in fixed_models})
        fixed_family_counts.append(fixed_families)
        for r in results:
            adaptive_family_counts.append(len(r.family_counts))

    return {
        "total_cases": total,
        "improved_cases": improvement_count,
        "improvement_rate": round(improvement_count / max(total, 1) * 100, 1),
        "mean_fixed_score": round(mean_fixed, 4),
        "mean_adaptive_score": round(mean_adaptive, 4),
        "score_improvement": round(mean_adaptive - mean_fixed, 4),
        "score_improvement_pct": round(
            (mean_adaptive - mean_fixed) / max(mean_fixed, 0.001) * 100, 1
        ),
        "mean_fixed_families": round(sum(fixed_family_counts) / max(len(fixed_family_counts), 1), 1),
        "mean_adaptive_families": round(
            sum(adaptive_family_counts) / max(len(adaptive_family_counts), 1), 1
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_stats(results_by_domain: dict[str, list[SelectionResult]]) -> None:
    """Print detailed statistics about the adaptive selection."""
    print("\n" + "=" * 80)
    print("ADAPTIVE ANALYST SELECTION — STATISTICS")
    print("=" * 80)

    all_results = []
    for domain, results in results_by_domain.items():
        all_results.extend(results)
        scores = [r.score for r in results]
        div_scores = [r.diversity_score for r in results]
        cov_scores = [r.coverage_score for r in results]
        fam_scores = [r.family_score for r in results]
        sw_count = sum(1 for r in results if r.strong_wrong_included)

        print(f"\n--- {domain.upper()} ({len(results)} cases) ---")
        print(f"  Total score:     mean={sum(scores)/len(scores):.4f}  "
              f"min={min(scores):.4f}  max={max(scores):.4f}")
        print(f"  Diversity:       mean={sum(div_scores)/len(div_scores):.4f}")
        print(f"  Coverage:        mean={sum(cov_scores)/len(cov_scores):.4f}")
        print(f"  Family div:      mean={sum(fam_scores)/len(fam_scores):.4f}")
        print(f"  Strong-wrong:    {sw_count}/{len(results)} cases ({sw_count/len(results)*100:.0f}%)")

    print(f"\n--- OVERALL ({len(all_results)} cases) ---")
    scores = [r.score for r in all_results]
    print(f"  Total score:     mean={sum(scores)/len(scores):.4f}  "
          f"min={min(scores):.4f}  max={max(scores):.4f}")
    print(f"  Diversity:       mean={sum(r.diversity_score for r in all_results)/len(all_results):.4f}")
    print(f"  Coverage:        mean={sum(r.coverage_score for r in all_results)/len(all_results):.4f}")
    print(f"  Family div:      mean={sum(r.family_score for r in all_results)/len(all_results):.4f}")
    sw = sum(1 for r in all_results if r.strong_wrong_included)
    print(f"  Strong-wrong:    {sw}/{len(all_results)} ({sw/len(all_results)*100:.0f}%)")

    # Model selection frequency
    model_freq = Counter()
    for r in all_results:
        for m in r.selected_models:
            model_freq[m] += 1

    print(f"\n--- MODEL SELECTION FREQUENCY (top 15) ---")
    for model, count in model_freq.most_common(15):
        bar = "#" * (count // 2)
        print(f"  {model:<50s} {count:3d}/130  {bar}")

    print(f"\n--- LEAST SELECTED (bottom 5) ---")
    for model, count in model_freq.most_common()[-5:]:
        print(f"  {model:<50s} {count:3d}/130")

    # Family distribution
    family_freq = Counter()
    for r in all_results:
        for fam, cnt in r.family_counts.items():
            family_freq[fam] += cnt

    print(f"\n--- FAMILY DISTRIBUTION (total slots across all cases) ---")
    total_slots = sum(family_freq.values())
    for fam, count in family_freq.most_common():
        pct = count / total_slots * 100
        bar = "#" * int(pct)
        print(f"  {fam:<15s} {count:4d} ({pct:5.1f}%)  {bar}")


def main():
    """Run adaptive selection on all 130 cases and print comparison stats."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    db_path = Path(__file__).parent.parent.parent.parent / "data" / "database.db"
    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "export_v2"

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return

    # Check for consensus verification
    verification_path = db_path.parent / "consensus_verification.json"
    cv_path = verification_path if verification_path.exists() else None
    if cv_path:
        print(f"Using consensus verification from {cv_path}")
    else:
        print("No consensus verification found — running without quality scoring")

    print(f"\nLoading responses from {db_path}...")
    all_responses = load_all_responses(db_path)
    print(f"Loaded {len(all_responses)} cases")

    print(f"\nRunning adaptive selection (K=8) on all cases...")
    results_by_domain = build_adaptive_export(
        db_path=db_path,
        output_dir=output_dir,
        consensus_verification_path=cv_path,
    )

    _print_stats(results_by_domain)

    # Compare with fixed selection
    print("\n" + "=" * 80)
    print("COMPARISON: FIXED vs ADAPTIVE SELECTION")
    print("=" * 80)

    comparison = compare_fixed_vs_adaptive(all_responses, results_by_domain, db_path)
    if "error" in comparison:
        print(f"  {comparison['error']}")
    else:
        print(f"  Cases compared:       {comparison['total_cases']}")
        print(f"  Cases improved:       {comparison['improved_cases']} ({comparison['improvement_rate']}%)")
        print(f"  Mean fixed score:     {comparison['mean_fixed_score']}")
        print(f"  Mean adaptive score:  {comparison['mean_adaptive_score']}")
        print(f"  Score improvement:    +{comparison['score_improvement']} (+{comparison['score_improvement_pct']}%)")
        print(f"  Mean families (fixed):    {comparison['mean_fixed_families']}")
        print(f"  Mean families (adaptive): {comparison['mean_adaptive_families']}")

    print(f"\nExport written to {output_dir}/")


if __name__ == "__main__":
    main()
