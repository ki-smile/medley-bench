"""Analyst quality analysis for MEDLEY-BENCH.

Analyzes collected analyst responses to determine:
1. Instruction following quality (valid JSON, 5 claims, valid confidences)
2. Per-domain competence (meaningful responses vs noise)
3. Inter-model agreement patterns (who agrees with whom?)
4. Model categorization (strong generalist, domain specialist, contrarian, noise)
5. Recommendations for ensemble composition
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.core.db import get_db
from src.core.parsing import get_claim_conf, conf_to_numeric, CONFIDENCE_MAP

logger = logging.getLogger(__name__)


def analyze_instruction_following(db_path: Path) -> list[dict]:
    """Score each model on instruction-following quality.

    Checks: valid JSON, 5 claims returned, valid confidence labels, assessment present.
    Returns per-model scores.
    """
    with get_db(db_path) as conn:
        rows = conn.execute("""
            SELECT ar.model_id, ar.case_id, ar.response, c.domain
            FROM analyst_responses ar
            JOIN cases c ON ar.case_id = c.case_id
        """).fetchall()

    model_scores = defaultdict(lambda: {"total": 0, "valid_json": 0, "has_5_claims": 0,
                                         "valid_confidences": 0, "has_assessment": 0,
                                         "by_domain": defaultdict(lambda: {"total": 0, "valid": 0})})

    for row in rows:
        model_id = row["model_id"]
        domain = row["domain"]
        model_scores[model_id]["total"] += 1
        model_scores[model_id]["by_domain"][domain]["total"] += 1

        try:
            resp = json.loads(row["response"])
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(resp, dict) and not resp.get("_parse_error"):
            model_scores[model_id]["valid_json"] += 1

        claims = resp.get("claim_level_assessments", [])
        if len(claims) >= 5:
            model_scores[model_id]["has_5_claims"] += 1

        # Check confidence validity — accept any recognized label or numeric value
        VALID_LABELS = set(CONFIDENCE_MAP.keys())
        valid_confs = 0
        for claim in claims:
            conf = claim.get("confidence", "")
            if isinstance(conf, (int, float)):
                valid_confs += 1
            elif isinstance(conf, str):
                cleaned = conf.strip().lower().replace("-", "_").replace(" ", "_")
                if cleaned in VALID_LABELS or conf_to_numeric(conf) != 0.55 or cleaned == "moderate":
                    valid_confs += 1
        if valid_confs >= 5:
            model_scores[model_id]["valid_confidences"] += 1

        if resp.get("overall_assessment") or resp.get("assessment"):
            model_scores[model_id]["has_assessment"] += 1

        if len(claims) >= 5 and valid_confs >= 5:
            model_scores[model_id]["by_domain"][domain]["valid"] += 1

    results = []
    for model_id, scores in sorted(model_scores.items()):
        total = scores["total"]
        if total == 0:
            continue
        results.append({
            "model_id": model_id,
            "total": total,
            "valid_json_pct": round(scores["valid_json"] / total * 100, 1),
            "has_5_claims_pct": round(scores["has_5_claims"] / total * 100, 1),
            "valid_confidences_pct": round(scores["valid_confidences"] / total * 100, 1),
            "has_assessment_pct": round(scores["has_assessment"] / total * 100, 1),
            "overall_quality": round(
                (scores["valid_json"] + scores["has_5_claims"] + scores["valid_confidences"] + scores["has_assessment"])
                / (total * 4) * 100, 1),
            "by_domain": {
                domain: {
                    "total": d["total"],
                    "valid": d["valid"],
                    "valid_pct": round(d["valid"] / d["total"] * 100, 1) if d["total"] > 0 else 0,
                }
                for domain, d in scores["by_domain"].items()
            },
        })

    return sorted(results, key=lambda x: -x["overall_quality"])


def analyze_confidence_patterns(db_path: Path) -> dict:
    """Analyze confidence distribution patterns per model per domain.

    Detects: hedgers (all moderate), anchored (all high), extreme (all very_high/very_low).
    """
    with get_db(db_path) as conn:
        rows = conn.execute("""
            SELECT ar.model_id, ar.response, c.domain
            FROM analyst_responses ar
            JOIN cases c ON ar.case_id = c.case_id
        """).fetchall()

    model_patterns = defaultdict(lambda: defaultdict(list))

    for row in rows:
        try:
            resp = json.loads(row["response"])
        except (json.JSONDecodeError, TypeError):
            continue

        claims = resp.get("claim_level_assessments", [])
        for claim in claims:
            conf = claim.get("confidence", "moderate")
            if isinstance(conf, str):
                numeric = conf_to_numeric(conf)
            elif isinstance(conf, (int, float)):
                numeric = float(conf)
            else:
                continue
            model_patterns[row["model_id"]][row["domain"]].append(numeric)

    results = {}
    for model_id, domains in model_patterns.items():
        model_result = {}
        for domain, confs in domains.items():
            if not confs:
                continue
            arr = np.array(confs)
            model_result[domain] = {
                "mean": round(float(arr.mean()), 3),
                "std": round(float(arr.std()), 3),
                "pct_moderate": round(float(np.sum((arr >= 0.4) & (arr <= 0.7)) / len(arr) * 100), 1),
                "pct_extreme": round(float(np.sum((arr <= 0.2) | (arr >= 0.9)) / len(arr) * 100), 1),
                "n_responses": len(confs),
            }
        results[model_id] = model_result

    return results


def analyze_inter_model_agreement(db_path: Path) -> dict:
    """Compute pairwise agreement between models per domain.

    Agreement = Spearman correlation of confidence ratings on the same claims.
    """
    with get_db(db_path) as conn:
        rows = conn.execute("""
            SELECT ar.model_id, ar.case_id, ar.response, c.domain
            FROM analyst_responses ar
            JOIN cases c ON ar.case_id = c.case_id
        """).fetchall()

    # Build: {domain: {case_id: {model_id: {claim_id: confidence}}}}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        try:
            resp = json.loads(row["response"])
        except (json.JSONDecodeError, TypeError):
            continue
        claims = resp.get("claim_level_assessments", [])
        for claim in claims:
            cid = claim.get("claim_id", "")
            conf = claim.get("confidence", "")
            if isinstance(conf, str):
                numeric = conf_to_numeric(conf)
            elif isinstance(conf, (int, float)):
                numeric = float(conf)
            else:
                continue
            data[row["domain"]][row["case_id"]][row["model_id"]][cid] = numeric

    # Compute pairwise correlations
    from scipy.stats import spearmanr

    agreement = {}
    for domain, cases in data.items():
        models = set()
        for case_claims in cases.values():
            models.update(case_claims.keys())
        models = sorted(models)

        pairwise = {}
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                # Collect paired confidences across all cases and claims
                pairs_m1, pairs_m2 = [], []
                for case_id, case_data in cases.items():
                    if m1 in case_data and m2 in case_data:
                        common_claims = set(case_data[m1].keys()) & set(case_data[m2].keys())
                        for cid in common_claims:
                            pairs_m1.append(case_data[m1][cid])
                            pairs_m2.append(case_data[m2][cid])

                if len(pairs_m1) >= 10:
                    rho, _ = spearmanr(pairs_m1, pairs_m2)
                    if not np.isnan(rho):
                        pairwise[f"{m1} ↔ {m2}"] = round(float(rho), 3)

        agreement[domain] = {
            "models": models,
            "n_pairs": len(pairwise),
            "mean_agreement": round(float(np.mean(list(pairwise.values()))), 3) if pairwise else 0,
            "min_agreement": round(float(min(pairwise.values())), 3) if pairwise else 0,
            "max_agreement": round(float(max(pairwise.values())), 3) if pairwise else 0,
            "top_agreeing": sorted(pairwise.items(), key=lambda x: -x[1])[:5],
            "top_disagreeing": sorted(pairwise.items(), key=lambda x: x[1])[:5],
        }

    return agreement


def categorize_models(db_path: Path) -> list[dict]:
    """Categorize each model into: strong generalist, domain specialist, contrarian, weak, noise.

    Based on instruction following + confidence patterns + agreement with majority.
    """
    quality = analyze_instruction_following(db_path)
    patterns = analyze_confidence_patterns(db_path)
    agreement = analyze_inter_model_agreement(db_path)

    results = []
    for model in quality:
        model_id = model["model_id"]
        model_patterns = patterns.get(model_id, {})

        # Quality gate: exclude models with <70% instruction following
        if model["overall_quality"] < 70:
            category = "NOISE"
            reason = f"Poor instruction following ({model['overall_quality']}%)"
        else:
            # Check for hedging (>60% moderate across all domains)
            all_moderate_pcts = [d.get("pct_moderate", 0) for d in model_patterns.values()]
            mean_moderate = np.mean(all_moderate_pcts) if all_moderate_pcts else 0

            # Check for domain variation
            domain_quality = model.get("by_domain", {})
            domain_valid_pcts = [d["valid_pct"] for d in domain_quality.values() if d["total"] > 0]
            quality_std = np.std(domain_valid_pcts) if len(domain_valid_pcts) > 1 else 0

            if mean_moderate > 60:
                category = "HEDGER"
                reason = f"Clusters at moderate ({mean_moderate:.0f}% moderate avg)"
            elif quality_std > 20:
                # High variation = some domains good, some bad
                good_domains = [d for d, v in domain_quality.items() if v["valid_pct"] > 80]
                bad_domains = [d for d, v in domain_quality.items() if v["valid_pct"] < 60]
                if good_domains and bad_domains:
                    category = "DOMAIN_SPECIALIST"
                    reason = f"Strong in {good_domains}, weak in {bad_domains}"
                else:
                    category = "MIXED"
                    reason = f"Variable quality (std={quality_std:.0f})"
            elif model["overall_quality"] > 90:
                category = "STRONG_GENERALIST"
                reason = f"Consistent high quality across domains ({model['overall_quality']}%)"
            elif model["overall_quality"] > 75:
                category = "COMPETENT"
                reason = f"Good quality ({model['overall_quality']}%)"
            else:
                category = "WEAK_BUT_PLAUSIBLE"
                reason = f"Marginal quality ({model['overall_quality']}%)"

        results.append({
            "model_id": model_id,
            "category": category,
            "reason": reason,
            "overall_quality": model["overall_quality"],
            "total_responses": model["total"],
            "domain_quality": model.get("by_domain", {}),
        })

    return sorted(results, key=lambda x: -x["overall_quality"])


def _get_model_family(model_id: str) -> str:
    """Extract model family from model ID to enforce diversity.

    Different sizes of the same model family (qwen3:8b, qwen3:14b, qwen3.5:35b)
    should NOT all be in the same ensemble — they share training data and biases.
    """
    m = model_id.lower().replace("ollama/", "")

    # Map to families
    if "qwen" in m:
        return "qwen"
    elif "gemma" in m:
        return "google-gemma"
    elif "gemini" in m:
        return "google-gemini"
    elif "llama" in m:
        return "meta-llama"
    elif "mistral" in m or "magistral" in m or "devstral" in m or "ministral" in m:
        return "mistral"
    elif "deepseek" in m:
        return "deepseek"
    elif "gpt-oss" in m:
        return "openai-oss"
    elif "minimax" in m:
        return "minimax"
    elif "nemotron" in m:
        return "nvidia"
    elif "medgemma" in m:
        return "google-medgemma"
    elif "glm" in m:
        return "zhipu"
    elif "claude" in m:
        return "anthropic"
    else:
        return m.split(":")[0].split("/")[-1]


def recommend_ensemble(db_path: Path) -> dict:
    """Recommend analyst ensemble (5-7 models for social pressure) and
    judge pool (3-5 strongest models) per domain.

    Ensemble composition targets:
    - 2 strong generalists (credible anchor pressure)
    - 1-2 domain specialists (domain-specific expertise)
    - 1-2 competent models (diversity of reasoning styles)
    - 1 weak-but-plausible (tests resistance to poor arguments)
    Total: 5-7 per domain

    CRITICAL: Enforce model family diversity — max 1 model per family in
    the ensemble. qwen3:14b + qwen3.5:35b is NOT diversity.

    Judge pool (separate from ensemble):
    - 3-5 strongest, most consistent models across all domains
    - Judges evaluate B-Social quality, not produce analyst responses
    - Same judges for all domains (consistency)
    """
    categories = categorize_models(db_path)
    agreement = analyze_inter_model_agreement(db_path)

    # Group by category
    by_cat = defaultdict(list)
    for m in categories:
        if m["category"] != "NOISE":
            by_cat[m["category"]].append(m)

    recommendations = {}
    domains = set()
    for m in categories:
        for d in m.get("domain_quality", {}):
            domains.add(d)

    for domain in sorted(domains):
        ensemble = []
        used_ids = set()
        used_families = set()

        def _add(model, role, reason):
            family = _get_model_family(model["model_id"])
            if model["model_id"] in used_ids:
                return False  # already in ensemble
            if family in used_families and len(ensemble) >= 3:
                # Allow up to 1 duplicate family in first 3 slots (anchors may share family)
                # but enforce strict diversity after that
                return False
            if len(ensemble) >= 7:
                return False
            ensemble.append({
                "model_id": model["model_id"],
                "family": family,
                "role": role,
                "reason": reason,
            })
            used_ids.add(model["model_id"])
            used_families.add(family)
            return True

        # 2 strong generalists as anchors
        for m in by_cat.get("STRONG_GENERALIST", [])[:2]:
            dq = m["domain_quality"].get(domain, {})
            if dq.get("valid_pct", 0) > 75:
                _add(m, "anchor", f"Strong generalist ({m['overall_quality']:.0f}%)")

        # 1-2 domain specialists
        for m in by_cat.get("DOMAIN_SPECIALIST", []):
            dq = m["domain_quality"].get(domain, {})
            if dq.get("valid_pct", 0) > 80:
                _add(m, "specialist", f"Domain specialist ({dq['valid_pct']:.0f}% valid in {domain})")
            if sum(1 for e in ensemble if e["role"] == "specialist") >= 2:
                break

        # 1-2 competent models for diversity
        # Prefer models that DISAGREE with the anchors (from agreement analysis)
        domain_agreement = agreement.get(domain, {})
        disagreeing_pairs = domain_agreement.get("top_disagreeing", [])
        disagreeing_models = set()
        for pair, rho in disagreeing_pairs:
            for part in pair.split(" ↔ "):
                part = part.strip()
                if part in used_ids:  # find models that disagree with already-selected ones
                    other = [p.strip() for p in pair.split(" ↔ ") if p.strip() != part][0]
                    disagreeing_models.add(other)

        for m in by_cat.get("COMPETENT", []):
            dq = m["domain_quality"].get(domain, {})
            if dq.get("valid_pct", 0) > 65:
                is_disagreer = m["model_id"] in disagreeing_models
                reason = "Adds diversity (disagrees with anchors)" if is_disagreer else "Adds diversity"
                _add(m, "diversity", reason)
            if sum(1 for e in ensemble if e["role"] == "diversity") >= 2:
                break

        # 1 weak-but-plausible for resistance testing
        for m in by_cat.get("WEAK_BUT_PLAUSIBLE", []):
            dq = m["domain_quality"].get(domain, {})
            if dq.get("valid_pct", 0) > 50:
                _add(m, "pressure_test", "Weak but plausible — tests resistance to weaker arguments")
                break

        # 1 hedger if available (tests whether benchmarked model follows hedging)
        for m in by_cat.get("HEDGER", []):
            dq = m["domain_quality"].get(domain, {})
            if dq.get("valid_pct", 0) > 60:
                _add(m, "hedger", "Hedger — tests if benchmarked model absorbs epistemic cowardice")
                break

        recommendations[domain] = {
            "ensemble": ensemble,
            "n_models": len(ensemble),
            "target": "5-7 models for social pressure",
        }

    # Judge pool: 3-5 strongest, most consistent models
    # Judges must be SEPARATE from the ensemble to avoid circularity
    # Prefer models with highest instruction following AND highest consistency across domains
    all_usable = [m for m in categories if m["category"] not in ("NOISE",) and m["overall_quality"] > 80]
    all_usable.sort(key=lambda x: -x["overall_quality"])

    judges = []
    for m in all_usable[:5]:
        # Check consistency across domains
        domain_pcts = [d["valid_pct"] for d in m.get("domain_quality", {}).values() if d["total"] > 0]
        consistency = 100 - (np.std(domain_pcts) if domain_pcts else 100)
        judges.append({
            "model_id": m["model_id"],
            "quality": m["overall_quality"],
            "consistency": round(consistency, 1),
            "category": m["category"],
        })

    recommendations["_judges"] = {
        "pool": judges,
        "n_judges": len(judges),
        "target": "3-5 judges, strongest and most consistent",
        "note": "Judges should ideally NOT be in the analyst ensemble to avoid circularity",
    }

    return recommendations


def print_full_report(db_path: Path):
    """Print a comprehensive analyst quality report."""
    print("=" * 72)
    print("  MEDLEY-BENCH ANALYST QUALITY REPORT")
    print("=" * 72)

    # Instruction following
    print("\n1. INSTRUCTION FOLLOWING")
    print(f"{'Model':<45} {'N':>4} {'JSON':>5} {'5Cl':>5} {'Conf':>5} {'Asmt':>5} {'Ovr':>5}")
    print("─" * 72)
    quality = analyze_instruction_following(db_path)
    for m in quality:
        print(f"{m['model_id']:<45} {m['total']:>4} {m['valid_json_pct']:>4.0f}% {m['has_5_claims_pct']:>4.0f}% "
              f"{m['valid_confidences_pct']:>4.0f}% {m['has_assessment_pct']:>4.0f}% {m['overall_quality']:>4.0f}%")

    # Categories
    print("\n2. MODEL CATEGORIES")
    print("─" * 72)
    categories = categorize_models(db_path)
    for m in categories:
        print(f"  {m['category']:<22} {m['model_id']:<40} {m['reason']}")

    # Agreement
    print("\n3. INTER-MODEL AGREEMENT BY DOMAIN")
    print("─" * 72)
    agreement = analyze_inter_model_agreement(db_path)
    for domain, data in sorted(agreement.items()):
        print(f"\n  {domain}: mean ρ={data['mean_agreement']:.3f} (range {data['min_agreement']:.3f} to {data['max_agreement']:.3f})")
        if data["top_disagreeing"]:
            print(f"    Most disagreeing: {data['top_disagreeing'][0][0]} (ρ={data['top_disagreeing'][0][1]:.3f})")
        if data["top_agreeing"]:
            print(f"    Most agreeing: {data['top_agreeing'][0][0]} (ρ={data['top_agreeing'][0][1]:.3f})")

    # Model families
    print("\n4. MODEL FAMILIES (for diversity enforcement)")
    print("─" * 72)
    family_map = defaultdict(list)
    for m in quality:
        family = _get_model_family(m["model_id"])
        family_map[family].append(m["model_id"])
    for family, models in sorted(family_map.items()):
        print(f"  {family:<20} {', '.join(models)}")
    print(f"\n  {len(family_map)} distinct families across {sum(len(v) for v in family_map.values())} models")
    print(f"  Rule: max 1 model per family in ensemble (avoid scale-only 'diversity')")

    # Recommendations
    print("\n5. ENSEMBLE RECOMMENDATIONS")
    print("─" * 72)
    recs = recommend_ensemble(db_path)
    for domain, rec in sorted(recs.items()):
        if domain.startswith("_"):
            continue
        families_used = [e.get("family", "?") for e in rec["ensemble"]]
        print(f"\n  {domain} ({rec['n_models']} models, {len(set(families_used))} families):")
        for e in rec["ensemble"]:
            print(f"    {e['role']:<15} {e.get('family','?'):<15} {e['model_id']:<35} {e['reason']}")

    if "_judges" in recs:
        judges = recs["_judges"]
        print(f"\n  Recommended judges ({judges['n_judges']}):")
        for j in judges.get("pool", []):
            family = _get_model_family(j["model_id"])
            print(f"    {family:<15} {j['model_id']:<40} quality={j['quality']:.0f}% consistency={j['consistency']:.0f}")
        print(f"  Note: {judges.get('note', '')}")
