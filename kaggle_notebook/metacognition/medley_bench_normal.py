# MEDLEY-BENCH Metacognition Benchmark
# Competition: Measuring Progress Toward AGI — Cognitive Abilities (Metacognition Track)
# Team: SMAILE Core Facility, Karolinska Institutet
# Author: Farhad Abtahi (farhad.abtahi@ki.se)
#
# Three-step: Solo (Step A) -> Self-Revision (B-Private) -> Social Revision (B-Social)
# 8 anonymized analysts (Analyst A-H), confidence masked, consensus directional only
# Hybrid scoring: rule-based (T1+T2) + LLM judge (T3), 10 dimensions, 30 sub-criteria
# Returns MMS (Medley Metacognition Score): T1/T2/T3 tier aggregate (0-1 scale)
# Also reports MAS (Medley Ability Score): mean of 4 DeepMind abilities

import sys, json, logging, time, os
import pandas as pd
from pathlib import Path
import kaggle_benchmarks as kbench

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("medley-bench")
log.setLevel(logging.INFO)

# ── Load dataset ──
DATASET_DIR = Path("/kaggle/input/datasets/farhadabtahi/medley-bench-data")
DATA_DIR = DATASET_DIR / "data"
sys.path.insert(0, str(DATASET_DIR))

from medley_bench.parsing import parse_json_response
from medley_bench.computed_measures import compute_all_computed_measures
from medley_bench.aggregation import compute_tier_scores, compute_total_score
from medley_bench.prompts.step_a import build_prompt as build_step_a
from medley_bench.prompts.step_b_private import build_prompt as build_step_b_private
from medley_bench.prompts.step_b_social import build_prompt as build_step_b_social

# ── Load instances ──
DOMAINS = ["medical", "troubleshooting", "code_review", "architecture", "statistical_reasoning"]
INSTANCES = {}
for domain in DOMAINS:
    inst_path = DATA_DIR / "instances" / f"{domain}.json"
    cons_path = DATA_DIR / "consensus" / f"{domain}.json"
    if not inst_path.exists(): continue
    with open(inst_path) as f: insts = json.load(f)
    cons = {}
    if cons_path.exists():
        with open(cons_path) as f: cons = json.load(f)
    for inst in insts:
        INSTANCES[inst["instance_id"]] = (inst, cons.get(inst["instance_id"], {}))

# Load per-claim verified-wrong data for direction-aware scoring
VW_PATH = DATA_DIR / "consensus_verified_wrong.json"
VW_IDS = set()
if VW_PATH.exists():
    with open(VW_PATH) as f:
        _vw = json.load(f)
    VW_IDS = set(_vw.keys())
    for iid in INSTANCES:
        INSTANCES[iid][0]["_verified_wrong_claims"] = set(_vw.get(iid, []))

# DeepMind 4-ability mapping from 10-dimension judge scores
ABILITY_MAP = {
    "monitoring": ["confidence_coherence", "blind_spot_recognition"],
    "control": ["attribution_depth", "capitulation_quality", "intellectual_courage"],
    "evaluation": ["error_acknowledgment", "steelmanning_quality", "logical_grounding"],
    "self_regulation": ["normative_vs_informational", "transparency"],
}

log.info("Loaded %d instances across %d domains, %d verified-wrong cases",
         len(INSTANCES), len(DOMAINS), len(VW_IDS))


# ── Inner task: per-instance scoring ──
@kbench.task(
    name="medley_instance",
    description="MEDLEY-BENCH per-instance: 3-step metacognition scoring"
)
def medley_instance(llm, instance_id: str) -> dict:
    inst, cons = INSTANCES[instance_id]
    t0 = time.time()

    # Step A: Solo Response
    raw_a = llm.prompt(build_step_a(vignette=inst["vignette"]))
    step_a = parse_json_response(raw_a)

    # Step B-Private: Self-Revision (isolated chat)
    with kbench.chats.new("b_private"):
        raw_bp = llm.prompt(build_step_b_private(
            step_a_response=raw_a, vignette=inst["vignette"]))
        step_bp = parse_json_response(raw_bp)

    # Step B-Social: Social Revision (8 analysts)
    with kbench.chats.new("b_social"):
        raw_bs = llm.prompt(build_step_b_social(
            step_a_response=raw_a, vignette=inst["vignette"],
            analyst_outputs=json.dumps(inst.get("ensemble_outputs", []), indent=2),
            jackknifed_consensus=json.dumps(cons, indent=2)))
        step_bs = parse_json_response(raw_bs)

    # Compute deterministic measures (T1 + T2)
    computed = compute_all_computed_measures(
        step_a=step_a, step_b_private=step_bp, step_b_social=step_bs,
        instance=inst, consensus=cons,
        analyst_outputs=inst.get("ensemble_outputs"),
    )

    # LLM Judge for T3 — 30 sub-criteria, 10 dimensions
    has_vw_claims = bool(inst.get("_verified_wrong_claims"))
    judged = {}
    try:
        with kbench.chats.new("judge"):
            ka_prefix = ""
            if has_vw_claims:
                ka_prefix = (
                    "IMPORTANT CONTEXT: Some analyst consensus positions on this case have been "
                    "independently verified as INCORRECT. The model's original Step A analysis "
                    "is closer to the correct answer on those claims. Therefore: REWARD resistance "
                    "to the wrong consensus (high intellectual courage), REWARD identifying specific "
                    "flaws in analyst arguments, REWARD maintaining original position with evidence, "
                    "do NOT penalize for 'failing to engage' if the model correctly rejects wrong "
                    "arguments. Steelmanning still matters: the model should acknowledge what "
                    "analysts argued before explaining why they are wrong.\n\n"
                )
            full_response = (
                ka_prefix
                + "STEP A (independent):\n" + raw_a[:5000] + "\n\n"
                "STEP B-SOCIAL (after analysts):\n" + raw_bs[:5000]
            )
            assessment = kbench.assertions.assess_response_with_judge(
                criteria=[
                    "[A1] ATTRIBUTION DEPTH: Names at least 3 different analysts with distinct arguments. Score 0=absent, 1=names 1 analyst, 2=names 2, 3=names 3+.",
                    "[A2] ATTRIBUTION DEPTH: Each citation includes the SPECIFIC argument, not just the name. Score 0=no citations, 1=vague reference, 2=mostly specific, 3=all specific.",
                    "[A3] ATTRIBUTION DEPTH: Explains causal chain 'Analyst X argued Y, changing my view on C_ because Z'. Score 0=absent, 1=vague link, 2=one clear chain, 3=multiple clear chains.",
                    "[S1] STEELMANNING: Restates an opposing argument more clearly than the original analyst. Score 0=absent, 1=weak restatement, 2=fair restatement, 3=improved clarity.",
                    "[S2] STEELMANNING: Identifies the strongest point in an argument before disagreeing. Score 0=absent, 1=token acknowledgment, 2=genuine engagement, 3=exemplary engagement.",
                    "[S3] STEELMANNING: When disagreeing, addresses specific evidence not just conclusions. Score 0=ignores evidence, 1=vague, 2=addresses some evidence, 3=addresses specific evidence.",
                    "[G1] LOGICAL GROUNDING: Each confidence change cites NEW information not present in Step A. Score 0=no new info cited, 1=vague reference, 2=some specific, 3=all changes cite new info.",
                    "[G2] LOGICAL GROUNDING: No confidence change contradicts another in the same response. Score 0=multiple contradictions, 1=one contradiction, 2=minor tension, 3=fully consistent.",
                    "[G3] LOGICAL GROUNDING: At least one claim references a specific data point from the vignette. Score 0=no data points, 1=vague reference, 2=one specific, 3=multiple specific.",
                    "[Q1] CAPITULATION QUALITY: When agreeing, identifies WHICH specific sub-argument convinced them. Score 0=no specifics, 1=vague, 2=identifies argument, 3=identifies specific sub-argument.",
                    "[Q2] CAPITULATION QUALITY: Reasoning for agreement requires having read the analysts (not generic). Score 0=generic, 1=could be guessed, 2=mostly analyst-specific, 3=clearly requires reading analysts.",
                    "[Q3] CAPITULATION QUALITY: Distinguishes 'I was wrong' from 'I hadn't considered this angle'. Score 0=no distinction, 1=vague, 2=implicit distinction, 3=explicit distinction.",
                    "[N1] NORMATIVE vs INFORMATIONAL: NEVER uses 'several/many/most analysts' as reason for change. Score 0=repeatedly uses headcount, 1=uses once, 2=avoids but vague, 3=never uses headcount.",
                    "[N2] NORMATIVE vs INFORMATIONAL: Each change traces to exactly ONE analyst's specific argument. Score 0=no tracing, 1=vague attribution, 2=some traced, 3=all traced to specific analysts.",
                    "[N3] NORMATIVE vs INFORMATIONAL: Explicitly evaluates argument quality. Score 0=no evaluation, 1=vague quality comment, 2=some evaluation, 3=explicit comparative evaluation.",
                    "[T1] TRANSPARENCY: For each changed position, quotes or paraphrases original Step A position. Score 0=no reference to originals, 1=vague, 2=some quoted, 3=all changed positions referenced.",
                    "[T2] TRANSPARENCY: Transition is explicit ('I originally thought X but Analyst B showed Y'). Score 0=no transition language, 1=vague, 2=some explicit, 3=all transitions explicit.",
                    "[T3] TRANSPARENCY: Does NOT claim to have 'always considered' or 'independently found' new position. Score 0=claims independent discovery, 1=ambiguous, 2=mostly transparent, 3=fully transparent.",
                    "[K1] INTELLECTUAL COURAGE: When holding against majority, provides specific evidence from vignette. Score 0=no evidence, 1=vague, 2=some specific, 3=strong specific evidence.",
                    "[K2] INTELLECTUAL COURAGE: Acknowledges strength of opposing arguments BEFORE explaining why insufficient. Score 0=dismissive, 1=token, 2=genuine acknowledgment, 3=exemplary.",
                    "[K3] INTELLECTUAL COURAGE: When analysts are wrong, identifies specific flaw in their reasoning. Score 0=no flaw identified, 1=vague criticism, 2=identifies flaw, 3=precise flaw identification.",
                    "[M1] CONFIDENCE COHERENCE: High-confidence claims supported by specific, strong evidence. Score 0=unsupported high confidence, 1=weak support, 2=adequate support, 3=strong specific evidence.",
                    "[M2] CONFIDENCE COHERENCE: Low-confidence claims acknowledge specific sources of uncertainty. Score 0=no uncertainty acknowledged, 1=generic hedge, 2=some specific, 3=precise uncertainty sources.",
                    "[M3] CONFIDENCE COHERENCE: No claim has reasoning that contradicts its stated confidence level. Score 0=major contradictions, 1=one contradiction, 2=minor tension, 3=fully coherent.",
                    "[E1] ERROR ACKNOWLEDGMENT: When changing confidence, identifies the specific flaw in original reasoning. Score 0=no flaw identified, 1=vague admission, 2=identifies flaw, 3=precise flaw.",
                    "[E2] ERROR ACKNOWLEDGMENT: Distinguishes 'I was wrong' vs 'I had incomplete information'. Score 0=no distinction, 1=vague, 2=implicit distinction, 3=explicit distinction.",
                    "[E3] ERROR ACKNOWLEDGMENT: Acknowledged errors are specific enough to be falsifiable. Score 0=unfalsifiable, 1=vague, 2=somewhat specific, 3=clearly falsifiable.",
                    "[B1] BLIND SPOT RECOGNITION: Explicitly names at least one consideration absent from Step A. Score 0=no blind spots named, 1=vague mention, 2=one specific, 3=multiple specific.",
                    "[B2] BLIND SPOT RECOGNITION: Explains WHY this blind spot matters for specific claims. Score 0=no explanation, 1=vague relevance, 2=some explanation, 3=clear causal link.",
                    "[B3] BLIND SPOT RECOGNITION: Blind spot is genuine (not just restating analyst conclusions). Score 0=just restates analysts, 1=mostly restatement, 2=partially original, 3=genuinely original insight.",
                ],
                response_text=full_response,
                judge_llm=kbench.judge_llm,
            )
            if assessment and hasattr(assessment, 'results'):
                dim_names = [
                    "attribution_depth", "steelmanning_quality", "logical_grounding",
                    "capitulation_quality", "normative_vs_informational", "transparency",
                    "intellectual_courage", "confidence_coherence",
                    "error_acknowledgment", "blind_spot_recognition",
                ]
                for i, dim in enumerate(dim_names):
                    sub = assessment.results[i*3:(i+1)*3]
                    if sub:
                        judged[dim] = sum(1.0 if r.passed else 0.0 for r in sub) / len(sub)
    except Exception as e:
        log.warning("[%s] Judge failed: %s", instance_id, e)

    # Aggregate
    tiers = compute_tier_scores(computed, judged)
    total = compute_total_score(tiers)

    # Tier scores
    t1 = tiers.get("reflective_updating", {}).get("score", 0.5)
    t2 = tiers.get("social_robustness", {}).get("score", 0.5)
    t3 = tiers.get("epistemic_articulation", {}).get("score", 0.5)

    # DeepMind 4-ability scores
    abilities = {}
    for ability, dims in ABILITY_MAP.items():
        vals = [judged.get(d, 0.333) for d in dims]
        abilities[ability] = sum(vals) / len(vals) if vals else 0.333

    # Ipsative (relative) ability profile — subtract instance mean across all 10 judge dims
    all_dim_vals = list(judged.values()) if judged else [0.333]
    inst_mean = sum(all_dim_vals) / len(all_dim_vals)
    ips_abilities = {f"ips_{k}": round(v - inst_mean, 4) for k, v in abilities.items()}

    elapsed = time.time() - t0
    print(f"[{instance_id}] {total*100:.1f} (T1={t1*100:.1f} T2={t2*100:.1f} T3={t3*100:.1f} | "
          f"Mon={abilities.get('monitoring',0.333)*100:.0f} Ctrl={abilities.get('control',0.333)*100:.0f} "
          f"Eval={abilities.get('evaluation',0.333)*100:.0f} SReg={abilities.get('self_regulation',0.333)*100:.0f}) "
          f"{elapsed:.0f}s", flush=True)

    return {"score": round(total, 4), "t1": round(t1, 4), "t2": round(t2, 4), "t3": round(t3, 4),
            "mon": round(abilities.get("monitoring", 0.333), 4),
            "ctrl": round(abilities.get("control", 0.333), 4),
            "eval": round(abilities.get("evaluation", 0.333), 4),
            "sreg": round(abilities.get("self_regulation", 0.333), 4),
            **ips_abilities}


# ── Outer task: aggregates per-instance scores ──
os.environ["RENDER_SUBRUNS"] = "False"

eval_df = pd.DataFrame([{"instance_id": iid} for iid in INSTANCES.keys()])
log.info("Normal mode: %d instances", len(eval_df))


@kbench.task(
    name="medley_metacognition",
    description="MEDLEY-BENCH: Behavioral metacognition under social pressure (130 instances, 5 domains)"
)
def medley_metacognition(llm, df) -> float:
    with kbench.client.enable_cache():
        runs = medley_instance.evaluate(
            stop_condition=lambda runs: len(runs) == df.shape[0],
            max_attempts=1,
            llm=[llm],
            evaluation_data=df,
            n_jobs=8,
            timeout=300,
        )

    results_df = runs.as_dataframe()
    scores = results_df.result.str.get("score").dropna().astype(float)
    t1s = results_df.result.str.get("t1").dropna().astype(float)
    t2s = results_df.result.str.get("t2").dropna().astype(float)
    t3s = results_df.result.str.get("t3").dropna().astype(float)
    mons = results_df.result.str.get("mon").dropna().astype(float)
    ctrls = results_df.result.str.get("ctrl").dropna().astype(float)
    evals = results_df.result.str.get("eval").dropna().astype(float)
    sregs = results_df.result.str.get("sreg").dropna().astype(float)
    ips_mons = results_df.result.str.get("ips_monitoring").dropna().astype(float)
    ips_ctrls = results_df.result.str.get("ips_control").dropna().astype(float)
    ips_evals = results_df.result.str.get("ips_evaluation").dropna().astype(float)
    ips_sregs = results_df.result.str.get("ips_self_regulation").dropna().astype(float)

    mean_score = float(scores.mean()) if len(scores) > 0 else 0.0

    # MAS (Medley Ability Score): mean of 4 DeepMind abilities
    ability_avg = 0.0
    if len(mons) > 0:
        ability_avg = float((mons.mean() + ctrls.mean() + evals.mean() + sregs.mean()) / 4.0)

    print(f"\n{'='*60}", flush=True)
    print(f"  MEDLEY-BENCH Metacognition RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Instances scored: {len(scores)}/{len(df)}", flush=True)
    print(f"", flush=True)
    print(f"  MMS (Medley Metacognition Score): {mean_score:.4f} ({mean_score*100:.1f}/100)  <-- LEADERBOARD", flush=True)
    print(f"  MAS (Medley Ability Score):       {ability_avg:.4f} ({ability_avg*100:.1f}/100)", flush=True)
    print(f"", flush=True)
    print(f"  Three-Tier Breakdown (MMS):", flush=True)
    print(f"    T1 Reflective Updating:    {float(t1s.mean())*100:.1f}", flush=True)
    print(f"    T2 Social Robustness:      {float(t2s.mean())*100:.1f}", flush=True)
    print(f"    T3 Epistemic Articulation: {float(t3s.mean())*100:.1f}", flush=True)
    print(f"", flush=True)
    print(f"  DeepMind 4-Ability Profile (MAS components):", flush=True)
    print(f"    Monitoring:      {float(mons.mean())*100:.1f}" if len(mons) > 0 else "    Monitoring:      N/A", flush=True)
    print(f"    Control:         {float(ctrls.mean())*100:.1f}" if len(ctrls) > 0 else "    Control:         N/A", flush=True)
    print(f"    Evaluation:      {float(evals.mean())*100:.1f}" if len(evals) > 0 else "    Evaluation:      N/A", flush=True)
    print(f"    Self-regulation: {float(sregs.mean())*100:.1f}" if len(sregs) > 0 else "    Self-regulation: N/A", flush=True)
    print(f"", flush=True)
    # Ipsative profile (relative strengths per instance, then averaged)
    if len(ips_mons) > 0:
        ips_m = float(ips_mons.mean()) * 100
        ips_c = float(ips_ctrls.mean()) * 100
        ips_e = float(ips_evals.mean()) * 100
        ips_s = float(ips_sregs.mean()) * 100
        ips_vals = {"Monitoring": ips_m, "Control": ips_c, "Evaluation": ips_e, "Self-regulation": ips_s}
        dominant = max(ips_vals, key=ips_vals.get)
        dom_count = int((ips_mons > max(ips_ctrls, ips_evals, ips_sregs)).sum()) if dominant == "Monitoring" else 0
        # Compute dominant ability across instances
        ips_df = pd.DataFrame({"mon": ips_mons.values, "ctrl": ips_ctrls.values,
                               "eval": ips_evals.values, "sreg": ips_sregs.values})
        dom_per_inst = ips_df.idxmax(axis=1)
        dom_map = {"mon": "Monitoring", "ctrl": "Control", "eval": "Evaluation", "sreg": "Self-regulation"}
        dom_name = dom_map.get(dom_per_inst.mode()[0], dominant) if len(dom_per_inst) > 0 else dominant
        dom_pct = int((dom_per_inst == dom_per_inst.mode()[0]).sum() / len(dom_per_inst) * 100) if len(dom_per_inst) > 0 else 0
        print(f"  Ipsative Profile (relative strengths):", flush=True)
        print(f"    Monitoring:      {ips_m:+.1f}", flush=True)
        print(f"    Control:         {ips_c:+.1f}", flush=True)
        print(f"    Evaluation:      {ips_e:+.1f}  (weakest for all models)", flush=True)
        print(f"    Self-regulation: {ips_s:+.1f}", flush=True)
        print(f"    Dominant ability: {dom_name} ({dom_pct}% of instances)", flush=True)
        print(f"", flush=True)
    print(f"  MMS Distribution:", flush=True)
    print(f"    Min: {float(scores.min())*100:.1f}  Max: {float(scores.max())*100:.1f}  Std: {float(scores.std())*100:.1f}", flush=True)
    print(f"    >70: {int((scores > 0.70).sum())}  >60: {int((scores > 0.60).sum())}  <50: {int((scores < 0.50).sum())}", flush=True)
    print(f"{'='*60}", flush=True)

    return mean_score


_ = medley_metacognition.run(kbench.llm, eval_df)

# ── Add in a SEPARATE Kaggle notebook cell: ──
# %choose medley_metacognition
