"""Benchmark a local Ollama model with an external judge.

Runs the three-step metacognition pipeline on a target model served by local
Ollama, then calls the v2 judge on each instance using any OpenAI-compatible
endpoint (Gemini, Anthropic via OpenRouter, or an Ollama cloud model).

Usage
-----
Env vars / defaults:
    TARGET_MODEL      ollama/MedAIBase/MedGemma1.5:4b   — the model under test
    JUDGE_MODEL       gemini-2.5-flash                  — judge model id
    JUDGE_BASE_URL    https://generativelanguage.googleapis.com/v1beta/openai
    JUDGE_API_KEY     (from GOOGLE_API_KEY)
    DATA_DIR          data/metacognition/v1.0
    DOMAIN            medical
    N_INSTANCES       5                                 — limit for smoke runs
    OUTPUT            results/local_with_cloud_judge.json

Examples
--------
# 1. Local MedGemma, judged by Gemini Flash (recommended — accurate, cheap, fast)
GOOGLE_API_KEY=... python examples/ollama_with_cloud_judge.py

# 2. Local MedGemma, judged by an Ollama cloud model (fully offline-capable)
JUDGE_MODEL=qwen3-coder:480b-cloud \
JUDGE_BASE_URL=http://localhost:11434/v1 \
JUDGE_API_KEY=ollama \
python examples/ollama_with_cloud_judge.py

# 3. Reasoning judge (e.g. gpt-oss, glm-4.6) — the library now handles the
#    `reasoning` field automatically, so these models "just work":
JUDGE_MODEL=gpt-oss:20b-cloud \
JUDGE_BASE_URL=http://localhost:11434/v1 \
JUDGE_API_KEY=ollama \
python examples/ollama_with_cloud_judge.py
"""
import asyncio
import json
import os
import time
from pathlib import Path

from openai import OpenAI

from src.core.providers import get_provider
from src.core.parsing import parse_json_response
from src.tracks.metacognition.prompts.step_a import build_prompt as build_step_a
from src.tracks.metacognition.prompts.step_b_private import build_prompt as build_step_b_private
from src.tracks.metacognition.prompts.step_b_social import build_prompt as build_step_b_social
from src.tracks.metacognition.scoring.measures import compute_all_computed_measures
from src.tracks.metacognition.scoring.aggregation import (
    compute_tier_scores, compute_total_score,
)
from src.tracks.metacognition.scoring.judge import call_judge_v2


TARGET_MODEL = os.environ.get("TARGET_MODEL", "ollama/MedAIBase/MedGemma1.5:4b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
JUDGE_BASE_URL = os.environ.get(
    "JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
DATA_DIR = Path(os.environ.get("DATA_DIR", "data/metacognition/v1.0"))
DOMAIN = os.environ.get("DOMAIN", "medical")
N_INSTANCES = int(os.environ.get("N_INSTANCES", "5"))
OUTPUT = Path(os.environ.get("OUTPUT", "results/local_with_cloud_judge.json"))


async def main() -> None:
    instances = json.load(open(DATA_DIR / "instances" / f"{DOMAIN}.json"))[:N_INSTANCES]
    consensus_map = json.load(open(DATA_DIR / "consensus" / f"{DOMAIN}.json"))

    provider = get_provider(TARGET_MODEL)
    judge_client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY or "unused")

    print(f"Target : {TARGET_MODEL}")
    print(f"Judge  : {JUDGE_MODEL}  @  {JUDGE_BASE_URL}")
    print(f"Data   : {DATA_DIR} / {DOMAIN}.json  (first {N_INSTANCES} instances)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for idx, inst in enumerate(instances, 1):
        iid = inst["instance_id"]
        cons = consensus_map.get(iid, {})
        print(f"\n[{idx}/{len(instances)}] {iid}")
        t0 = time.time()

        raw_a = await provider.complete(build_step_a(vignette=inst["vignette"]))
        step_a = parse_json_response(raw_a)

        raw_bp = await provider.complete(
            build_step_b_private(step_a_response=raw_a, vignette=inst["vignette"])
        )
        step_b_priv = parse_json_response(raw_bp)

        analyst_text = json.dumps(inst.get("ensemble_outputs", []), indent=2)
        consensus_text = json.dumps(cons, indent=2)
        raw_bs = await provider.complete(build_step_b_social(
            step_a_response=raw_a, vignette=inst["vignette"],
            analyst_outputs=analyst_text, jackknifed_consensus=consensus_text,
        ))
        step_b_soc = parse_json_response(raw_bs)

        computed = compute_all_computed_measures(
            step_a=step_a, step_b_private=step_b_priv, step_b_social=step_b_soc,
            instance=inst, consensus=cons,
            analyst_outputs=inst.get("ensemble_outputs"),
        )

        judged = call_judge_v2(
            raw_a=raw_a, raw_bs=raw_bs, vignette=inst["vignette"],
            ensemble_outputs=inst.get("ensemble_outputs", []),
            key_claims=inst.get("key_claims", []),
            client=judge_client, model=JUDGE_MODEL,
            is_known_answer=inst.get("is_known_answer", False),
        )

        tier_scores = compute_tier_scores(computed, judged)
        total = compute_total_score(tier_scores)

        result = {
            "instance_id": iid,
            "target_model": TARGET_MODEL,
            "judge_model": JUDGE_MODEL,
            "total_score": total,
            "tier_scores": {k: round(v["score"], 4) for k, v in tier_scores.items()},
            "computed": {k: round(v, 4) for k, v in computed.items() if isinstance(v, (int, float))},
            "judged_dims": {k: round(v, 3) for k, v in judged.items()},
            "elapsed_s": round(time.time() - t0, 1),
        }
        results.append(result)
        print(
            f"  total={result['total_score']:.3f}  "
            f"T1={result['tier_scores']['reflective_updating']:.3f}  "
            f"T2={result['tier_scores']['social_robustness']:.3f}  "
            f"T3={result['tier_scores']['epistemic_articulation']:.3f}  "
            f"({result['elapsed_s']:.0f}s)"
        )

        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)

    if results:
        avg = sum(r["total_score"] for r in results) / len(results)
        print(f"\nMean total across {len(results)} instances: {avg:.4f}")
        print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
