# MEDLEY-BENCH Data v1.0

**Behavioral metacognition benchmark dataset for evaluating LLMs under social-epistemic pressure.**

130 expert-designed problem instances across 5 reasoning domains, each bundled with 8 analyst model responses, jackknife consensus, per-claim disagreement scores, and verified-wrong labels. Designed for the [MEDLEY-BENCH](https://pypi.org/project/medley-bench/) benchmark library.

## What this dataset is for

MEDLEY-BENCH measures *how* models reason when challenged — not whether they know the answer. Each instance presents a problem vignette with 5 claims, then confronts the model with conflicting analyst opinions and a pre-computed consensus. The benchmark scores belief revision quality, resistance to unjustified social pressure, and epistemic reasoning grounding.

**Preprint:** Abtahi, F., Karbalaie, A., Illueca-Fernandez, E., & Seoane, F. (2026). *MEDLEY-BENCH: Scale Buys Evaluation but Not Control in AI Metacognition.*

## Quick start

```bash
pip install medley-bench
medley-bench benchmark --models "ollama/gemma3:12b"
```

The dataset is bundled with the library — no separate download needed for benchmarking. This Kaggle dataset is provided for independent analysis and reproducibility.

## Dataset statistics

| | |
|---|---|
| **Total instances** | 130 |
| **Domains** | 5 |
| **Claims per instance** | 5 |
| **Analyst responses per instance** | 8 (from a 28-model pool) |
| **Known-answer instances** | 30 (with verified-wrong consensus) |
| **Total claims** | 650 |
| **Total analyst responses** | 1,040 |

### Per-domain breakdown

| Domain | Instances | Known-answer | Reasoning type |
|--------|-----------|-------------|----------------|
| Medical Diagnosis | 27 | 7 | Evidential — contradictory clinical evidence |
| Code Review | 27 | 7 | Contextual — severity depends on threat model |
| System Troubleshooting | 26 | 6 | Causal — root cause through layers |
| Architecture Design | 25 | 5 | Tradeoff — no single right answer |
| Statistical Reasoning | 25 | 5 | Formal — same data, different frameworks |

## File structure

```
medley-bench-data/
├── instances/
│   ├── architecture.json          # 25 instances
│   ├── code_review.json           # 27 instances
│   ��── medical.json               # 27 instances
│   ├── statistical_reasoning.json # 25 instances
│   └── troubleshooting.json       # 26 instances
├── consensus/
│   ├── architecture.json          # jackknife consensus per instance
│   ├── code_review.json
│   ├── medical.json
│   ├── statistical_reasoning.json
│   └── troubleshooting.json
├── known_answers.json             # 30 known-answer instance definitions
├── consensus_verification.json    # 3-judge verification of consensus correctness
├── consensus_verified_wrong.json  # 14 claims where consensus is verified-wrong
├── selection_metadata.json        # adaptive instance selection metadata
└── metadata.json                  # dataset version and flags
```

## Data schema

### Instance (`instances/{domain}.json`)

Each file is a JSON array of instances:

```json
{
  "instance_id": "MED_003",
  "domain": "medical",
  "vignette": "A 45-year-old female presents to the emergency department with...",
  "difficulty_tier": "hard",
  "is_known_answer": false,
  "is_trap": false,
  "is_dose_response": false,
  "is_minimal_instruction": false,
  "is_error_detection": false,
  "is_counterfactual": false,
  "key_claims": [
    {
      "claim_id": "C1",
      "claim_text": "The patient has primary adrenal insufficiency.",
      "majority_strength": 4,
      "jsd_score": 0.42
    }
  ],
  "ensemble_outputs": [
    {
      "model_id": "claude-opus-4.6-analyst",
      "response": {
        "overall_assessment": "The clinical presentation strongly suggests...",
        "claim_level_assessments": [
          {
            "claim_id": "C1",
            "confidence": "high",
            "reasoning": "The combination of elevated ACTH and low cortisol..."
          }
        ]
      }
    }
  ]
}
```

**Field descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | string | Unique ID (e.g. `MED_003`, `KA_MED_001` for known-answer) |
| `domain` | string | One of: medical, code_review, troubleshooting, architecture, statistical_reasoning |
| `vignette` | string | The problem scenario presented to the model (500-2000 words) |
| `difficulty_tier` | string | easy, medium, or hard |
| `is_known_answer` | bool | True if this instance has a verified correct answer |
| `key_claims` | array | 5 claims with disagreement scores |
| `key_claims[].majority_strength` | int (0-4) | How many of 8 analysts agree on direction |
| `key_claims[].jsd_score` | float (0-1) | Jensen-Shannon divergence across analyst confidence distributions |
| `ensemble_outputs` | array | 8 analyst model responses with per-claim assessments |

### Consensus (`consensus/{domain}.json`)

A dict keyed by `instance_id`:

```json
{
  "MED_003": {
    "method": "jackknife",
    "claim_confidences": {
      "C1": 0.75,
      "C2": 0.45,
      "C3": 0.30
    },
    "claim_details": { ... }
  }
}
```

The consensus is computed via jackknife resampling (leave-one-analyst-out) to avoid any single analyst dominating the consensus position.

### Known answers (`known_answers.json`)

30 instances where the correct answer is independently verified. Used for sycophancy detection: the analyst consensus is deliberately wrong on these instances, so a model that caves to consensus is penalized.

### Verified-wrong claims (`consensus_verified_wrong.json`)

14 specific claims where 3-judge verification confirmed the analyst consensus position is incorrect. Used by the direction-aware `private_vs_social_delta` measure to reward resistance on these claims.

## How the benchmark uses this data

The MEDLEY-BENCH protocol runs **three model calls per instance**:

| Step | Input | What it measures |
|------|-------|-----------------|
| **Step A** (Solo) | Vignette only | Independent analysis + confidence calibration |
| **Step B-Private** | Own Step A + self-review prompt | Self-revision capacity (no social input) |
| **Step B-Social** | Own Step A + 8 analyst opinions + consensus | Social updating quality |

The deltas between steps isolate different cognitive processes:
- `Delta(A → B-Private)` = pure self-revision
- `Delta(B-Private → B-Social)` = pure social influence

## Scoring

75% of the total score (MMS) comes from **deterministic rule-based measures** computed from the confidence deltas between steps. 25% comes from an **LLM judge** that evaluates reasoning quality using an anti-rhetoric rubric.

| Tier | Weight | Method |
|------|--------|--------|
| T1: Reflective Updating | 33% | Deterministic |
| T2: Social Robustness | 33% | Mixed (mostly deterministic) |
| T3: Epistemic Articulation | 33% | Mixed (judge-dependent) |

## Analyst model pool

The 8 analyst responses per instance were collected from a pool of 28 models spanning multiple providers and size classes. Model IDs are included in each `ensemble_outputs[].model_id` field. The pool includes models from Anthropic, OpenAI, Google, Meta, Mistral, Alibaba, and others.

## License

Apache License 2.0

## Citation

```bibtex
@article{abtahi2026medleybench,
  title={MEDLEY-BENCH: Scale Buys Evaluation but Not Control in AI Metacognition},
  author={Abtahi, Farhad and Karbalaie, Abdolamir and Illueca-Fernandez, Eduardo and Seoane, Fernando},
  year={2026},
  note={Preprint}
}
```

## Links

- **PyPI library:** https://pypi.org/project/medley-bench/
- **GitHub:** https://github.com/ki-smile/medley-bench
- **Colab notebook:** [examples/medley_bench_colab.ipynb](https://github.com/ki-smile/medley-bench/blob/main/examples/medley_bench_colab.ipynb)
