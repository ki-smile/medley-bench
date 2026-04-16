# MEDLEY-BENCH

**Package: v0.5.2 (beta)** · **Dataset: v1.0**

**Behavioral Metacognition Under Social Pressure**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/medley-bench.svg)](https://pypi.org/project/medley-bench/)
[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-20beff.svg)](https://www.kaggle.com/datasets/farhadabtahi/medley-bench-data)
[![Models](https://img.shields.io/badge/Models_Evaluated-35-green.svg)](https://github.com/ki-smile/medley-bench/tree/main/results/metacognition/v1.0/normal/)

MEDLEY-BENCH measures **behavioural metacognition** in large language models -- the capacity to monitor, evaluate, and control one's own reasoning under escalating social-epistemic pressure. Unlike accuracy-focused benchmarks, MEDLEY-BENCH measures *how models behave when challenged*, not whether they know the answer.

> ⚠️ **Beta release.** The `medley-bench` package is published as **v0.5.0 (beta)**: APIs, prompts, and scoring weights may change before the stable 1.0 line. The **dataset is frozen at v1.0** and is reproducible as released.
>
> ⏱️ **Expect long runs.** A single model on the full 130-instance dataset issues **several hundred API calls** (3 target calls/instance × 130 = 390, plus 130 judge calls = 520 total). Wall-clock time depends entirely on provider latency: **~1 hour on fast hosted APIs** (Gemini Flash, Claude Haiku, GPT-4.1-mini, or Ollama cloud), **several hours on slower ones**, and **many hours on local Ollama with mid-size open-weight models** (Step B-Social alone runs 2–3 min/instance on a 4B-class local model). Plan accordingly — the runner saves results incrementally and is resumable.

---

## Installation

```bash
pip install medley-bench
```

## Supported Providers

| Model ID pattern | Provider | Example |
|---|---|---|
| `claude-*` | Anthropic (direct) | `claude-haiku-4.5` |
| `gpt-*`, `o1-*`, `o3-*` | OpenAI (direct) | `gpt-4.1`, `gpt-5.4-mini` |
| `gemini-*` | Google (direct) | `gemini-2.5-flash` |
| `ollama/model` | Ollama (local or cloud) | `ollama/gemma3:12b`, `ollama/gpt-oss:20b-cloud` |
| `org/model` | OpenRouter | `anthropic/claude-haiku-4.5` |

Set the corresponding API key as an environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`). Ollama requires no key for local models.

## Quick Start

### Run benchmark on a model

The 130-instance dataset is bundled with the package — no separate download needed.

```bash
# Cloud model via OpenRouter (one API key for all providers)
export OPENROUTER_API_KEY="sk-or-..."
medley-bench benchmark --models "anthropic/claude-haiku-4.5"

# Local Ollama model
medley-bench benchmark --models "ollama/gemma3:12b"
```

### Run benchmark with a live judge

By default, the benchmark scores only the deterministic measures (T1 + most of T2). To also score the judge-dependent measures (T3), pass a judge model:

```bash
# Recommended judge: Gemini 2.5 Flash (fast, cheap, excellent structured output)
export GOOGLE_API_KEY="AI..."
medley-bench benchmark \
  --models "ollama/gemma3:12b" \
  --judge-model gemini-2.5-flash \
  --judge-api-key $GOOGLE_API_KEY

# Fully offline: use an Ollama cloud model as judge
medley-bench benchmark \
  --models "ollama/gemma3:12b" \
  --judge-model gemma4:31b-cloud \
  --judge-base-url http://localhost:11434/v1

# Smoke test: limit to first N instances per domain
medley-bench benchmark \
  --models "ollama/gemma3:12b" \
  --judge-model gemini-2.5-flash \
  --n-instances 3
```

Any OpenAI-compatible endpoint works as a judge. Reasoning models (gpt-oss, glm-4.6, DeepSeek v3.1, etc.) are supported transparently.

### View leaderboard

```bash
medley-bench leaderboard --results results/
```

### Run in Google Colab

Open the [example notebook](https://github.com/ki-smile/medley-bench/blob/main/examples/medley_bench_colab.ipynb) in Google Colab to benchmark any model — cloud APIs (OpenRouter, Anthropic, OpenAI, Google) or local HuggingFace models via vLLM on a free T4 GPU. No local setup needed.

### More help

```bash
medley-bench --help          # Quick start + provider table
medley-bench about           # Project info, scoring, links, citation
medley-bench examples        # 7 numbered usage recipes
medley-bench benchmark --help  # All CLI options with examples
```

### Note on lm-eval-harness

MEDLEY-BENCH cannot run as a native [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) task because the three-step protocol requires sequential, state-dependent API calls (Step B's prompt depends on Step A's output). This is a common limitation for multi-turn behavioural benchmarks — AlpacaEval, MT-Bench, and Arena-Hard use the same approach.

---

## Three-Step Decomposition

Every benchmark instance runs three model calls in isolated contexts:

| Step | What the model sees | What it isolates |
|------|-------------------|-----------------|
| **Step A** (Solo) | Problem vignette only | Independent analysis + confidence calibration |
| **Step B-Private** | Own Step A + self-review nudge | Self-revision capacity |
| **Step B-Social** | 8 analyst opinions + consensus | Social updating quality |

```
Delta(A -> B-Private)  = self-revision
Delta(B-Private -> B-Social) = pure social influence
```

## Scoring Framework

### Scores

| Score | What it measures | Composition |
|-------|-----------------|-------------|
| **MMS** (Medley Metacognition Score) | Articulation quality | T1 Reflective Updating + T2 Social Robustness + T3 Epistemic Articulation (equal weights) |
| **MAS** (Medley Ability Score) | Behavioural competence | Mean of Monitoring, Control, Evaluation, Self-regulation |

### Three-Tier Aggregation

| Tier | Weight | Measures | Method |
|------|--------|----------|--------|
| **T1: Reflective Updating** | 33% | Proportionality, selectivity, volatility, uncertainty localisation, Brier change | Deterministic |
| **T2: Social Robustness** | 33% | Private-vs-social delta, epistemic cowardice, resistance appropriateness, majority pressure, capitulation quality, normative/informational | Mixed |
| **T3: Epistemic Articulation** | 33% | Content engagement, steelmanning, argument specificity, synthesis necessity, attribution depth, intellectual courage, error acknowledgement + 6 more | Mixed |

75% of scoring weight is deterministic (rule-based behavioural deltas). 25% uses an LLM judge with anti-rhetoric rubric.

### Anti-Gaming Controls

- Consensus masking (directional labels, not raw numbers)
- Anonymised analysts in prompts
- 30 known-answer instances with verified-wrong claims
- Per-claim ground truth from consensus verification
- Circularity-aware judge rotation (no model judged by own family)

## Dataset

**130 instances** across 5 domains:

| Domain | Instances | Reasoning type |
|--------|-----------|---------------|
| Medical Diagnosis | 27 | Evidential -- contradictory clinical evidence |
| System Troubleshooting | 26 | Causal -- root cause through layers |
| Code Review | 27 | Contextual -- severity depends on threat model |
| Architecture Design | 25 | Tradeoff -- no single right answer |
| Statistical Reasoning | 25 | Formal -- same data, different frameworks |

Each instance includes a vignette, 5 claims with disagreement scores, 8 analyst responses (from 28-model pool), jackknife consensus, and per-claim verified-wrong labels.

The dataset is also available on Kaggle: [farhadabtahi/medley-bench-data](https://www.kaggle.com/datasets/farhadabtahi/medley-bench-data)

## Benchmark Modes

### Normal Mode (Kaggle-compatible)
3 calls per instance x 130 = 390 API calls. Standard three-step protocol.

### Progressive Mode (5-stage stress test)

| Stage | Analysts | Instances | Purpose |
|-------|----------|-----------|---------|
| Baseline | 0 | 130 | Solo calibration |
| Mild | 2 | 130 | Basic social responsiveness |
| Moderate | 4 | 130 | Proportional updating |
| Strong | 6 | 50 | Argument discrimination |
| Adversarial | 6 (wrong consensus) | 30 | Intellectual courage under max pressure |

## Kaggle vs Local Scoring

The Kaggle competition framework (`kbench`) imposes limitations on judge scoring compared to the local benchmark:

| Feature | Local Benchmark | Kaggle (`kbench`) |
|---------|----------------|-------------------|
| Judge scale | Graded 0-3 per sub-criterion | Binary pass/fail |
| Family exclusion | No model judged by own family | Not available |
| T3 resolution | Fine-grained (30 sub-criteria x 4 levels) | Compressed (30 x 2 levels) |
| Score offset | Reference | +2-4 pts higher (compressed T3) |
| Rank correlation | Reference | rho > 0.97 (rankings preserved) |

**Why rankings are preserved:** 75% of MMS comes from deterministic rule-based measures (T1 + T2) that are identical on both platforms. The judge limitations only affect T3 (25% of score).

**Recommendation:** Use the local benchmark (`pip install medley-bench`) for research. Use Kaggle notebooks for competition submission and quick model comparison.

## Results: 35 Models

| Rank | Model | MMS | MAS | T1 | T2 | T3 |
|------|-------|-----|-----|----|----|-----|
| 1 | Claude Haiku 4.5 | 62.2 | 61.8 | 61.1 | 56.3 | 69.2 |
| 2 | Gemma 3 27B | 61.1 | 62.0 | 60.1 | 55.8 | 67.5 |
| 3 | Qwen 3.5 397B | 61.0 | 59.2 | 59.8 | 56.5 | 66.7 |
| 4 | Gemini 3 Flash | 60.7 | 60.2 | 59.5 | 56.0 | 66.5 |
| 5 | Claude Sonnet 4.5 | 60.4 | 60.7 | 59.3 | 55.7 | 66.3 |
| 6 | Gemma 3 12B | 60.1 | 61.5 | 58.9 | 55.5 | 65.9 |

Full results for all 35 models are available in the [GitHub repository](https://github.com/ki-smile/medley-bench/tree/main/results/metacognition/v1.0/normal/).

### Key Findings

1. **Scale buys Evaluation, not Control.** Evaluation ability scales with model size, but Control (social robustness) shows no scaling -- GPT-4.1-Nano achieves the best T2 score.

2. **Argument-evaluators vs. statistics-followers.** Two behavioural profiles invisible to standard benchmarks, predicted by a single judge dimension (Normative/Informational, rho = -0.82).

3. **Universal evaluation deficit.** Under ipsative scoring, Evaluation is every model's weakest relative ability.

4. **Non-monotonic scale returns.** Gemma family: 4B(30) -> 9B(50) -> 12B(60) -> 27B(61) -> Gen4-31B(57).

## Citation

```bibtex
@article{abtahi2026medleybench,
  title={MEDLEY-BENCH: Scale Buys Evaluation but Not Control in AI Metacognition},
  author={Abtahi, Farhad and Karbalaie, Abdolamir and Illueca-Fernandez, Eduardo and Seoane, Fernando},
  year={2026},
  note={Preprint}
}
```

## License

Apache License 2.0
