# Reproducing MEDLEY-BENCH Results

This guide explains how to reproduce the results reported in the preprint, from verifying existing results to running the full benchmark on new models.

## Prerequisites

- Python 3.10+
- API keys for model providers you want to evaluate (see [Provider Setup](#provider-setup))

```bash
# Install from PyPI (includes the 130-instance dataset)
pip install medley-bench

# Or install from source for development
git clone https://github.com/ki-smile/medley-bench.git
cd medley-bench
pip install -e ".[dev]"
```

## 1. Verify Existing Results (No API keys needed)

The `results/` directory contains complete outputs for all 35 models. You can regenerate the leaderboard and verify all scoring computations without any API calls.

### Regenerate leaderboard

```bash
medley-bench leaderboard --results results/metacognition/v1.0/normal/
```

This reads the raw responses and recomputes all deterministic measures (T1 + T2 rule-based components) from scratch, then combines with stored judge scores to produce the final ranking.

### Verify scoring for a specific model

```python
import json
from src.benchmark.scoring.aggregation import compute_total_score
from src.benchmark.scoring.computed_measures import compute_all_measures

# Load a result file
with open("results/metacognition/v1.0/normal/anthropic_claude-haiku-4.5.json") as f:
    instances = json.load(f)

# Recompute measures from raw responses
for inst in instances:
    recomputed = compute_all_measures(
        step_a=inst["raw_responses"]["step_a"],
        step_b_private=inst["raw_responses"]["step_b_private"],
        step_b_social=inst["raw_responses"]["step_b_social"],
        instance_data=load_instance(inst["instance_id"])  # from data/
    )
    # Compare with stored measures
    for key in recomputed:
        assert abs(recomputed[key] - inst["computed"][key]) < 1e-6, f"Mismatch: {key}"
```

### Run the test suite

```bash
# All tests
pytest tests/

# Scoring validation (verifies weight constraints, score bounds, known patterns)
pytest tests/scoring_validation/ -v

# Unit tests for individual measures
pytest tests/unit/test_scoring/ -v
```

## 2. Run the Benchmark on a New Model

### Provider Setup

See [`docs/PROVIDERS.md`](PROVIDERS.md) for detailed setup instructions for each provider. Quick summary:

| Provider | Environment Variable | Model ID format | Example |
|----------|---------------------|-----------------|---------|
| **OpenRouter** (recommended) | `OPENROUTER_API_KEY` | `org/model` | `anthropic/claude-haiku-4.5` |
| Anthropic (direct) | `ANTHROPIC_API_KEY` | `claude-*` | `claude-haiku-4.5` |
| OpenAI (direct) | `OPENAI_API_KEY` | `gpt-*` | `gpt-4.1` |
| Google (direct) | `GOOGLE_API_KEY` | `gemini-*` | `gemini-2.5-flash` |
| Ollama (local) | `OLLAMA_BASE_URL` | `ollama/model` | `ollama/gemma3:12b` |

All 35 models in the v1.0 release were collected via OpenRouter with a single API key.

### Run normal mode (390 API calls)

```bash
export ANTHROPIC_API_KEY="your-key-here"

medley-bench benchmark \
    --models "anthropic/claude-haiku-4.5" \
    --output results/normal/
```

The dataset (130 instances) is bundled with the package — no `--data` flag needed. This executes the three-step protocol on all 130 instances and writes a result JSON file.

**Estimated cost:** $2-15 per model depending on provider pricing and model size.
**Estimated time:** 30-90 minutes depending on rate limits.

### Run progressive mode (600 API calls)

```bash
medley-bench benchmark \
    --models "anthropic/claude-haiku-4.5" \
    --mode progressive \
    --output results/progressive/
```

### Run on multiple models

```bash
medley-bench benchmark \
    --models "anthropic/claude-haiku-4.5,openai/gpt-4.1,google/gemini-3-flash-preview" \
    --output results/normal/
```

## 3. Reproduce the Full Pipeline (Instance Generation)

The benchmark dataset in `data/` was generated using the admin pipeline. To regenerate from scratch:

### Step 1: Initialise database

```bash
medley-bench init
```

### Step 2: Load seed cases

```bash
medley-bench load-seeds
```

Loads 100 seed cases (+ 30 known-answer cases) across 5 domains.

### Step 3: Expand seeds to full vignettes

```bash
medley-bench expand --model "anthropic/claude-sonnet-4.5"
```

Each seed is expanded into a detailed problem vignette (~500 words) by a designer model.

### Step 4: Collect analyst responses

```bash
# Collect from multiple analyst models
for model in \
    "google/gemma-3-12b-it" \
    "qwen/qwen3-32b" \
    "meta-llama/llama-4-scout" \
    "mistralai/mistral-small-3.1-24b-instruct"; do
    medley-bench collect --model "$model" --role analyst
done

# Check collection status
medley-bench collect-status
```

Each instance needs responses from at least 8 analyst models. The adaptive selection algorithm (`src/admin/generation/adaptive_selection.py`) chooses the 8 most diverse analysts per instance from the full pool.

### Step 5: Build consensus

```bash
medley-bench build-consensus
```

Computes jackknife consensus (leave-one-out robustness) across analyst responses.

### Step 6: Validate and export

```bash
medley-bench validate   # Check quality gates G1-G15
medley-bench export     # Export benchmark dataset
```

## 4. Understanding the Result Files

### Normal mode result structure

Each JSON file contains an array of 130 instances. Per instance:

| Field | Description |
|-------|-------------|
| `instance_id` | Unique identifier (e.g., MED_003, CR_012) |
| `model` | Model identifier used for evaluation |
| `domain` | One of: medical, troubleshooting, code_review, architecture, statistical |
| `total_score` | MMS (0-100 scale) |
| `tier_scores` | `{t1, t2, t3}` -- per-tier scores |
| `computed` | All deterministic measures (proportionality, selectivity, etc.) |
| `judged` | All LLM judge dimension scores (0-1 scale, 10 dimensions) |
| `raw_responses` | Full model outputs for steps A, B-Private, B-Social |

### Progressive mode result structure

Same fields plus:

| Field | Description |
|-------|-------------|
| `stage` | baseline, adversarial, or stripped |
| `stage_scores` | Per-stage tier scores |
| `delta` | Score change from baseline |

### Recomputing scores

All deterministic measures (T1 and most of T2) can be recomputed from `raw_responses` + `data/`. The judge scores in `judged` were computed by a 3-judge rotation (Claude Sonnet 4.5, GPT-4.1, Gemini 2.5 Pro) and are stored as the mean across judges.

To rerun judging (requires API keys for judge models):

```bash
medley-bench rejudge --results results/metacognition/v1.0/normal/anthropic_claude-haiku-4.5.json --output results/rejudged/
```

## 5. Kaggle Submission

The `kaggle_notebook/` directory contains notebooks formatted for the Kaggle competition:

- `medley_bench_normal.py` -- Returns MMS (0-1 scale)
- `medley_bench_abilities.py` -- Returns MAS (0-1 scale)

These use the Kaggle `@kbench.task` harness with a binary pass/fail judge (simpler than the local 3-judge graded rotation). This produces a consistent +2-4 pt offset from local scores but preserves rankings.

## 6. Key Design Decisions

Documented here for transparency:

| Decision | Rationale |
|----------|-----------|
| 33/33/33 tier weights | Robust across weight variations (rho >= 0.977 with alternatives) |
| 3-judge rotation | Circularity-aware: no model judged by its own family |
| Per-claim verified-wrong scoring | More precise than blanket known-answer flags |
| Adaptive analyst selection | +19.9% ensemble diversity vs. random selection |
| Jackknife consensus | Robust to single-analyst outliers |
| Binary Kaggle judge | Matches competition framework; local graded judge for research |

## Troubleshooting

### Rate limits
Most providers impose rate limits. The benchmark runner includes exponential backoff. For large-scale runs, use `--n-jobs 1` and expect 60-90 minutes per model.

### Incomplete runs
If a run is interrupted, rerunning the same command will skip already-completed instances (results are written incrementally).

### Local models via Ollama
```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull gemma3:12b

# Run benchmark
medley-bench benchmark --models "ollama/gemma3:12b"
```
