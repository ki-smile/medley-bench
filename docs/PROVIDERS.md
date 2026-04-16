# Provider Setup Guide

MEDLEY-BENCH supports five provider backends. The benchmark auto-detects the correct provider from the model ID string.

## Model ID Routing

| Model ID pattern | Provider | Example |
|-----------------|----------|---------|
| `claude-*` | Anthropic (direct) | `claude-haiku-4.5` |
| `gpt-*`, `o1-*`, `o3-*` | OpenAI (direct) | `gpt-4.1`, `gpt-5.4-mini` |
| `gemini-*` | Google (direct) | `gemini-2.5-flash` |
| `ollama/model` | Ollama (local) | `ollama/gemma3:12b` |
| `org/model` | OpenRouter (remote) | `anthropic/claude-haiku-4.5` |

The routing logic is in `src/core/providers.py:get_provider()`.

---

## 1. OpenRouter (Recommended for Multi-Model Runs)

OpenRouter provides a single API endpoint for 200+ models from all major providers. This is the simplest way to benchmark many models, and the method used to collect all 35 results in the v1.0 dataset.

### Setup

1. Create an account at [openrouter.ai](https://openrouter.ai)
2. Generate an API key at [openrouter.ai/keys](https://openrouter.ai/keys)
3. Add credit ($5-20 is sufficient for several model runs)

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### Usage

Use the `org/model` format:

```bash
# Anthropic models via OpenRouter
medley-bench benchmark --models "anthropic/claude-haiku-4.5"

# Google models via OpenRouter
medley-bench benchmark --models "google/gemini-2.5-flash"

# OpenAI models via OpenRouter
medley-bench benchmark --models "openai/gpt-4.1"

# Multiple models in one run
medley-bench benchmark \
    --models "anthropic/claude-haiku-4.5,google/gemma-3-27b-it,openai/gpt-4.1"
```

### Free models (for testing)

OpenRouter offers several free models for testing your setup:

```bash
medley-bench benchmark \
    --models "google/gemma-3-27b-it:free"
```

Available free models: `google/gemma-3-27b-it:free`, `meta-llama/llama-3.3-70b-instruct:free`, `mistralai/mistral-small-3.1-24b-instruct:free`

### Cost

A full 130-instance run costs approximately $2-15 per model depending on model pricing. Frontier models (GPT-5.4, Claude Sonnet) cost more than mid-range models (Gemma, Llama).

---

## 2. Anthropic (Direct API)

For direct access to Claude models without OpenRouter as intermediary.

### Setup

1. Get an API key at [console.anthropic.com](https://console.anthropic.com)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Usage

Use the model ID directly (no org/ prefix):

```bash
medley-bench benchmark --models "claude-haiku-4.5"
medley-bench benchmark --models "claude-sonnet-4-20250514"
```

### Extended thinking

Claude models support extended thinking (System 2 reasoning). The provider handles this automatically for models that support it. Thinking tokens are tracked separately in usage stats but not included in scored output.

### Notes

- Temperature is set to 0 by default
- Max tokens: 10,000 (configurable)
- Retry with exponential backoff on rate limits (429)

---

## 3. OpenAI (Direct API)

For direct access to GPT models.

### Setup

1. Get an API key at [platform.openai.com](https://platform.openai.com)

```bash
export OPENAI_API_KEY="sk-..."
```

### Usage

```bash
medley-bench benchmark --models "gpt-4.1"
medley-bench benchmark --models "gpt-4.1-mini"
medley-bench benchmark --models "gpt-5.4"
```

### Reasoning models (o-series)

The provider auto-detects o1/o3/o4 models and adjusts accordingly:
- No temperature parameter (not supported)
- Uses `reasoning_effort` parameter (default: "medium")
- Uses `max_completion_tokens` instead of `max_tokens`
- Reasoning tokens tracked separately

```bash
medley-bench benchmark --models "o3-mini"
```

---

## 4. Google (Direct API)

For direct access to Gemini and Gemma models.

### Setup

1. Get an API key at [aistudio.google.com](https://aistudio.google.com)

```bash
export GOOGLE_API_KEY="AI..."
```

### Usage

```bash
medley-bench benchmark --models "gemini-2.5-flash"
medley-bench benchmark --models "gemini-2.5-pro"
```

### Notes

- Uses the `google-generativeai` Python SDK
- Async calls are wrapped in `asyncio.to_thread` (the SDK is synchronous)
- Some Gemma models are not available via Google AI Studio; use OpenRouter or Ollama instead

---

## 5. Ollama (Local Models)

For running open-weight models locally. Supports both local Ollama and remote Open WebUI instances.

### Local setup

1. Install Ollama: [ollama.com](https://ollama.com)
2. Pull a model:

```bash
ollama pull gemma3:12b
ollama pull qwen3:32b
ollama pull llama4:scout
```

3. Start the server (if not already running):

```bash
ollama serve
```

### Usage

Use the `ollama/` prefix:

```bash
medley-bench benchmark --models "ollama/gemma3:12b"
medley-bench benchmark --models "ollama/qwen3:32b"
```

### Remote Ollama / Open WebUI

For models served on a remote machine (e.g., a GPU server):

```bash
export OLLAMA_BASE_URL="https://your-server.example.com"
export OLLAMA_API_KEY="your-api-key"  # if authentication is required

medley-bench benchmark --models "ollama/gemma3:12b"
```

The provider auto-detects local vs remote:
- **Local** (`http://localhost:*`): Uses Ollama native `/api/generate` endpoint
- **Remote** (`https://*`): Uses OpenAI-compatible `/api/chat/completions` endpoint

### Thinking/reasoning models

For models with thinking capabilities (Qwen3, DeepSeek-R1, QwQ), the provider automatically disables thinking mode by default to preserve output tokens for structured JSON. Thinking tokens are wasted on internal reasoning chains that are not scored.

```bash
# Thinking disabled by default (recommended)
medley-bench benchmark --models "ollama/qwen3:32b"
```

### Ollama cloud models (`*-cloud` / `*:cloud`)

Ollama can route tagged model names (e.g. `gpt-oss:20b-cloud`, `qwen3-coder:480b-cloud`, `glm-4.6:cloud`, `deepseek-v3.1:671b-cloud`) through its cloud backend. From the client's perspective these look like any other local Ollama model — they share the same daemon and OpenAI-compatible endpoint at `http://localhost:11434/v1`. List what's available with `ollama list`.

Cloud models are convenient as **judge** backends: they give you frontier-scale models without a separate API key, and they route through the same local endpoint your target model uses.

```bash
# Target: local 4B model; judge: frontier model via Ollama cloud.
JUDGE_MODEL=qwen3-coder:480b-cloud \
JUDGE_BASE_URL=http://localhost:11434/v1 \
JUDGE_API_KEY=ollama \
python examples/ollama_with_cloud_judge.py
```

See the [`examples/ollama_with_cloud_judge.py`](../examples/ollama_with_cloud_judge.py) script for a full runnable example.

### Hardware requirements

| Model size | VRAM required | Example models |
|-----------|---------------|----------------|
| 4B-9B | 6-8 GB | gemma3:4b, qwen3:8b |
| 12B-14B | 10-12 GB | gemma3:12b, mistral-small |
| 27B-32B | 20-24 GB | gemma3:27b, qwen3:32b |
| 70B+ | 48+ GB (or quantised) | llama-3.1:70b, qwen2.5:72b |

### Timeouts

Large models can take minutes to load into memory on first call. The provider uses generous timeouts:
- TCP connection: 30 seconds
- Response wait: 10 minutes (covers model loading)
- Total: 10 minutes

---

## Provider Comparison

| Feature | OpenRouter | Anthropic | OpenAI | Google | Ollama |
|---------|-----------|-----------|--------|--------|--------|
| Models available | 200+ | Claude family | GPT family | Gemini/Gemma | Any open-weight |
| API key | `OPENROUTER_API_KEY` | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` | `GOOGLE_API_KEY` | None (local) |
| Cost | Pay-per-token | Pay-per-token | Pay-per-token | Pay-per-token | Free (your hardware) |
| Rate limits | Generous | Moderate | Moderate | Moderate | None |
| Free tier | Yes (some models) | No | No | Yes (limited) | Free |
| Multi-model | Single key for all | Claude only | GPT only | Gemini only | One at a time |
| Best for | Benchmarking many models | Claude-specific work | GPT-specific work | Gemini-specific work | Open-weight models |

## Recommended Approach

For reproducing our 35-model results, we recommend **OpenRouter** — one API key gives access to all model families. This is how the v1.0 dataset results were collected (March 29 - April 3, 2026).

For running a single model family, use the direct API (lower latency, no intermediary).

For open-weight models on your own hardware, use **Ollama**.

---

## Choosing a Judge Model

The Tier-3 (Epistemic Articulation) score and several Tier-2 sub-measures come from an LLM judge that scores 30 criteria on a 0–3 scale using the anti-rhetoric rubric in `src/tracks/metacognition/prompts/judge.py`. The judge's job is strict structured JSON output, not deep reasoning — so the best judge is a fast, cheap, non-reasoning model with reliable JSON compliance.

### Recommended: Gemini 2.5 Flash

**Gemini 2.5 Flash** is our default judge recommendation:

- Fast and cheap (fractions of a cent per instance).
- Excellent at structured JSON output with the anti-rhetoric rubric.
- Non-reasoning — returns content directly with no thinking-budget concerns.
- Reachable via Google's OpenAI-compatible endpoint, so it plugs straight into `call_judge_v2`.

```bash
export GOOGLE_API_KEY="AI..."
JUDGE_MODEL=gemini-2.5-flash \
JUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
JUDGE_API_KEY=$GOOGLE_API_KEY \
python examples/ollama_with_cloud_judge.py
```

Similar-tier alternatives: `claude-haiku-4.5` (via Anthropic or OpenRouter), `openai/gpt-4.1-mini` (via OpenRouter).

### Free / offline alternative: Ollama cloud

For air-gapped or fully-offline setups, any of Ollama's cloud-tagged models works as a judge. Non-reasoning models (which route their output through the standard `content` field) are the easiest to use:

| Ollama cloud model | Reasoning mode | Judge-ready out of the box? |
|---|---|---|
| `qwen3-coder:480b-cloud` | No | Yes — returns content directly |
| `gpt-oss:20b-cloud` | Yes | Yes — handled via `reasoning`-field fallback (see below) |
| `glm-4.6:cloud` | Yes | Yes — handled via `reasoning`-field fallback |
| `deepseek-v3.1:671b-cloud` | Yes | Yes — handled via `reasoning`-field fallback |
| `minimax-m2:cloud` | Yes | Yes — handled via `reasoning`-field fallback |

### Reasoning-model judges

Reasoning models (gpt-oss, glm-4.6, the Qwen3 "thinking" family, DeepSeek v3.1, MiniMax M2, …) route their chain of thought through a separate `reasoning` field on the chat completion response and may leave `message.content` empty until their thinking budget is exhausted. `call_judge_v2` and `call_judge_solo` handle this automatically: they concatenate `content`, `reasoning`, and `reasoning_content`, and the JSON-block parser picks the rubric output out of whichever field it lands in. You do not need to disable thinking or do anything special — just pass the model id.

The default `max_tokens` for `call_judge_v2` is 4096 to give reasoning models room to finish thinking *and* emit the structured output. If you see a "Judge returned mostly defaults" warning with a reasoning model, try raising `max_tokens` further via the `call_judge_v2(..., max_tokens=8192)` argument.

### When the judge is called

Note that `medley-bench benchmark` does **not** call a live judge during the target-model run — it only reads precomputed judge scores from `{data_dir}/precomputed_judge_scores.json` when present. Raw responses are always saved in each result file, so the judge can be run as a post-processing pass. The example script at `examples/ollama_with_cloud_judge.py` shows the live-judge pattern: it calls `call_judge_v2` inline after each instance, which is the recommended way to run one model with a judge end-to-end in a single script.
