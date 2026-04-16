"""LLM provider abstraction for MEDLEY-BENCH.

Async interface with retry logic and token tracking.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Tracks token usage and cost across calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0  # Extended thinking tokens (billed but not scored)
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider implementations."""

    @property
    def model_name(self) -> str: ...

    async def complete(self, prompt: str, **kwargs) -> str: ...


async def _retry_with_backoff(coro_fn, max_retries: int = 5, base_delay: float = 2.0):
    """Retry an async function with exponential backoff.

    Special handling:
    - Rate limits (429): aggressive backoff (3^n)
    - Timeouts: longer waits (model may be loading into memory)
    - Other errors: standard backoff (2^n)
    """
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            err_str = str(e).lower()
            err_type = type(e).__name__.lower()

            if "429" in err_str or "rate" in err_str:
                delay = base_delay * (3 ** attempt)
            elif "timeout" in err_str or "timeout" in err_type:
                # Model may be loading — wait longer, log clearly
                delay = max(30, base_delay * (2 ** attempt))
                logger.warning(
                    "Attempt %d timed out (model may be loading), retrying in %.0fs...",
                    attempt + 1, delay,
                )
                await asyncio.sleep(delay)
                continue
            else:
                delay = base_delay * (2 ** attempt)

            logger.warning(
                "Attempt %d failed (%s: %s), retrying in %.1fs...",
                attempt + 1, type(e).__name__, str(e)[:80], delay,
            )
            await asyncio.sleep(delay)


@dataclass
class AnthropicProvider:
    """Anthropic Claude API provider."""
    model_id: str = "claude-sonnet-4-20250514"
    max_tokens: int = 10000
    temperature: float = 0.0
    usage: UsageStats = field(default_factory=UsageStats)
    _client: object = field(default=None, repr=False)

    @property
    def model_name(self) -> str:
        return self.model_id

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        return self._client

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()

        async def _call():
            create_kwargs = dict(
                model=kwargs.get("model", self.model_id),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                messages=[{"role": "user", "content": prompt}],
            )
            # Only set temperature for non-thinking models (thinking models
            # ignore temperature and raise errors on some API versions)
            if not kwargs.get("thinking", False):
                create_kwargs["temperature"] = kwargs.get("temperature", self.temperature)
            else:
                # Enable extended thinking for Claude models
                budget = kwargs.get("thinking_budget", 8000)
                create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                # thinking requires max_tokens > budget_tokens
                create_kwargs["max_tokens"] = max(create_kwargs["max_tokens"], budget + 4000)

            response = await client.messages.create(**create_kwargs)
            self.usage.calls += 1
            self.usage.input_tokens += response.usage.input_tokens
            self.usage.output_tokens += response.usage.output_tokens

            # Extract text from content blocks, skipping thinking blocks
            # Claude returns: [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]
            text_parts = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "thinking":
                    # Track thinking tokens but don't include in output
                    thinking_text = getattr(block, "thinking", "")
                    self.usage.thinking_tokens += len(thinking_text) // 4  # approx tokens
                    logger.debug("Anthropic thinking: %d chars", len(thinking_text))
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
            return "\n".join(text_parts) if text_parts else response.content[0].text

        return await _retry_with_backoff(_call)


@dataclass
class OpenAIProvider:
    """OpenAI API provider."""
    model_id: str = "gpt-4o"
    max_tokens: int = 10000
    temperature: float = 0.0
    usage: UsageStats = field(default_factory=UsageStats)
    _client: object = field(default=None, repr=False)

    @property
    def model_name(self) -> str:
        return self.model_id

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        return self._client

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        is_reasoning = any(x in self.model_id.lower() for x in ["o1", "o3", "o4"])

        async def _call():
            create_kwargs = dict(
                model=kwargs.get("model", self.model_id),
                messages=[{"role": "user", "content": prompt}],
            )
            if is_reasoning:
                # o1/o3/o4 models: no temperature, use reasoning_effort
                create_kwargs["reasoning_effort"] = kwargs.get("reasoning_effort", "medium")
                create_kwargs["max_completion_tokens"] = kwargs.get("max_tokens", self.max_tokens)
            else:
                create_kwargs["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
                create_kwargs["temperature"] = kwargs.get("temperature", self.temperature)

            response = await client.chat.completions.create(**create_kwargs)
            self.usage.calls += 1
            if response.usage:
                self.usage.input_tokens += response.usage.prompt_tokens
                self.usage.output_tokens += response.usage.completion_tokens
                # Track reasoning tokens for o-series models
                if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                    rt = getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0)
                    if rt:
                        self.usage.thinking_tokens += rt
                        logger.debug("OpenAI reasoning tokens: %d", rt)
            return response.choices[0].message.content

        return await _retry_with_backoff(_call)


@dataclass
class OpenRouterProvider:
    """OpenRouter API provider (OpenAI-compatible, access to many models incl. free)."""
    model_id: str = "meta-llama/llama-3.2-3b-instruct:free"
    max_tokens: int = 10000
    temperature: float = 0.0
    usage: UsageStats = field(default_factory=UsageStats)
    _client: object = field(default=None, repr=False)

    @property
    def model_name(self) -> str:
        return self.model_id

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        return self._client

    async def complete(self, prompt: str, **kwargs) -> str:
        client = self._get_client()

        async def _call():
            response = await client.chat.completions.create(
                model=kwargs.get("model", self.model_id),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": prompt}],
            )
            self.usage.calls += 1
            if response.usage:
                self.usage.input_tokens += response.usage.prompt_tokens or 0
                self.usage.output_tokens += response.usage.completion_tokens or 0
            return response.choices[0].message.content

        return await _retry_with_backoff(_call)


@dataclass
class GoogleProvider:
    """Google Generative AI provider."""
    model_id: str = "gemini-2.5-pro"
    max_tokens: int = 10000
    temperature: float = 0.0
    usage: UsageStats = field(default_factory=UsageStats)
    _model: object = field(default=None, repr=False)

    @property
    def model_name(self) -> str:
        return self.model_id

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self._model = genai.GenerativeModel(self.model_id)
        return self._model

    async def complete(self, prompt: str, **kwargs) -> str:
        model = self._get_model()

        async def _call():
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={
                    "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                },
            )
            self.usage.calls += 1
            return response.text

        return await _retry_with_backoff(_call)


@dataclass
class OllamaProvider:
    """Ollama model provider — local or remote (Open WebUI compatible).

    Supports:
    - Local Ollama: OLLAMA_BASE_URL=http://localhost:11434
    - Remote Open WebUI: OLLAMA_BASE_URL=https://llm.example.com
      with OLLAMA_API_KEY for authentication
    """
    model_id: str = "llama3.2"
    max_tokens: int = 10000
    temperature: float = 0.0
    base_url: str = ""
    api_key: str = ""
    usage: UsageStats = field(default_factory=UsageStats)

    def __post_init__(self):
        if not self.base_url:
            self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        if not self.api_key:
            self.api_key = os.environ.get("OLLAMA_API_KEY", "")

    @property
    def model_name(self) -> str:
        return f"ollama/{self.model_id}"

    async def complete(self, prompt: str, **kwargs) -> str:
        import aiohttp

        # Large models (llama4:scout=67GB, mistral-large=675B) can take
        # minutes to load into memory on first call. Use generous timeouts.
        # sock_connect: time to establish TCP connection
        # sock_read: time to receive first byte (model loading happens here)
        # total: overall timeout for the full request
        timeout = aiohttp.ClientTimeout(
            total=600,       # 10 minutes total (covers model loading + generation)
            sock_connect=30, # 30s to establish connection
            sock_read=600,   # 10 minutes to wait for response (model loading)
        )

        async def _call():
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            is_remote = self.base_url.startswith("https://")
            # Cloud-routed models (e.g., "gpt-oss:20b-cloud") run through local
            # Ollama but route to a cloud backend. They need /api/chat (not /api/generate).
            is_cloud_routed = "cloud" in self.model_id.lower() or "-cloud" in self.model_id.lower()

            # Disable thinking/reasoning mode for models that support it.
            # Thinking models waste tokens on <think>...</think> chains,
            # leaving too few tokens for structured JSON output.
            no_think = kwargs.get("no_think", True)  # Default: disable thinking

            if is_remote or is_cloud_routed:
                # Open WebUI: use OpenAI-compatible chat/completions endpoint
                # For Qwen3/3.5 models, prepend /no_think to disable thinking
                actual_prompt = prompt
                if no_think and any(x in self.model_id.lower() for x in ["qwen3", "deepseek-r1", "magistral"]):
                    actual_prompt = "/no_think\n\n" + prompt
                _max_out = kwargs.get("max_tokens", self.max_tokens)
                payload = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": actual_prompt}],
                    "max_tokens": _max_out,
                    "temperature": kwargs.get("temperature", self.temperature),
                    # Ensure context window fits input + output (min 24096 for 20k output)
                    "num_ctx": max(kwargs.get("num_ctx", 16384), _max_out + 4096),
                }
                if is_remote:
                    # Open WebUI: OpenAI-compatible endpoint
                    url = f"{self.base_url}/api/chat/completions"
                else:
                    # Cloud-routed through local Ollama: use native /api/chat
                    url = f"{self.base_url}/api/chat"
                    # Reformat payload for Ollama native chat format
                    payload = {
                        "model": self.model_id,
                        "messages": [{"role": "user", "content": actual_prompt}],
                        "stream": False,
                        "options": {
                            "num_predict": _max_out,
                            "temperature": kwargs.get("temperature", self.temperature),
                            "num_ctx": max(kwargs.get("num_ctx", 16384), _max_out + 4096),
                        },
                    }
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers,
                                           timeout=timeout) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        self.usage.calls += 1
                        if is_remote:
                            return data["choices"][0]["message"]["content"]
                        else:
                            return data.get("message", {}).get("content", "")
            else:
                # Local Ollama: use native /api/generate endpoint
                # For Qwen3/3.5, prepend /no_think to disable thinking
                actual_prompt = prompt
                if no_think and any(x in self.model_id.lower() for x in ["qwen3", "deepseek-r1", "magistral"]):
                    actual_prompt = "/no_think\n\n" + prompt
                _max_out = kwargs.get("max_tokens", self.max_tokens)
                payload = {
                    "model": self.model_id,
                    "prompt": actual_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": _max_out,
                        "temperature": kwargs.get("temperature", self.temperature),
                        # Ensure context window fits input + output (min 24096 for 20k output)
                        "num_ctx": max(kwargs.get("num_ctx", 16384), _max_out + 4096),
                    },
                    # Ollama native thinking control (supported by QwQ, Qwen3, etc.)
                    "think": not no_think,
                }
                url = f"{self.base_url}/api/generate"
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers,
                                           timeout=timeout) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        self.usage.calls += 1
                        # Ollama native API separates thinking from response:
                        #   data["thinking"] = reasoning trace (if think=true)
                        #   data["response"] = clean answer
                        thinking_text = data.get("thinking", "")
                        if thinking_text:
                            self.usage.thinking_tokens += len(thinking_text) // 4
                            logger.debug("Ollama thinking: %d chars", len(thinking_text))
                        return data.get("response", "")

        return await _retry_with_backoff(_call)


# Provider registry
_PROVIDER_MAP: dict[str, type] = {}


def _init_provider_map():
    """Build prefix-to-provider mapping."""
    global _PROVIDER_MAP
    _PROVIDER_MAP = {
        "claude": AnthropicProvider,
        "gpt": OpenAIProvider,
        "o1": OpenAIProvider,
        "o3": OpenAIProvider,
        "gemini": GoogleProvider,
    }

# OpenRouter free models for testing (verified available)
OPENROUTER_FREE_MODELS = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-4b-it:free",
]


def _load_env_file():
    """Load API keys from .env file if present (no dependency on python-dotenv)."""
    for env_path in [".env", os.path.expanduser("~/.medley-bench.env")]:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            logger.debug("Loaded API keys from %s", env_path)
            return
    logger.debug("No .env file found (checked .env, ~/.medley-bench.env)")


# Required environment variable per provider
_REQUIRED_KEYS = {
    "AnthropicProvider": ("ANTHROPIC_API_KEY", "https://console.anthropic.com"),
    "OpenAIProvider": ("OPENAI_API_KEY", "https://platform.openai.com"),
    "OpenRouterProvider": ("OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    "GoogleProvider": ("GOOGLE_API_KEY", "https://aistudio.google.com"),
}


def get_provider(model_id: str, **kwargs) -> LLMProvider:
    """Factory: create an LLM provider from a model ID string.

    Routing:
    - 'ollama/model' -> OllamaProvider (local, no API key needed)
    - 'org/model' -> OpenRouterProvider (requires OPENROUTER_API_KEY)
    - 'claude-*' -> AnthropicProvider, 'gpt-*' -> OpenAIProvider, etc.

    API keys are read from environment variables or ~/.medley-bench.env.
    """
    # Load .env file on first call
    _load_env_file()

    if not _PROVIDER_MAP:
        _init_provider_map()

    # Ollama: explicit ollama/ prefix (no API key required for local)
    if model_id.startswith("ollama/"):
        ollama_model = model_id[len("ollama/"):]
        return OllamaProvider(model_id=ollama_model, **kwargs)

    # Determine provider class
    provider_cls = None
    if "/" in model_id:
        provider_cls = OpenRouterProvider
    else:
        for prefix, cls in _PROVIDER_MAP.items():
            if model_id.lower().startswith(prefix):
                provider_cls = cls
                break

    if provider_cls is None:
        raise ValueError(
            f"Unknown model '{model_id}'. Known prefixes: {list(_PROVIDER_MAP.keys())}. "
            f"Use 'ollama/model' for local, 'org/model' for OpenRouter."
        )

    # Validate API key is set
    cls_name = provider_cls.__name__
    if cls_name in _REQUIRED_KEYS:
        env_var, signup_url = _REQUIRED_KEYS[cls_name]
        if not os.environ.get(env_var):
            raise EnvironmentError(
                f"\n{'='*60}\n"
                f"  API key not found: {env_var}\n\n"
                f"  Set it via environment variable:\n"
                f"    export {env_var}='your-key-here'\n\n"
                f"  Or create a ~/.medley-bench.env file:\n"
                f"    {env_var}=your-key-here\n\n"
                f"  Get a key at: {signup_url}\n"
                f"  See: docs/PROVIDERS.md\n"
                f"{'='*60}"
            )

    if "/" in model_id and provider_cls == OpenRouterProvider:
        return OpenRouterProvider(model_id=model_id, **kwargs)
    return provider_cls(model_id=model_id, **kwargs)
