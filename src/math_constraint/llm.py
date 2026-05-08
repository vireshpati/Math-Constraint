"""Single unified LLM client.

One :class:`LLMClient` class talks to anything that speaks the OpenAI
Chat-Completions wire format: OpenRouter, vLLM, HuggingFace Inference router,
or any custom proxy. Provider-specific defaults (base URL, env-var name for
the API key, app-attribution headers, provider-routing knobs) are selected
by :func:`create_client`.

Why this works: vLLM, HuggingFace's Inference router, and OpenRouter all
expose the same ``POST /chat/completions`` schema. They differ only in:
  * which URL you hit
  * which env var holds the API key
  * a small set of OpenRouter-specific request fields (``provider``,
    ``models``, ``reasoning``) that other endpoints ignore
  * a small set of OpenRouter-specific response fields (``cost``,
    ``reasoning_tokens``, ``cached_tokens``) that other endpoints simply
    don't return

So one client carries them all; the factory just chooses defaults.

Tool calling is uniform: the OpenAI ``tools`` array works the same on
OpenRouter and on HuggingFace's router endpoint for any model whose chat
template understands tool tokens (Llama 3.1+, Qwen 2.5+, etc).
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HUGGINGFACE_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

# Retryable per OpenRouter's error-and-debugging docs: 408 timeout, 429 rate
# limit, 502 upstream down, 503 no provider available. 504 isn't documented
# but is observed in the wild and is plainly retryable. Same policy is fine
# for any OpenAI-compatible backend.
_RETRYABLE_STATUS = {408, 429, 502, 503, 504}


@dataclass
class LLMResponse:
    """Unified response across all providers.

    OpenRouter populates the optional fields (``cost``, ``reasoning_tokens``,
    ``cached_tokens``, ``finish_reason``, ``native_finish_reason``,
    ``refusal``); other providers leave them as ``None`` / 0.
    """

    content: str | None
    tool_calls: list[dict] | None
    prompt_tokens: int
    completion_tokens: int
    response_id: str | None = None

    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cost: float | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    refusal: str | None = None
    reasoning: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Single OpenAI-Chat-Completions client.

    Works against any compatible endpoint. OpenRouter-specific knobs
    (``provider_routing``, ``models_fallback``, ``HTTP-Referer`` /
    ``X-OpenRouter-Title`` headers) are accepted and forwarded; non-
    OpenRouter backends ignore unknown extra-body fields and irrelevant
    headers, so passing them is harmless.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None,
        *,
        default_headers: dict[str, str] | None = None,
        provider_routing: dict | None = None,
        models_fallback: list[str] | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        request_timeout: float = 600.0,
    ):
        from openai import OpenAI

        if not base_url:
            raise ValueError("LLMClient requires base_url")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",  # vLLM/internal proxies often ignore
            default_headers=default_headers or {},
            timeout=request_timeout,
        )
        self.model = model
        self.provider_routing = provider_routing or {}
        self.models_fallback = list(models_fallback) if models_fallback else []
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        reasoning: dict | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        stop: list[str] | str | None = None,
        tool_choice: str | dict | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop is not None:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = parallel_tool_calls
        if response_format:
            kwargs["response_format"] = response_format

        # Non-OpenAI-spec params live in extra_body. Unknown extra-body fields
        # are ignored by vLLM and HF's router, so this is safe across backends.
        extra_body: dict[str, Any] = {}
        if reasoning:
            extra_body["reasoning"] = reasoning
        if self.provider_routing:
            extra_body["provider"] = self.provider_routing
        if self.models_fallback:
            extra_body["models"] = self.models_fallback
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self._call_with_retry(kwargs, has_tools=bool(tools))
        return self._parse_response(response)

    # ------------------------------------------------------------------ retry

    def _call_with_retry(self, kwargs: dict[str, Any], *, has_tools: bool) -> Any:
        from openai import (
            APIStatusError,
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
        )

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except APIStatusError as e:
                status = getattr(e, "status_code", None)
                # Surface a helpful message for the classic vLLM tool-use
                # misconfiguration rather than a bare 400.
                if (
                    has_tools
                    and status == 400
                    and "tool choice requires --enable-auto-tool-choice" in str(e)
                ):
                    raise RuntimeError(
                        "vLLM rejected tool calls. Start vLLM with "
                        "--enable-auto-tool-choice and a model-compatible "
                        "--tool-call-parser value."
                    ) from e
                if status not in _RETRYABLE_STATUS or attempt == self.max_retries:
                    raise
                last_exc = e
                self._sleep_for_retry(e, attempt)
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                if attempt == self.max_retries:
                    raise
                last_exc = e
                self._sleep_for_retry(e, attempt)
        if last_exc:
            raise last_exc
        raise RuntimeError("LLMClient retry loop exited without result")

    def _sleep_for_retry(self, exc: Exception, attempt: int) -> None:
        retry_after = self._extract_retry_after(exc)
        if retry_after is not None:
            time.sleep(min(retry_after, self.backoff_cap))
            return
        delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_cap)
        delay *= 1 + random.uniform(-0.25, 0.25)
        time.sleep(max(delay, 0.0))

    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        if not headers:
            return None
        val = headers.get("Retry-After") or headers.get("retry-after")
        if not val:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    # ----------------------------------------------------------------- parser

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        # Some providers (observed: nvidia/nemotron free) occasionally return a
        # 200 with response.choices == None or []. Surface as an empty response
        # rather than crashing the eval with a TypeError.
        choices = getattr(response, "choices", None)
        if not choices:
            usage = getattr(response, "usage", None)
            return LLMResponse(
                content=None, tool_calls=None,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                response_id=getattr(response, "id", None),
                finish_reason="empty_response",
            )
        choice = choices[0]
        message = choice.message

        # Content can be str, list of typed blocks (Anthropic-via-OR), or None.
        content: str | None = None
        raw_content = getattr(message, "content", None)
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, dict):
                    t = item.get("text")
                    if isinstance(t, str):
                        parts.append(t)
                else:
                    t = getattr(item, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
            content = "".join(parts) if parts else None

        # Local vLLM with reasoning-parser enabled may stash the answer text
        # in message.reasoning_content. Promote it only if content is truly
        # empty AND the response isn't a tool-call (where empty content is
        # expected). Skipped for OpenRouter cloud responses since their
        # message.reasoning is internal CoT, not the answer.
        if (not content) and not message.tool_calls:
            rc = getattr(message, "reasoning_content", None)
            if isinstance(rc, str) and rc.strip():
                content = rc

        refusal = getattr(message, "refusal", None)

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        reasoning_tokens: int | None = None
        cached_tokens: int | None = None
        cost: float | None = None
        if usage is not None:
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd is not None:
                rt = getattr(ctd, "reasoning_tokens", None)
                if isinstance(rt, int):
                    reasoning_tokens = rt
            ptd = getattr(usage, "prompt_tokens_details", None)
            if ptd is not None:
                ct = getattr(ptd, "cached_tokens", None)
                if isinstance(ct, int):
                    cached_tokens = ct
            c = getattr(usage, "cost", None)
            if isinstance(c, (int, float)):
                cost = float(c)

        # OpenRouter cloud surfaces the CoT trace on `message.reasoning`.
        # Capture for trace logging; never confuse with the answer content.
        reasoning_text = getattr(message, "reasoning", None)
        if not isinstance(reasoning_text, str):
            reasoning_text = None

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_id=getattr(response, "id", None),
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
            finish_reason=getattr(choice, "finish_reason", None),
            native_finish_reason=getattr(choice, "native_finish_reason", None),
            refusal=refusal if isinstance(refusal, str) else None,
            reasoning=reasoning_text,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_client(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> LLMClient:
    """Construct an :class:`LLMClient` with provider-appropriate defaults.

    Supported providers:
      ``openrouter``  : OpenRouter cloud panel (the paper). Auto-pulls
        ``OPENROUTER_API_KEY`` and sets ``HTTP-Referer`` /
        ``X-OpenRouter-Title`` headers. Pass ``provider_routing=`` and
        ``models_fallback=`` for upstream-routing control.
      ``huggingface`` : HuggingFace Inference router
        (``https://router.huggingface.co/v1``). Auto-pulls ``HF_TOKEN``.
      ``vllm``        : Local vLLM server. ``base_url`` required (e.g.
        ``http://localhost:8000/v1``). API key defaults to ``"EMPTY"``.
      ``generic``     : Any OpenAI-Chat-Completions-compatible endpoint.
        ``base_url`` required.
    """
    if provider == "openrouter":
        token = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not token:
            raise ValueError("openrouter requires OPENROUTER_API_KEY env var (or api_key=...)")
        headers = kwargs.pop("default_headers", None) or {}
        referer = os.environ.get("OPENROUTER_REFERER")
        if referer:
            headers.setdefault("HTTP-Referer", referer)
        headers.setdefault(
            "X-OpenRouter-Title",
            os.environ.get("OPENROUTER_TITLE", "math-constraint-eval"),
        )
        return LLMClient(
            model=model,
            base_url=base_url or OPENROUTER_BASE_URL,
            api_key=token,
            default_headers=headers,
            **kwargs,
        )

    if provider == "huggingface":
        token = api_key or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("huggingface requires HF_TOKEN env var (or api_key=...)")
        return LLMClient(
            model=model,
            base_url=base_url or HUGGINGFACE_ROUTER_BASE_URL,
            api_key=token,
            **kwargs,
        )

    if provider == "vllm":
        if not base_url:
            raise ValueError("vllm requires base_url (e.g. http://localhost:8000/v1)")
        return LLMClient(model=model, base_url=base_url, api_key=api_key or "EMPTY", **kwargs)

    if provider == "generic":
        if not base_url:
            raise ValueError("generic requires base_url")
        return LLMClient(model=model, base_url=base_url, api_key=api_key, **kwargs)

    raise ValueError(
        f"Unknown provider: {provider}. Use one of: openrouter, huggingface, vllm, generic"
    )
