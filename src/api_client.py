"""
src/api_client.py
=================
Unified async API client for OpenRouter.
Handles retries, rate-limiting, cost tracking, and response parsing.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    raw: dict = field(default_factory=dict)


@dataclass
class CostTracker:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Approximate costs per 1M tokens (USD) — update as needed
    COST_PER_1M = {
        "anthropic/claude-sonnet-4-5": {"in": 3.0,  "out": 15.0},
        "openai/o1-mini":              {"in": 3.0,  "out": 12.0},
        "openai/gpt-4o-mini":          {"in": 0.15, "out": 0.60},
        "deepseek/deepseek-r1":        {"in": 0.55, "out": 2.19},
        "deepseek/deepseek-chat":      {"in": 0.27, "out": 1.10},
        "deepseek/deepseek-r1-distill-llama-70b": {"in": 0.23, "out": 0.69},
        "meta-llama/llama-3.1-70b-instruct":      {"in": 0.12, "out": 0.30},
        "qwen/qwq-32b":                {"in": 0.15, "out": 0.60},
        "qwen/qwen-2.5-72b-instruct":  {"in": 0.35, "out": 0.40},
        "deepseek/deepseek-r1-distill-qwen-7b":   {"in": 0.10, "out": 0.20},
        "qwen/qwen-2.5-7b-instruct":   {"in": 0.10, "out": 0.20},
    }

    def add(self, model: str, prompt_tok: int, completion_tok: int):
        self.calls += 1
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok

    def estimated_cost_usd(self, model: str, prompt_tok: int, completion_tok: int) -> float:
        rates = self.COST_PER_1M.get(model, {"in": 1.0, "out": 3.0})
        return (prompt_tok * rates["in"] + completion_tok * rates["out"]) / 1_000_000

    def summary(self) -> dict:
        return {
            "total_calls": self.calls,
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


# Global tracker
cost_tracker = CostTracker()


class OpenRouterClient:
    """
    Synchronous OpenRouter client with retry logic and rate limiting.
    Use call_model() for all requests.
    """

    def __init__(self, api_key: str, base_url: str, extra_headers: dict = None,
                 max_retries: int = 3, retry_delay: float = 5.0,
                 requests_per_minute: int = 20):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rpm = requests_per_minute
        self._call_times: list[float] = []

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _rate_limit(self):
        """Sliding-window rate limiter — enforces self.rpm."""
        now = time.time()
        # Remove calls older than 60 s
        self._call_times = [t for t in self._call_times if now - t < 60]
        if len(self._call_times) >= self.rpm:
            sleep_for = 60 - (now - self._call_times[0]) + 0.1
            logger.debug(f"Rate limit: sleeping {sleep_for:.1f}s")
            time.sleep(max(0, sleep_for))
        self._call_times.append(time.time())

    def call_model(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system: Optional[str] = None,
        json_mode: bool = False,
        reasoning_enabled: Optional[bool] = None,
        extra_params: dict = None,
    ) -> APIResponse:
        """
        Call a model via OpenRouter. Returns APIResponse.
        Raises RuntimeError after max_retries failures.
        """
        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_params:
            payload.update(extra_params)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Reasoning toggle for hybrid models (Qwen3, Gemini Flash, DeepSeek V3.1)
        if reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": reasoning_enabled}

        last_error = None
        for attempt in range(self.max_retries):
            self._rate_limit()
            t0 = time.time()
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=120,
                )
                latency = time.time() - t0

                if resp.status_code == 429:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"429 rate-limited. Sleeping {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Server error {resp.status_code}. Sleeping {wait}s")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    raise RuntimeError(f"API error: {data['error']}")

                content = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {})
                prompt_tok = usage.get("prompt_tokens", 0)
                comp_tok   = usage.get("completion_tokens", 0)

                cost_tracker.add(model, prompt_tok, comp_tok)
                cost = cost_tracker.estimated_cost_usd(model, prompt_tok, comp_tok)
                logger.debug(f"{model} | {prompt_tok}+{comp_tok} tok | ${cost:.4f} | {latency:.1f}s")

                return APIResponse(
                    content=content,
                    model=model,
                    prompt_tokens=prompt_tok,
                    completion_tokens=comp_tok,
                    total_tokens=prompt_tok + comp_tok,
                    latency_s=latency,
                    raw=data,
                )

            except requests.exceptions.Timeout:
                last_error = "Timeout"
                logger.warning(f"Timeout on attempt {attempt+1}")
                time.sleep(self.retry_delay)
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Request error attempt {attempt+1}: {e}")
                time.sleep(self.retry_delay)
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Unexpected error attempt {attempt+1}: {e}")
                time.sleep(self.retry_delay)

        raise RuntimeError(f"All {self.max_retries} attempts failed. Last error: {last_error}")

    def call_json(self, model: str, messages: list[dict], system: str = None,
                  temperature: float = 0.3, max_tokens: int = 2048) -> dict:
        """Call model and parse JSON response. Tolerant of trailing garbage."""
        resp = self.call_model(model, messages, temperature=temperature,
                               max_tokens=max_tokens, system=system, json_mode=True)
        text = resp.content.strip()
        # Strip ```json ... ``` fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Truncate at last valid closing bracket (handles "Extra data" errors)
        for closer in ("]", "}"):
            last = text.rfind(closer)
            if last != -1:
                try:
                    return json.loads(text[:last + 1])
                except json.JSONDecodeError:
                    pass
        # Regex fallback: grab first complete array or object
        for pattern in (r'(\[.*?\])', r'(\{.*?\})'):
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        logger.error(f"JSON parse failed after all attempts.\nRaw: {text[:500]}")
        raise json.JSONDecodeError("Could not parse JSON", text, 0)


def make_client() -> OpenRouterClient:
    """Build a client from config."""
    from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_HEADERS
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set. Export it before running.")
    return OpenRouterClient(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        extra_headers=OPENROUTER_HEADERS,
        max_retries=3,
        retry_delay=5.0,
        requests_per_minute=20,
    )
