"""OpenAI-compatible client with retries and automatic provider failover.

Every call records tokens + latency. If a FALLBACK_* provider is configured
and the primary fails (rate limit, outage), the same call transparently
retries there — a live demo never surfaces a provider hiccup.
"""
import sys
import time

from openai import OpenAI

from . import config


class LLMNotConfigured(Exception):
    pass


_clients = {}


def _client(base_url, api_key):
    key = (base_url, api_key)
    if key not in _clients:
        # fail fast into our own retry/failover instead of hanging on a dead call
        _clients[key] = OpenAI(base_url=base_url, api_key=api_key,
                               timeout=config.LLM_TIMEOUT, max_retries=0)
    return _clients[key]


def _providers(model):
    """[(client, model, system_prefix), ...] — primary first, then fallback."""
    if not config.LLM_API_KEY:
        raise LLMNotConfigured(
            "No API key set. Copy .env.example to .env and set GROQ_API_KEY "
            "(or OPENAI_API_KEY with LLM_BASE_URL/GEN_MODEL for another provider)."
        )
    providers = [(_client(config.LLM_BASE_URL, config.LLM_API_KEY), model, "")]
    if config.FALLBACK_API_KEY and config.FALLBACK_BASE_URL:
        fb_model = (config.FALLBACK_GEN_MODEL if model == config.GEN_MODEL
                    else config.FALLBACK_FAST_MODEL) or model
        providers.append((_client(config.FALLBACK_BASE_URL, config.FALLBACK_API_KEY),
                          fb_model, config.FALLBACK_GEN_SYSTEM_PREFIX))
    return providers


def chat(messages, model, json_mode=False, temperature=0.2, max_tokens=800, usage_log=None):
    providers = _providers(model)
    last_err = None
    for p_idx, (client, p_model, prefix) in enumerate(providers):
        msgs = messages
        if prefix:
            if msgs and msgs[0].get("role") == "system":
                if not msgs[0]["content"].startswith(prefix):
                    msgs = ([{"role": "system", "content": prefix + msgs[0]["content"]}]
                            + list(msgs[1:]))
            else:
                # No system message (smalltalk / query composer): the prefix
                # must still reach the provider, or e.g. Nemotron emits
                # <think> reasoning tokens straight into user-visible text.
                msgs = [{"role": "system", "content": prefix.strip()}] + list(msgs)
        kwargs = dict(model=p_model, messages=msgs,
                      temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        attempts, max_attempts = 0, (2 if len(providers) > 1 else 4)
        while attempts < max_attempts:
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
                if usage_log is not None and resp.usage:
                    usage_log.append({
                        "model": p_model,
                        "input_tokens": resp.usage.prompt_tokens,
                        "output_tokens": resp.usage.completion_tokens,
                        "latency_s": round(latency, 3),
                    })
                return resp.choices[0].message.content or ""
            except Exception as e:
                msg = str(e).lower()
                # Some providers/models reject response_format — the prompts
                # already demand JSON-only output, so drop it and continue.
                # Guarded on the key actually being present, so this branch can
                # fire at most once and can never become an infinite loop.
                if "response_format" in kwargs and (
                        "response_format" in msg or "json_object" in msg):
                    kwargs.pop("response_format", None)
                    continue
                last_err = e
                attempts += 1
                rate_limited = ("429" in msg or "rate" in msg or "quota" in msg
                                or "exhausted" in msg)
                # Rate-limited with a fallback available: fail over NOW —
                # waiting out a quota window mid-demo is pointless.
                if rate_limited and p_idx < len(providers) - 1:
                    break
                if attempts < max_attempts:
                    # Short, capped backoff: the old 10/20/30s waits produced
                    # 60s+ worst-case answers (visible in metrics.jsonl).
                    time.sleep(min((3 if rate_limited else 1) * attempts, 8))
        if p_idx < len(providers) - 1:
            print(f"[llm] failing over to backup provider ({str(last_err)[:160]})",
                  file=sys.stderr)
    raise last_err if last_err is not None else RuntimeError("LLM call failed")
