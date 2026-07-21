"""Multimodal LLM — cited answer synthesis from frames, per-tenant switchable.

Every call takes an LLMConfig. Where it comes from (resolved in
src/rag/search.py):
  1. the user's own hosted model (ms_user_llms row — a vLLM/Ollama/LM Studio/
     Together/OpenRouter endpoint via base_url, NVIDIA NIM, or Anthropic), or
  2. the server-wide LLM_* env config as the fallback.

The two multimodal calls are where latency and cost actually live (retrieval
is milliseconds), so frames are downscaled to LLM_IMAGE_MAX_PX before they are
sent and only TOP_K of them ever reach the model.

Providers:
  * "openai"    — Chat Completions; covers every OpenAI-compatible server
                  (vLLM, Ollama, LM Studio, Together, Groq, OpenRouter, ...)
                  via base_url.
  * "nvidia"    — NVIDIA NIM / build.nvidia.com hosted vision models.
                  OpenAI-compatible, same client with NVIDIA's endpoint.
  * "anthropic" — the Anthropic Messages API.

Provider SDKs are imported lazily — only the one you use.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from . import config

# NVIDIA's hosted inference endpoint (OpenAI-compatible).
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

PROVIDERS = ("openai", "nvidia", "anthropic")

SYSTEM = (
    "You answer questions about a video using ONLY the numbered frames provided. "
    "The frames are stills sampled from the video at specific timestamps. "
    "Describe what is visibly shown. Cite every claim with the frame number(s) "
    "in square brackets, e.g. [1] or [2, 3]. If the frames do not show enough to "
    "answer, say so plainly — never invent detail that isn't visible."
)


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 1024


def env_config() -> LLMConfig | None:
    """The server-wide fallback model from LLM_* env vars, if configured."""
    if not config.llm_configured():
        return None
    return LLMConfig(provider=config.LLM_PROVIDER, model=config.LLM_MODEL,
                     api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                     max_tokens=config.LLM_MAX_TOKENS)


def from_row(row: dict) -> LLMConfig:
    """A tenant's own hosted model (ms_user_llms row)."""
    return LLMConfig(provider=row.get("provider") or "openai",
                     model=row.get("model") or "",
                     api_key=row.get("api_key") or "",
                     base_url=row.get("base_url") or "",
                     max_tokens=config.LLM_MAX_TOKENS)


def _prompt(question: str, n: int) -> str:
    return (
        f"Question: {question}\n\n"
        f"You are given {n} frames, numbered 1 to {n} in order. "
        "Answer the question from what is visible, citing frames as [n]."
    )


def _downscale(jpeg: bytes) -> bytes:
    """Shrink a frame before it becomes LLM image tokens."""
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg))
    if max(img.size) <= config.LLM_IMAGE_MAX_PX:
        return jpeg
    img.thumbnail((config.LLM_IMAGE_MAX_PX, config.LLM_IMAGE_MAX_PX))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def answer(question: str, frames: list[bytes], cfg: LLMConfig) -> str:
    """Synthesize a cited answer from retrieved frame JPEGs with `cfg`'s model."""
    frames = [_downscale(f) for f in frames]
    if cfg.provider == "anthropic":
        return _answer_anthropic(cfg, question, frames)
    return _answer_openai(cfg, question, frames)


def ping(cfg: LLMConfig) -> str:
    """Connectivity + vision check: one tiny image, one word back. Raises with
    the provider's error on failure (surfaced to the settings UI)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (220, 40, 40)).save(buf, format="JPEG")
    return answer("Reply with the dominant color of frame 1, one word.",
                  [buf.getvalue()], cfg)


def _base_url(cfg: LLMConfig) -> str | None:
    if cfg.base_url:
        return cfg.base_url
    if cfg.provider == "nvidia":
        return NVIDIA_BASE_URL
    return None


def _answer_openai(cfg: LLMConfig, question: str, frames: list[bytes]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key or "not-needed", base_url=_base_url(cfg))
    content: list[dict] = [{"type": "text", "text": _prompt(question, len(frames))}]
    for f in frames:
        uri = f"data:image/jpeg;base64,{base64.b64encode(f).decode()}"
        content.append({"type": "image_url", "image_url": {"url": uri}})
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=cfg.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _answer_anthropic(cfg: LLMConfig, question: str, frames: list[bytes]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.api_key,
                                 base_url=cfg.base_url or None)
    blocks: list[dict] = [{"type": "text", "text": _prompt(question, len(frames))}]
    for f in frames:
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(f).decode()}})
    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=SYSTEM,
        messages=[{"role": "user", "content": blocks}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()
