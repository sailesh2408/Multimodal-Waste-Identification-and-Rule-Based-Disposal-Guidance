"""Stage 1: identify the waste item with an open vision-language model.

The model only names the item and picks keys from the closed taxonomy.
It never decides the disposal method; that comes from the rules table.

Backends, tried in order (whichever keys are set):
  1. Google AI Studio (GEMINI_API_KEY) - Gemma 4 / Gemini Flash on your own free quota
  2. OpenRouter free tier (OPENROUTER_API_KEY) - shared free pool, often busy
  3. Hugging Face Inference Providers (HF_TOKEN) - same Gemma 4, needs HF credits
"""
import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import httpx
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field

from .taxonomy import (
    CONDITIONS, ITEM_TYPES, MATERIALS,
    keyword_guess, normalize_condition, normalize_item_type, normalize_material,
)

GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEFAULT_GOOGLE_MODELS = "gemma-4-26b-a4b-it,gemini-2.5-flash,gemini-2.5-flash-lite"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free vision models from different upstream providers, so one throttled upstream doesn't block us.
DEFAULT_OPENROUTER_MODELS = ",".join([
    "google/gemma-4-26b-a4b-it:free",       # Google AI Studio
    "google/gemma-4-31b-it:free",           # Google AI Studio
    "thinkingmachines/inkling-small:free",  # Thinking Machines
    "nvidia/nemotron-3.5-super-vl:free",    # NVIDIA
    "openrouter/free",                      # any free model that accepts images
])

DEFAULT_MODELS = "google/gemma-4-26B-A4B-it,Qwen/Qwen3-VL-8B-Instruct"  # Hugging Face

TIME_BUDGET_SECONDS = 35  # stop starting new attempts after this; Vercel's function limit is 60 s
MAX_TOKENS = 700
ACCOUNT_ERRORS = (401, 402, 403)  # the same key will fail for every model on that backend


class Identification(BaseModel):
    item_name: str = "Unknown item"
    item_type: str = "other"
    material: str = "unknown"
    condition: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alternatives: list[str] = []
    reason: str = ""
    model: str = ""


class VLMError(RuntimeError):
    pass


def build_prompt(note: str = "", has_image: bool = True) -> str:
    items = "\n".join(f"- {key}: {info['hint']}" for key, info in ITEM_TYPES.items())
    conditions = "\n".join(f"- {key}: {label}" for key, label in CONDITIONS.items())
    subject = "the main waste item in the photo" if has_image else "the waste item the user describes"
    user_note = f'\nThe user added this description: "{note.strip()}"\n' if note.strip() else ""
    return f"""You help people in India sort household waste. Identify {subject}.
{user_note}
Choose item_type from exactly one of these keys:
{items}

Choose material from: {", ".join(MATERIALS)}

Choose condition from:
{conditions}

Rules:
- If several items are visible, describe the most prominent one.
- If there is no waste item, use item_type "other" and confidence below 0.3.
- Look for safety-relevant details: swollen or leaking batteries, needles, broken glass, expiry dates.
- alternatives: up to 2 other item_type keys it could plausibly be, most likely first.
- confidence: a number from 0 to 1 for item_type.

Reply with ONLY a JSON object, no other text:
{{"item_name": "short everyday name", "item_type": "<key>", "material": "<material>", "condition": "<condition>", "confidence": 0.0, "alternatives": ["<key>"], "reason": "one short sentence on what you saw"}}"""


def parse_identification(text: str, model: str = "") -> Identification:
    """Pull the JSON object out of a model reply and coerce it onto the taxonomy."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    data = json.loads(text[start:end + 1])

    item_type = normalize_item_type(data.get("item_type", ""))
    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence > 1:
        confidence /= 100
    confidence = min(max(confidence, 0.0), 1.0)

    raw_alts = data.get("alternatives") or []
    if isinstance(raw_alts, str):
        raw_alts = [raw_alts]
    alternatives = []
    for alt in raw_alts:
        key = normalize_item_type(alt)
        if key not in (item_type, "other") and key not in alternatives:
            alternatives.append(key)

    return Identification(
        item_name=str(data.get("item_name") or ITEM_TYPES[item_type]["label"])[:80],
        item_type=item_type,
        material=normalize_material(data.get("material", "")),
        condition=normalize_condition(data.get("condition", "")),
        confidence=confidence,
        alternatives=alternatives[:3],
        reason=str(data.get("reason") or "")[:200],
        model=model,
    )


def _split(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


class OpenAICompatible:
    """Minimal client for OpenAI-style chat/completions endpoints (Google AI Studio, OpenRouter).

    Mirrors the `client.chat.completions.create(...)` shape of InferenceClient so backends are
    interchangeable; HTTP errors are raised as httpx.HTTPStatusError, which carries `.response`.
    """

    def __init__(self, url: str, api_key: str, headers: dict | None = None, timeout: float = 25):
        self.url = url
        self.headers = {"Authorization": f"Bearer {api_key}", **(headers or {})}
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model: str, messages: list, extra_body: dict | None = None, **params):
        payload = {"model": model, "messages": messages, **params, **(extra_body or {})}
        response = httpx.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        choices = response.json().get("choices")
        if not choices:
            raise ValueError("reply had no choices")
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@dataclass
class Backend:
    name: str
    key_env: str
    client: Any
    models: list[str]
    extra_for: Callable[[str], dict] = field(default=lambda model: {})


def _google_extra(model: str) -> dict:
    # Gemini 2.5 Flash "thinks" by default and thinking can eat the token budget; Gemma doesn't think.
    if model.startswith("gemini-2.5-flash"):
        return {"extra_body": {"reasoning_effort": "none"}}
    return {}


def _backends() -> list[Backend]:
    """Configured backends, most reliable free one first."""
    backends = []
    if key := os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        backends.append(Backend(
            "Google AI Studio", "GEMINI_API_KEY", OpenAICompatible(GOOGLE_URL, key),
            _split(os.getenv("GOOGLE_MODELS", DEFAULT_GOOGLE_MODELS)), _google_extra,
        ))
    if key := os.getenv("OPENROUTER_API_KEY"):
        backends.append(Backend(
            "OpenRouter", "OPENROUTER_API_KEY", OpenAICompatible(OPENROUTER_URL, key, {"X-Title": "Waste Guide"}),
            _split(os.getenv("OPENROUTER_MODELS", DEFAULT_OPENROUTER_MODELS)),
            # Reasoning tokens count against max_tokens; some free models reason at high effort by default.
            lambda model: {"extra_body": {"reasoning": {"effort": "none"}}},
        ))
    if token := os.getenv("HF_TOKEN"):
        backends.append(Backend(
            "Hugging Face", "HF_TOKEN",
            InferenceClient(provider=os.getenv("HF_PROVIDER", "auto"), api_key=token, timeout=25),
            _split(os.getenv("VLM_MODELS", DEFAULT_MODELS)),
        ))
    return backends


def _status(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _server_detail(exc: Exception) -> str:
    """The provider's own explanation (OpenRouter error.message + metadata.raw, Google error.message)."""
    try:
        body = exc.response.json()
        if isinstance(body, list) and body:  # Google wraps errors in a list
            body = body[0]
        err = body.get("error")
        if isinstance(err, str):
            return err[:300]
        meta = err.get("metadata") or {}
        parts = [err.get("message"), meta.get("raw") if isinstance(meta.get("raw"), str) else None]
        return " - ".join(p for p in parts if p)[:300]
    except Exception:
        return str(getattr(exc, "server_message", None) or exc)[:300]


def _bad_key(detail: str) -> bool:
    d = detail.lower()  # Google answers a bad key with HTTP 400 and one of these messages
    return any(k in d for k in ("api key not valid", "api_key_invalid", "valid api key"))


def _is_account_problem(status: int | None, detail: str) -> bool:
    """Errors where every other model on the same backend would fail the same way."""
    d = detail.lower()
    return (
        status in ACCOUNT_ERRORS
        or _bad_key(detail)
        or any(k in d for k in ("per-day", "per-min", "per day", "per minute"))  # OpenRouter account quota
    )


def _friendly_error(exc: Exception, backend: Backend, detail: str = "") -> str:
    status, d, name = _status(exc), detail.lower(), backend.name
    if status == 401 or _bad_key(detail):
        return f"The {name} key is invalid. Check {backend.key_env}."
    if status == 402:
        if name == "Hugging Face":
            return ("Your Hugging Face account has no inference credits (free accounts get none). "
                    "Add credits at huggingface.co/settings/billing, or set GEMINI_API_KEY to use Google's free tier.")
        return f"{name} needs credits for this model."
    if status == 403:
        if name == "Hugging Face":
            return "The Hugging Face token lacks permission. Enable 'Make calls to Inference Providers' on the token."
        return f"{name} refused the request. Check that {backend.key_env} is allowed to use this API."
    if status == 429:
        if name == "OpenRouter" and "day" in d and _is_account_problem(status, detail):
            return "Today's free scans on OpenRouter are used up (50 a day; 1000 after a one-time $10 top-up). Try again tomorrow."
        if name == "OpenRouter" and _is_account_problem(status, detail):
            return "Too many scans in the last minute. Wait a minute and try again."
        if name == "Google AI Studio":
            return "Your free Google AI Studio quota is used up for now. Wait a minute (or until tomorrow) and try again."
        return f"The free vision models on {name} are busy right now. Please try again in a minute."
    return f"The vision model on {name} could not be reached ({status or type(exc).__name__})."


def _mock_identify(note: str) -> Identification:
    if note.strip():
        item_type = keyword_guess(note)
        conditions = [c for c in map(normalize_condition, note.split()) if c != "unknown"]
        return Identification(
            item_name=ITEM_TYPES[item_type]["label"], item_type=item_type,
            condition=conditions[0] if conditions else "unknown",
            confidence=0.9 if item_type != "other" else 0.2, reason="Mock mode: matched your description.", model="mock",
        )
    return Identification(
        item_name="Plastic bottle", item_type="plastic_bottle", material="plastic", condition="clean",
        confidence=0.55, alternatives=["glass_bottle_or_jar", "battery"],
        reason="Mock mode: no model was called.", model="mock",
    )


def identify(image: bytes | None, mime: str = "image/jpeg", note: str = "") -> Identification:
    if os.getenv("MOCK_VLM") == "1":
        return _mock_identify(note)

    backends = _backends()
    if not backends:
        raise VLMError("No vision model is configured. Set GEMINI_API_KEY (free), OPENROUTER_API_KEY or HF_TOKEN.")

    content: list[dict] = []
    if image:
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    content.append({"type": "text", "text": build_prompt(note, has_image=bool(image))})
    messages = [{"role": "user", "content": content}]

    deadline = time.monotonic() + TIME_BUDGET_SECONDS
    errors: list[tuple[bool, str]] = []  # (account-level?, message)
    for backend in backends:
        for model in backend.models:
            status, detail = None, ""
            for _attempt in range(2):
                if time.monotonic() > deadline:
                    raise VLMError("The vision model is taking too long right now. Please try again.")
                try:
                    reply = backend.client.chat.completions.create(
                        model=model, messages=messages, max_tokens=MAX_TOKENS, temperature=0.1,
                        **backend.extra_for(model),
                    )
                    return parse_identification(reply.choices[0].message.content, model=model)
                except ValueError:  # includes JSONDecodeError and empty replies: retry the same model once
                    errors.append((False, f"{model} returned a reply that was not valid JSON."))
                    continue
                except Exception as exc:  # provider error: move on to the next model
                    status, detail = _status(exc), _server_detail(exc)
                    print(json.dumps({"log": "vlm_error", "backend": backend.name, "model": model,
                                      "status": status, "detail": detail}), flush=True)
                    errors.append((_is_account_problem(status, detail), _friendly_error(exc, backend, detail)))
                    break
            if _is_account_problem(status, detail):
                break  # key, billing or account quota problem: this backend's other models will fail too
    account = [msg for is_account, msg in errors if is_account]
    raise VLMError((account or [msg for _, msg in errors] or ["No vision models configured."])[0])
