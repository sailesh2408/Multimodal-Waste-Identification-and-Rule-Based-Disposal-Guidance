"""Stage 1: identify the waste item with an open vision-language model.

The model only names the item and picks keys from the closed taxonomy.
It never decides the disposal method; that comes from the rules table.

Backends, tried in order (whichever are configured):
  1. OpenRouter free tier  (OPENROUTER_API_KEY) - Gemma 4, no cost, rate limited
  2. Hugging Face Inference Providers (HF_TOKEN) - same Gemma 4, needs HF credits
"""
import base64
import json
import os
import re
import time

from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field

from .taxonomy import (
    CONDITIONS, ITEM_TYPES, MATERIALS,
    keyword_guess, normalize_condition, normalize_item_type, normalize_material,
)

DEFAULT_MODELS = "google/gemma-4-26B-A4B-it,Qwen/Qwen3-VL-8B-Instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODELS = "google/gemma-4-26b-a4b-it:free,google/gemma-4-31b-it:free"
TIME_BUDGET_SECONDS = 35  # stop starting new attempts after this; Vercel's function limit is 60 s
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


def _backends() -> list[tuple[str, InferenceClient, list[str]]]:
    """Configured backends as (name, client, models), free one first."""
    backends = []
    if key := os.getenv("OPENROUTER_API_KEY"):
        client = InferenceClient(base_url=OPENROUTER_URL, api_key=key, timeout=25, headers={"X-Title": "Waste Guide"})
        backends.append(("OpenRouter", client, _split(os.getenv("OPENROUTER_MODELS", DEFAULT_OPENROUTER_MODELS))))
    if token := os.getenv("HF_TOKEN"):
        client = InferenceClient(provider=os.getenv("HF_PROVIDER", "auto"), api_key=token, timeout=25)
        backends.append(("Hugging Face", client, _split(os.getenv("VLM_MODELS", DEFAULT_MODELS))))
    return backends


def _status(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _friendly_error(exc: Exception, backend: str) -> str:
    status = _status(exc)
    key_name = "OPENROUTER_API_KEY" if backend == "OpenRouter" else "HF_TOKEN"
    if status == 401:
        return f"The {backend} key is invalid. Check {key_name}."
    if status == 402:
        if backend == "OpenRouter":
            return "OpenRouter needs credits for this model. Use a ':free' model in OPENROUTER_MODELS."
        return ("Your Hugging Face account has no inference credits (free accounts get none). "
                "Add credits at huggingface.co/settings/billing, or set OPENROUTER_API_KEY to use the free tier.")
    if status == 403:
        if backend == "OpenRouter":
            return "OpenRouter refused the request. Check the key's limits and your privacy settings for free models."
        return "The Hugging Face token lacks permission. Enable 'Make calls to Inference Providers' on the token."
    if status == 429:
        if backend == "OpenRouter":
            return "The free tier's limit is reached (20 per minute, 50 per day). Please try again later."
        return f"Too many requests to {backend}. Wait a moment and try again."
    return f"The vision model on {backend} could not be reached ({type(exc).__name__})."


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
        raise VLMError("No vision model is configured. Set OPENROUTER_API_KEY (free) or HF_TOKEN.")

    content: list[dict] = []
    if image:
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    content.append({"type": "text", "text": build_prompt(note, has_image=bool(image))})
    messages = [{"role": "user", "content": content}]

    deadline = time.monotonic() + TIME_BUDGET_SECONDS
    errors: list[str] = []
    for backend, client, models in backends:
        for model in models:
            status = None
            for _attempt in range(2):
                if time.monotonic() > deadline:
                    raise VLMError("The vision model is taking too long right now. Please try again.")
                try:
                    reply = client.chat.completions.create(
                        model=model, messages=messages, max_tokens=400, temperature=0.1,
                    )
                    return parse_identification(reply.choices[0].message.content, model=model)
                except ValueError:  # includes JSONDecodeError: retry the same model once
                    errors.append(f"{model} returned a reply that was not valid JSON.")
                    continue
                except Exception as exc:  # provider error: move on to the next model
                    status = _status(exc)
                    errors.append(_friendly_error(exc, backend))
                    break
            if status in ACCOUNT_ERRORS:
                break  # key or billing problem: skip this backend's other models
    raise VLMError(errors[0] if errors else "No vision models configured.")
