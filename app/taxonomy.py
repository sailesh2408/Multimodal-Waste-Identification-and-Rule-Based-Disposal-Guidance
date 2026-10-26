"""Controlled vocabulary shared by the VLM prompt, the rules table and the UI."""
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_raw = json.loads((DATA_DIR / "taxonomy.json").read_text(encoding="utf-8"))

ITEM_TYPES: dict[str, dict] = _raw["item_types"]
MATERIALS: list[str] = _raw["materials"]
CONDITIONS: dict[str, str] = _raw["conditions"]
STREAMS: dict[str, dict] = _raw["streams"]

REGIONS = {
    "IN-KL": "Kerala",
    "IN": "Rest of India",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_item_type(value: str) -> str:
    """Map a model's answer onto a known item_type key, else 'other'."""
    slug = _slug(value)
    if slug in ITEM_TYPES:
        return slug
    for key, info in ITEM_TYPES.items():
        if slug == _slug(info["label"]):
            return key
    return "other"


def normalize_material(value: str) -> str:
    slug = _slug(value)
    return slug if slug in MATERIALS else "unknown"


def normalize_condition(value: str) -> str:
    slug = _slug(value)
    if slug in CONDITIONS:
        return slug
    aliases = {
        "dirty": "food_soiled", "soiled": "food_soiled", "oily": "greasy",
        "broken": "damaged", "leaking": "damaged", "swollen": "damaged", "cracked": "damaged",
        "good": "clean", "new": "clean", "empty": "clean",
    }
    return aliases.get(slug, "unknown")


def keyword_guess(text: str) -> str:
    """Crude keyword match used only by the offline mock identifier."""
    words = set(_slug(text).split("_"))
    best, best_hits = "other", 0
    for key, info in ITEM_TYPES.items():
        if key == "other":
            continue
        vocab = set(_slug(info["label"] + " " + info["hint"]).split("_")) | set(key.split("_"))
        vocab -= {"and", "or", "with", "a", "the", "of", "small", "old"}
        hits = len(words & vocab)
        if hits > best_hits:
            best, best_hits = key, hits
    return best
