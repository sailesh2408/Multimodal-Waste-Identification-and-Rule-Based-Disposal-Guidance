"""FastAPI app: serves the mobile site and the identify -> confirm -> rules pipeline."""
import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ON_VERCEL = bool(os.getenv("VERCEL"))
if ON_VERCEL:
    os.environ.setdefault("HF_HOME", "/tmp/hf")  # Vercel's filesystem is read-only except /tmp

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from .rules import RuleBook  # noqa: E402
from .taxonomy import CONDITIONS, ITEM_TYPES, REGIONS, normalize_condition, normalize_item_type, normalize_material  # noqa: E402
from .vlm import Identification, VLMError, identify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # Vercel rejects request bodies over 4.5 MB
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
DEFAULT_REGION = os.getenv("DEFAULT_REGION", "IN-KL")

rulebook = RuleBook.load()
app = FastAPI(title="Waste Guide")


def _log(name: str, record: dict) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "log": name.removesuffix(".jsonl"), **record}
    if ON_VERCEL and not os.getenv("LOG_DIR"):
        print(json.dumps(record, ensure_ascii=False), flush=True)  # shows up in Vercel's Runtime Logs
        return
    log_dir = Path(os.getenv("LOG_DIR", ROOT / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _region(value: str) -> str:
    return value if value in REGIONS else DEFAULT_REGION


def confirmation(ident: Identification) -> dict:
    """Confidence gate: decide whether to ask the user to confirm the item."""
    primary_hazard = rulebook.is_hazardous_type(ident.item_type) if ident.item_type != "other" else False
    hazardous_alt = next((a for a in ident.alternatives if rulebook.is_hazardous_type(a)), None)

    reason = ""
    if ident.item_type == "other":
        reason = "We couldn't recognise this item. Which is closest?"
    elif hazardous_alt and not primary_hazard:
        reason = f"This might be a {ITEM_TYPES[hazardous_alt]['label'].lower()}, which needs special handling. Please confirm."
    elif ident.confidence < CONFIDENCE_THRESHOLD:
        reason = "We're not completely sure. Is it one of these?"

    options = [t for t in [ident.item_type, *ident.alternatives] if t != "other"]
    return {
        "needed": bool(reason),
        "reason": reason,
        "candidates": [
            {"item_type": t, "label": ITEM_TYPES[t]["label"], "hazardous": rulebook.is_hazardous_type(t)}
            for t in dict.fromkeys(options)
        ][:4],
    }


@app.get("/api/config")
def config() -> dict:
    return {
        "item_types": [{"key": k, "label": v["label"], "group": v["group"]} for k, v in ITEM_TYPES.items()],
        "conditions": [{"key": k, "label": v} for k, v in CONDITIONS.items()],
        "regions": [{"key": k, "label": v} for k, v in REGIONS.items()],
        "default_region": DEFAULT_REGION,
        "mock": os.getenv("MOCK_VLM") == "1",
    }


@app.post("/api/identify")
def identify_item(
    image: UploadFile | None = File(default=None),
    note: str = Form(default=""),
    region: str = Form(default=""),
) -> dict:
    image_bytes, mime = None, "image/jpeg"
    if image is not None and image.filename:
        mime = image.content_type or "image/jpeg"
        if not mime.startswith("image/"):
            raise HTTPException(400, "Please upload an image file.")
        image_bytes = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "That photo is too large. Please use one under 4 MB.")
    note = note.strip()[:300]
    if not image_bytes and not note:
        raise HTTPException(400, "Take a photo or describe the item first.")

    try:
        ident = identify(image_bytes, mime, note)
    except VLMError as exc:
        raise HTTPException(502, str(exc)) from exc

    region = _region(region)
    request_id = uuid.uuid4().hex[:12]
    guidance = rulebook.guidance(ident.item_type, ident.material, ident.condition, region)
    confirm = confirmation(ident)
    _log("requests.jsonl", {
        "request_id": request_id, "region": region, "had_image": bool(image_bytes), "note": note,
        "identification": ident.model_dump(), "rule_id": guidance["rule_id"], "confirm_needed": confirm["needed"],
    })
    return {"request_id": request_id, "identification": ident.model_dump(), "guidance": guidance, "confirm": confirm}


class GuidanceRequest(BaseModel):
    item_type: str
    material: str = "unknown"
    condition: str = "unknown"
    region: str = ""
    request_id: str = ""


@app.post("/api/guidance")
def guidance(req: GuidanceRequest) -> dict:
    """Re-run only the rules stage after the user corrects the item or condition."""
    item_type = normalize_item_type(req.item_type)
    material = normalize_material(req.material)
    condition = normalize_condition(req.condition)
    region = _region(req.region)
    result = rulebook.guidance(item_type, material, condition, region)
    _log("corrections.jsonl", {
        "request_id": req.request_id, "item_type": item_type, "material": material,
        "condition": condition, "region": region, "rule_id": result["rule_id"],
    })
    return {"item_type": item_type, "label": ITEM_TYPES[item_type]["label"], "condition": condition, "guidance": result}


class Feedback(BaseModel):
    request_id: str
    helpful: bool
    item_type: str = ""
    condition: str = ""
    rule_id: str = ""
    comment: str = ""


@app.post("/api/feedback")
def feedback(fb: Feedback) -> dict:
    _log("feedback.jsonl", {**fb.model_dump(), "comment": fb.comment[:500]})
    return {"ok": True}


# API routes always take priority over frontend files. On Vercel these files are served from the CDN.
app.frontend("/", directory=ROOT / "static")
