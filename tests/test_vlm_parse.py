import pytest

from app.vlm import build_prompt, parse_identification


def test_parses_fenced_json_and_normalizes():
    reply = """```json
    {"item_name": "AA battery", "item_type": "Battery", "material": "Metal", "condition": "swollen",
     "confidence": 85, "alternatives": ["electronic device", "battery", "made_up"], "reason": "Cylindrical cell"}
    ```"""
    ident = parse_identification(reply, model="m")
    assert ident.item_type == "battery"
    assert ident.material == "metal"
    assert ident.condition == "damaged"
    assert ident.confidence == 0.85
    assert ident.alternatives == ["electronic_device"]  # duplicates of primary and unknowns dropped
    assert ident.model == "m"


def test_strips_thinking_and_surrounding_text():
    reply = '<think>maybe {not this}</think>Sure! {"item_name": "Pizza box", "item_type": "food_packaging", "material": "cardboard", "condition": "greasy", "confidence": 0.9} Hope that helps.'
    ident = parse_identification(reply)
    assert (ident.item_type, ident.condition) == ("food_packaging", "greasy")


def test_unknown_values_fall_back_safely():
    ident = parse_identification('{"item_type": "spaceship", "material": "unobtainium", "condition": "shiny", "confidence": "high"}')
    assert (ident.item_type, ident.material, ident.condition, ident.confidence) == ("other", "unknown", "unknown", 0.0)


def test_rejects_reply_without_json():
    with pytest.raises(ValueError):
        parse_identification("I think this is a battery.")


def test_prompt_lists_every_item_type_and_user_note():
    prompt = build_prompt("expired tablets")
    assert "cfl_or_tube_light" in prompt and "sharps" in prompt
    assert "expired tablets" in prompt
