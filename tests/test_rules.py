import pytest

from app.rules import RuleBook
from app.taxonomy import ITEM_TYPES

rb = RuleBook.load()


def test_rules_table_is_healthy():
    assert rb.lint() == []


@pytest.mark.parametrize("item_type", [k for k in ITEM_TYPES if k != "other"])
def test_every_item_type_has_its_own_rule(item_type):
    assert rb.match(item_type).rule_id != "R000"


@pytest.mark.parametrize("item_type,material,condition,expected_rule,stream", [
    ("battery", "metal", "clean", "R001", "hazardous"),
    ("battery", "metal", "damaged", "R002", "hazardous"),          # condition-specific beats generic
    ("food_packaging", "cardboard", "greasy", "R191", "wet"),      # greasy pizza box
    ("food_packaging", "cardboard", "clean", "R190", "dry"),
    ("paper_cup", "paper", "food_soiled", "R200", "dry"),          # item rule beats generic soiled-paper rule
    ("clothing_textile", "textile", "clean", "R241", "reuse"),     # waste hierarchy: reuse first
    ("other", "electronic", "unknown", "R900", "ewaste"),          # unknown item, but electronic material
    ("other", "plastic", "unknown", "R910", "dry"),
    ("other", "unknown", "unknown", "R000", "unknown"),            # fallback
])
def test_matching(item_type, material, condition, expected_rule, stream):
    rule = rb.match(item_type, material, condition, "IN-KL")
    assert (rule.rule_id, rule.stream) == (expected_rule, stream)


def test_hazard_override_beats_more_specific_safe_rule():
    # A "plastic bottle" holding pesticide must not be sent to dry waste.
    rule = rb.match("plastic_bottle", "chemical", "unknown", "IN-KL")
    assert rule.hazard and rule.stream == "hazardous"


def test_region_changes_where_to_take():
    kerala = rb.guidance("plastic_bottle", "plastic", "clean", "IN-KL")["where_to_take"]
    india = rb.guidance("plastic_bottle", "plastic", "clean", "IN")["where_to_take"]
    assert "Haritha Karma Sena" in kerala
    assert "Haritha Karma Sena" not in india


def test_rule_specific_where_to_take_overrides_region():
    g = rb.guidance("medicine", "unknown", "expired", "IN-KL")
    assert "pharmacy" in g["where_to_take"]
    assert g["hazard"] is True
