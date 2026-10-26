"""Stage 2: deterministic disposal decision from data/rules.csv.

Matching:
  1. A row matches when each of region / item_type / material / condition
     equals the input or is "*".
  2. Hazard override: if any matching row is hazardous, only hazardous rows
     are considered. A wrong "hazardous" answer is cheap; a battery in the
     dry bin is not.
  3. The most specific row wins. Specificity is ranked region > item_type >
     condition > material, so an item-specific rule always beats a generic
     material rule. Ties are broken by the priority column (lower wins).
"""
import csv
from dataclasses import dataclass
from pathlib import Path

from .taxonomy import CONDITIONS, DATA_DIR, ITEM_TYPES, MATERIALS, STREAMS

WILDCARD = "*"
WEIGHTS = {"region": 8, "item_type": 4, "condition": 2, "material": 1}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    region: str
    item_type: str
    material: str
    condition: str
    stream: str
    prep_steps: tuple[str, ...]
    where_to_take: str
    hazard: bool
    hazard_note: str
    priority: int
    source: str

    @property
    def specificity(self) -> int:
        return sum(w for field, w in WEIGHTS.items() if getattr(self, field) != WILDCARD)

    def matches(self, region: str, item_type: str, material: str, condition: str) -> bool:
        return (
            self.region in (WILDCARD, region)
            and self.item_type in (WILDCARD, item_type)
            and self.material in (WILDCARD, material)
            and self.condition in (WILDCARD, condition)
        )


class RuleBook:
    def __init__(self, rules: list[Rule], collection_points: dict[tuple[str, str], str]):
        self.rules = rules
        self.collection_points = collection_points

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR) -> "RuleBook":
        with open(data_dir / "rules.csv", newline="", encoding="utf-8") as f:
            rules = [
                Rule(
                    rule_id=row["rule_id"].strip(),
                    region=row["region"].strip() or WILDCARD,
                    item_type=row["item_type"].strip() or WILDCARD,
                    material=row["material"].strip() or WILDCARD,
                    condition=row["condition"].strip() or WILDCARD,
                    stream=row["stream"].strip(),
                    prep_steps=tuple(s.strip() for s in row["prep_steps"].split("|") if s.strip()),
                    where_to_take=row["where_to_take"].strip(),
                    hazard=row["hazard"].strip().lower() == "yes",
                    hazard_note=row["hazard_note"].strip(),
                    priority=int(row["priority"] or 100),
                    source=row["source"].strip(),
                )
                for row in csv.DictReader(f)
            ]
        with open(data_dir / "collection_points.csv", newline="", encoding="utf-8") as f:
            points = {(r["region"].strip(), r["stream"].strip()): r["where_to_take"].strip() for r in csv.DictReader(f)}
        return cls(rules, points)

    def match(self, item_type: str, material: str = "unknown", condition: str = "unknown", region: str = WILDCARD) -> Rule:
        candidates = [r for r in self.rules if r.matches(region, item_type, material, condition)]
        hazardous = [r for r in candidates if r.hazard]
        pool = hazardous or candidates
        if not pool:
            raise LookupError("rules.csv has no fallback row (*,*,*,*)")
        return max(pool, key=lambda r: (r.specificity, -r.priority))

    def is_hazardous_type(self, item_type: str) -> bool:
        return self.match(item_type).hazard

    def where_to_take(self, rule: Rule, region: str) -> str:
        if rule.where_to_take:
            return rule.where_to_take
        return (
            self.collection_points.get((region, rule.stream))
            or self.collection_points.get((WILDCARD, rule.stream))
            or ""
        )

    def guidance(self, item_type: str, material: str, condition: str, region: str) -> dict:
        rule = self.match(item_type, material, condition, region)
        stream = STREAMS.get(rule.stream, STREAMS["unknown"])
        return {
            "rule_id": rule.rule_id,
            "stream": rule.stream,
            "stream_label": stream["label"],
            "tone": stream["tone"],
            "prep_steps": list(rule.prep_steps),
            "where_to_take": self.where_to_take(rule, region),
            "hazard": rule.hazard,
            "hazard_note": rule.hazard_note,
            "source": rule.source,
        }

    def lint(self) -> list[str]:
        """Return a list of problems with the rules table (empty = healthy)."""
        problems = []
        seen: dict[tuple, str] = {}
        for r in self.rules:
            key = (r.region, r.item_type, r.material, r.condition)
            if key in seen:
                problems.append(f"{r.rule_id} duplicates {seen[key]} for key {key}")
            seen[key] = r.rule_id
            if r.item_type not in (WILDCARD, *ITEM_TYPES):
                problems.append(f"{r.rule_id}: unknown item_type '{r.item_type}'")
            if r.material not in (WILDCARD, *MATERIALS):
                problems.append(f"{r.rule_id}: unknown material '{r.material}'")
            if r.condition not in (WILDCARD, *CONDITIONS):
                problems.append(f"{r.rule_id}: unknown condition '{r.condition}'")
            if r.stream not in STREAMS:
                problems.append(f"{r.rule_id}: unknown stream '{r.stream}'")
            if not r.prep_steps:
                problems.append(f"{r.rule_id}: has no prep steps")
            if not r.source:
                problems.append(f"{r.rule_id}: has no source")
        if (WILDCARD,) * 4 not in seen:
            problems.append("missing fallback row (*,*,*,*)")
        for item in ITEM_TYPES:
            if item != "other" and (WILDCARD, item, WILDCARD, WILDCARD) not in seen:
                problems.append(f"item_type '{item}' has no catch-all rule (*,{item},*,*)")
        for stream in STREAMS:
            if (WILDCARD, stream) not in self.collection_points:
                problems.append(f"collection_points.csv has no default row for stream '{stream}'")
        return problems
