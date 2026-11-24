"""Direction Controls for H3 Open Director: each choice becomes one plain sentence, with no rules attached.

The Direction Controls node writes `Label: value` lines (H3 Transfer & Edit Director reads those with its own
rules). For H3 Open Director each line is said plainly ("It is set in the 2000s.") and the LLM works out what
that implies. Random picks are drawn here from the node's seed; the Video Task line (a Transfer & Edit setting)
is dropped.
"""
from __future__ import annotations

import random
import re

from . import direction_controls_fix as dc

HEADER = "DIRECTION CONTROLS:"
POOLS = {
    "Genre": dc.GENRE_CHOICES, "Situation": dc.SITUATIONS, "Motion style": dc.MOTION_CHOICES,
    "Visual look": dc.LOOK_CHOICES, "Outfit style": dc.OUTFIT_STYLES,
    "Era": [era for era in dc.ERA_CHOICES if era != "Auto"],
}
ERA_WORDS = {"Present day": "the present day", "Medieval": "medieval times", "Renaissance": "the Renaissance",
             "Victorian": "the Victorian era", "Stone Age": "the Stone Age", "Feudal Japan": "feudal Japan"}
FROM = {  # what Keep Reference / Keep Guide / Create New mean, in a few words
    "Keep Reference": "as in each character's reference picture",
    "Keep Guide": "as in the guide image or video",
    "Create New": "newly designed",
    "Creative Adaptation": "freely adapted",
    "Guide Garments + Reference Style": "the guide performer's garments, in the style of each character's reference",
}
THING = {"Wardrobe": "Clothes", "Hair & Makeup": "Hair and makeup", "Environment": "The place",
         "Body proportions": "Body build"}


def _era(value):
    if value in ERA_WORDS:
        return ERA_WORDS[value]
    return f"the {value}" if value[:1].isdigit() else value


def _sentence(label, value):
    if label == "Genre":
        return f"Genre: {value}."
    if label == "Situation":
        return f"The situation: {value}."
    if label == "Era":
        return f"It is set in {_era(value)}."
    if label == "Motion style":
        return f"Movement style: {value}."
    if label == "Visual look":
        return f"Visual look: {value}."
    if label == "Outfit style":
        return f"The characters dress in a {value} style."
    if label in THING:
        return f"{THING[label]}: {FROM.get(value, value)}."
    if label == "Dialogue":
        return "Nobody speaks." if value.lower() == "no dialogue" else f"Dialogue in {value}."
    if label == "Dialogue density":
        return f"Dialogue amount: {value.lower()}."
    if label == "Coverage" and value.lower().startswith("single take"):
        return "One continuous shot, no cuts."
    if label == "Adults only":
        return "Every character is an adult."
    if label == "Palette":
        return ""  # implied by the Visual look
    return f"{label}: {value}."


def expand(direction_context, seed=0):
    """Return (text for the LLM, list of 'Label: value' drawn at random). Empty context gives ('', [])."""
    lines = [line.strip() for line in str(direction_context or "").splitlines() if line.strip()]
    if not lines:
        return "", []
    random_labels, kept = [], []
    for line in lines:
        if line.lower().startswith("random pick:"):
            random_labels += [part.strip() for part in line.split(":", 1)[1].split(",") if part.strip()]
        elif not re.match(r"(?i)video task\s*:", line):
            kept.append(line)
    drawn = []
    for label in dict.fromkeys(random_labels):
        pool = POOLS.get(label)
        if pool:
            value = random.Random(f"{int(seed or 0)}:{label}").choice(pool)
            kept.append(f"{label}: {value}")
            drawn.append(f"{label}: {value}")
    out = []
    for line in kept:
        label, _, value = line.partition(":")
        label, value = label.strip(), value.strip()
        if label == "Adults only":
            value = value or "yes"
        text = _sentence(label, value) if value else line
        if text:
            out.append("- " + text)
    return (HEADER + "\n" + "\n".join(out), drawn) if out else ("", drawn)
