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

HEADER = "DIRECTION CONTROLS (apply every one of these):"
POOLS = {
    "Genre": dc.GENRE_CHOICES, "Situation": dc.SITUATIONS, "Motion style": dc.MOTION_CHOICES,
    "Visual look": dc.LOOK_CHOICES, "Outfit style": dc.OUTFIT_STYLES,
    "Era": [era for era in dc.ERA_CHOICES if era != "Auto"],
}
THING = {"Wardrobe": "Clothes", "Hair & Makeup": "Hair and makeup", "Environment": "The place",
         "Body proportions": "Body build"}
ERA_WORDS = {"Present day": "the present day", "Medieval": "medieval times", "Renaissance": "the Renaissance",
             "Victorian": "the Victorian era", "Stone Age": "the Stone Age", "Feudal Japan": "feudal Japan"}
KEEP = {  # Keep Reference / Keep Guide / Guide Garments, per control
    ("Wardrobe", "Keep Reference"): "Clothes: each character wears exactly the outfit of their reference picture; "
                                    "describe it.",
    ("Wardrobe", "Keep Guide"): "Clothes: each character wears exactly the outfit seen in the guide image or video; "
                                "describe it.",
    ("Wardrobe", "Guide Garments + Reference Style"): "Clothes: the same garments as the person in the guide, in the "
                                                      "colours and fabrics of each character's reference picture.",
    ("Hair & Makeup", "Keep Reference"): "Hair and makeup: exactly as in each character's reference picture.",
    ("Hair & Makeup", "Keep Guide"): "Hair and makeup: exactly as on the person in the guide image or video.",
    ("Environment", "Keep Reference"): "The place: exactly the place shown in the reference picture.",
    ("Environment", "Keep Guide"): "The place: exactly the place shown in the guide image or video.",
    ("Body proportions", "Keep Reference"): "Body build: exactly as in each character's reference picture.",
    ("Body proportions", "Keep Guide"): "Body build: exactly as the person in the guide image or video; faces stay "
                                        "the characters' own.",
}
NEW = {  # Create New / Creative Adaptation: the pictures are not the model for this part
    "Wardrobe": "Clothes: new outfits, not the ones in the reference pictures or guide (those show only who the "
                "people are); describe what each character wears.",
    "Hair & Makeup": "Hair and makeup: a new look, not the one in the reference pictures or guide; describe it.",
    "Environment": "The place: a new setting, not the one in the reference pictures or guide; describe it.",
    "Body proportions": "Body build: may differ from the reference pictures; faces and ages stay the same.",
}
DENSITY = {"sparse": "Only a few short spoken lines in the whole piece.",
           "balanced": "Each main character speaks once or twice.",
           "dense": "The characters talk almost all the time."}


_GARMENTS = ("describe each garment by what it is, its cut, fabric and colour, without using the style's name "
             "as a label.")


def _era(value):
    if value in ERA_WORDS:
        return ERA_WORDS[value]
    return f"the {value}" if value[:1].isdigit() else value


def _sentence(label, value, chosen):
    if label == "Genre":
        return f"Genre: {value}. The story, tone and ending are those of this genre."
    if label == "Situation":
        return f"The story is this situation: {value}. Everything happens within it."
    if label == "Era":
        return (f"Everything is set in {_era(value)}: clothes, hair, places, objects and technology belong to that "
                f"time, whatever time the reference pictures show.")
    if label == "Motion style":
        return f"Movement: {value}. The bodies and the camera move this way throughout."
    if label == "Visual look":
        return f"Visual look: {value}. Camera, colour and grade follow it throughout."
    if label == "Outfit style":
        wardrobe = chosen.get("Wardrobe", "")
        if wardrobe == "Keep Guide":
            return ""  # the guide's clothes are kept as they are
        if wardrobe in ("Keep Reference", "Guide Garments + Reference Style"):
            return (f"Each character's outfit is restyled as {value}, keeping its colours and fabrics; {_GARMENTS}")
        return (f"Every character wears {value} clothing, not the clothes of the reference pictures; {_GARMENTS}")
    if label in THING:
        if value in ("Create New", "Creative Adaptation"):
            return NEW[label]
        return KEEP.get((label, value), f"{THING[label]}: {value}.")
    if label == "Dialogue":
        if value.lower() == "no dialogue":
            return "Nobody speaks at any moment."
        return f"All spoken lines are in {value}."
    if label == "Dialogue density":
        return DENSITY.get(value.lower(), "")
    if label == "Coverage" and value.lower().startswith("single take"):
        return "One continuous shot from start to end, with no cuts."
    if label == "Adults only":
        return "Every character is an adult (18+)."
    if label == "Palette":
        return ""  # part of the Visual look
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
    chosen = {}
    for line in kept:
        label, _, value = line.partition(":")
        chosen[label.strip()] = value.strip()
    out = []
    for line in kept:
        label, _, value = line.partition(":")
        label, value = label.strip(), value.strip()
        if label == "Adults only":
            value = value or "yes"
        text = _sentence(label, value, chosen) if value else line
        if text:
            out.append("- " + text)
    return (HEADER + "\n" + "\n".join(out), drawn) if out else ("", drawn)
