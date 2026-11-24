"""Direction Controls for H3 Open Director: each chosen value travels with a short note of what it means.

The Direction Controls node writes plain `Label: value` lines (H3 Transfer & Edit Director reads those lines
with its own rules). H3 Open Director has no rules of its own, so this module turns the same lines into notes
the LLM can use next to the request: what the control decides plus what the chosen value means. Random picks
are drawn here from the node's seed, and the Video Task line (a Transfer & Edit setting) is dropped.
"""
from __future__ import annotations

import random
import re

from . import direction_controls_fix as dc
from .story_director import MOTION_STYLES, VISUAL_LOOKS

try:  # the cue tables live on the Transfer & Edit Director; the notes still work without them
    from .story_director_fix import H3CompactMultimodalEditDirectorFix as _Director
    GENRE_CUES = dict(getattr(_Director, "GENRE_MECHANISMS", {}) or {})
    SITUATION_CUES = dict(getattr(_Director, "SITUATION_CUES", {}) or {})
    OUTFIT_CUES = dict(getattr(_Director, "OUTFIT_CUES", {}) or {})
    ERA_CUES = {**dict(getattr(_Director, "HISTORIC_CUES", {}) or {}), **dict(getattr(_Director, "ERA_CUES", {}) or {})}
except Exception:  # noqa: BLE001
    GENRE_CUES, SITUATION_CUES, OUTFIT_CUES, ERA_CUES = {}, {}, {}, {}

HEADER = "DIRECTION CONTROLS (chosen in the Direction Controls node; where they disagree with the request, the request wins):"

ABOUT = {
    "Genre": "the tone, world and kind of payoff of the piece; it shapes the concrete situation, not only the mood",
    "Situation": "the premise, what is happening and where; the Genre gives its tone",
    "Motion style": "how the bodies and the camera move: cadence, energy and camera behaviour",
    "Visual look": "the capture and colour treatment (camera, film, grade) and the palette; it never changes the era, "
                   "clothes or sets",
    "Outfit style": "the clothing style of the characters",
    "Era": "the period the world lives in: mainstream fashion as people really wore it, sets, vehicles, technology "
           "and objects of that time, no costume caricature",
}
SOURCES = {  # Wardrobe / Hair & Makeup / Environment / Body proportions
    "Keep Reference": "exactly as in each character's reference picture",
    "Keep Guide": "exactly as in the guide image or guide video",
    "Create New": "newly designed for this piece, without changing who the characters are or what they do",
    "Creative Adaptation": "coherent new build and proportions, keeping face, age and performance",
    "Guide Garments + Reference Style": "the same garments as the guide performer (pieces, cut, layers, how they move), "
                                        "restyled in the colours, fabrics and patterns of each character's reference",
}
WHAT = {
    "Wardrobe": "clothes, shoes and accessories",
    "Hair & Makeup": "hairstyle, makeup, nails and grooming (face, age, tattoos and marks always stay the character's)",
    "Environment": "the place",
    "Body proportions": "build and body proportions (the face always stays the character's)",
}
DENSITY = {
    "Sparse": "a few short lines in the whole piece",
    "Balanced": "each lead speaks once or twice",
    "Dense": "a continuous exchange of lines",
}
POOLS = {
    "Genre": dc.GENRE_CHOICES, "Situation": dc.SITUATIONS, "Motion style": dc.MOTION_CHOICES,
    "Visual look": dc.LOOK_CHOICES, "Outfit style": dc.OUTFIT_STYLES,
    "Era": [era for era in dc.ERA_CHOICES if era != "Auto"],
}


def _cue(table, value, limit=260):
    """The first sentence of a table entry (keys are matched without case)."""
    text = table.get(value)
    if text is None:
        lowered = {str(k).lower(): v for k, v in table.items()}
        text = lowered.get(str(value).lower())
    text = re.sub(r"^18\+:\s*", "", str(text or "")).strip()
    if not text:
        return ""
    first = re.split(r"(?<=[.;])\s+", text, maxsplit=1)[0].rstrip(".;")
    first = first[:1].upper() + first[1:]
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + "…"


def _note(label, value):
    if label in ABOUT:
        table = {"Genre": GENRE_CUES, "Situation": SITUATION_CUES, "Motion style": MOTION_STYLES,
                 "Visual look": VISUAL_LOOKS, "Outfit style": OUTFIT_CUES, "Era": ERA_CUES}[label]
        cues = [_cue(table, part.strip()) for part in value.split(" + ")]
        return ABOUT[label] + "".join(f". {value.split(' + ')[i].strip()}: {cue}" if " + " in value else f". {cue}"
                                      for i, cue in enumerate(cues) if cue)
    if label in WHAT:
        return f"{WHAT[label]}: {SOURCES.get(value, value)}"
    if label == "Dialogue":
        if value.lower() == "no dialogue":
            return "nobody speaks; only sound and music"
        return f"spoken lines are in {value}; the rest of the prompt stays in English"
    if label == "Dialogue density":
        return DENSITY.get(value, "")
    if label == "Coverage" and value.lower().startswith("single take"):
        return "one continuous shot with no cuts, because the selected look or camera is a single fixed camera"
    return ""


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
        if label == "Palette" and any(k.startswith("Visual look:") for k in kept):
            continue  # the look's note already says it sets the palette
        note = _note(label, value) if value else ""
        mark = " (drawn at random for this run)" if f"{label}: {value}" in drawn else ""
        out.append(f"- {label}: {value}{mark}" + (f" — {note}" if note else ""))
    return HEADER + "\n" + "\n".join(out), drawn
