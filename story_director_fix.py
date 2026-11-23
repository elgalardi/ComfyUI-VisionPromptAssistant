"""H3 Compact Multimodal Edit Director: the official, decision-making compact director.

It subclasses ``H3CompactMultimodalEditDirector`` (same inputs/outputs, same
execution engine, hold cache and provider routing) and only overrides:

* the rule blocks: condensed (~55% fewer system-prompt tokens), conflict-free,
  with one shared creative-decision toolkit adapted from Shotcaller / Motiondojo
  / Story Bible Builder (lens register, sourced light, acting, geometry, mouth
  lock, no on-screen text, physics);
* ``execute``: direction-control rules, retries and output clean-up.

The base engine stays in story_director.py. Hold cache is isolated (``cls.__name__``).
"""
import contextvars
import inspect
import json
import re
import sys
import textwrap

from comfy_api.latest import io

from .story_director import H3CompactMultimodalEditDirector as _Base


# ComfyUI may run execute on a cloned class, so shared state lives at module level.
_PATCH_STATE = {"tried": False, "fn": None, "failed": (), "report": "patch not tried"}
_STATE_VAR = contextvars.ContextVar("h3_fix_run_state", default=None)
_HOOK_VAR = contextvars.ContextVar("h3_fix_provider_hook", default=None)


class _RunState:
    """Per-execution state held in a ContextVar: overlapping executions never share or overwrite it."""
    _fallback = {"seconds": 10.0, "prev_closing": None}

    def _data(self):
        data = _STATE_VAR.get()
        return data if data is not None else self._fallback

    def reset(self):
        _STATE_VAR.set({"seconds": 10.0, "prev_closing": None})

    def __getitem__(self, key):
        return self._data()[key]

    def __setitem__(self, key, value):
        self._data()[key] = value

    def __contains__(self, key):
        return key in self._data()

    def get(self, key, default=None):
        return self._data().get(key, default)

    def update(self, *args, **kwargs):
        self._data().update(*args, **kwargs)

    def pop(self, key, *default):
        return self._data().pop(key, *default)

    def setdefault(self, key, default=None):
        return self._data().setdefault(key, default)


_CLIP = _RunState()


def _install_provider_dispatch():
    """Route the original engine's provider function through one permanent dispatcher; each execution adds its
    own hook through a ContextVar instead of swapping the global function."""
    from . import story_director as _sd
    current = _sd._external_llm_request
    if getattr(current, "_h3_fix_dispatch", False):
        return current

    def dispatch(llm_model, payload):
        hook = _HOOK_VAR.get()
        return hook(llm_model, payload) if hook is not None else dispatch._original(llm_model, payload)
    dispatch._h3_fix_dispatch = True
    dispatch._original = current
    _sd._external_llm_request = dispatch
    return dispatch


class _RelaxedJSONLLM:
    """Wraps the connected LLM for a retry after a provider aborted strict JSON-schema generation
    (e.g. 'Model output became abnormal while generating a JSON response for response_format'):
    the schema travels as text in the system message and the reply is parsed leniently."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def h3_chat_completion(self, payload):
        from . import story_director as _sd
        payload = dict(payload)
        response_format = payload.pop("response_format", None) or {}
        schema = (response_format.get("json_schema") or {}).get("schema") if isinstance(response_format, dict) else None
        messages = [dict(message) for message in payload.get("messages", [])]
        if schema and messages and isinstance(messages[0].get("content"), str):
            messages[0]["content"] += (
                "\n\nReturn only one valid JSON object (no Markdown, no commentary) that follows this JSON Schema "
                "exactly, with the exact field names:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
        payload["messages"] = messages
        if isinstance(payload.get("provider"), dict):
            payload["provider"] = {k: v for k, v in payload["provider"].items() if k != "require_parameters"}
        request = getattr(_sd._external_llm_request, "_original", _sd._external_llm_request)
        return request(self._inner, payload)  # already counted and prepared once by the outer call


class _DrivenCapture(Exception):
    """Internal: stops the original Driven analysis right before its API call to reuse its payload."""


class H3CompactMultimodalEditDirectorFix(_Base):
    VERSION_TAG = "v147"

    # ------------------------------------------------------------------ rules
    DEFAULT_SYSTEM_PROMPT = """
You are a precise multimodal edit director for MiniMax H3. Inspect the actual pixels of every
connected source before writing. Return English only as one JSON object with exactly four keys:
`edit_type`, `source_roles`, `visual_evidence`, `edit_prompt`. `source_roles` binds each used tag to
one role; `visual_evidence` lists only visible traits the edit needs; `edit_prompt` is the final
compact instruction (the node adds the mode header).

Tags: <Picture 1>-<Picture 4> are literal visual references; <Video 1>/<Video 2> are source plates
or temporal references; <Driven Image 1>/<Driven Video 1> supply activity, scenario and staging to
adapt onto the assigned cast. Outside assigned or Driven roles never transfer a person, object,
setting, wardrobe, action or style.

Name every source used, state the exact change, describe the concrete visible traits that transfer
and preserve everything unrequested. Prefer specific visual language over "same person", "matching
outfit" or "use the reference". Explicit requests beat source evidence.
- Background: keep foreground and performance; match perspective, scale, parallax, occlusion,
  reflections, contact shadows, light direction/temperature, depth of field.
- Identity: transfer observed face, hair and assigned traits onto one existing body; keep pose,
  anatomy, action, timing, gaze and occlusions.
- Wardrobe: garment type, cut, fit, material, colour, pattern, accessories on the same wearer.
- Object: keep placement, grip/contact, scale, shadows, reflections, occlusion order.
- Pose/action: keep identity and scene; match body configuration, direction, weight, rhythm.
- Style/relight: transfer only the requested palette, contrast, texture, light, lens or motion.
Video: the source plate is the temporal blueprint (chronology, camera, cuts, timing, audio sync)
unless changed. Image: keep composition and geometry unless changed. No invented shots, story,
dialogue, negative prompts or production notes.
""".strip()

    DEEP_EDIT_RULES = """

EDIT CONTRACT
1. Cumulative change set: every explicit modification, including later clauses and corrections, is
   part of one set. "Only the background" limits that operation, not a later request to change the
   character, wardrobe, action, camera or look. Preserve only what remains after subtracting the
   set; never lock an attribute the user changes. Rewrite any sentence like "change nothing else"
   whose scope would contradict a requested change, naming the exceptions instead.
2. Source ledger: bind each tag to one function (locked plate, identity, body, wardrobe, object,
   environment, look, pose/action, camera, choreography, temporal reference). Unused sources stay
   unused. Driven sources authorise scenario transfer, not identity replacement.
3. Change/preserve pairs: target instance, replacement source, scope, traits to transfer/retain,
   physical integration. Resolve pronouns from visible evidence.
4. Lock the plate by default; unlock composition, camera, action or timing only on request.
5. One coherent result: assign face, hair, physique, wardrobe, motion, environment, look separately
   and recombine on one subject and one world; never average references.
6. Describe evidence concretely: facial geometry, hair, silhouette/fabric/fit, layout/materials/
   practical lights, capture medium/lens/contrast/palette — not adjectives.
Operation notes: background = 3D moving plate with matte edges, parallax, occlusion, spill, contact
shadows; character = replace the person (face, head, hair, skin, physique) on one body through every
angle and occlusion; add/remove = count, placement, ownership, no duplicates; interaction = initiator,
recipient, contact points, direction, weight transfer, reaction; camera change only when requested;
relight = relight subject and environment together, keep identity and material identity.
Audio: keep source audio, lip sync and performance timing unless edited. Positive wording; at most a
short task-specific failure clause. Typical edit_prompt 120–320 words (longer only for multi-operation
edits). Use canonical <Picture N>/<Video N>. Driven inputs are compiled into concrete visual details.
Video plate with several subjects or timed changes: use subject_definitions / summary /
retention_analysis / detailed_description / overall_soundscape / non_diegetic_music with <Subject N>,
retention labels fully_preserved, partially_preserved, attribute_transfer, weak_reference; [Shot 1]
only for real cuts. Simple edits and image edits: one precise paragraph. Never invent an <Audio N>.
""".strip()

    ELABORATE_RULES = """
ELABORATE — ONE AUTHORED PROMPT
Write one developed, production-ready prompt under the same change/preserve contract. Length follows
the screen-time contract below; every detail must control something visible or audible.

DECIDE, DON'T LIST. The user text may be only a seed. For everything it leaves open you MAKE the
choices (never alternatives, "could", "may" or stock filler): the situation and payoff, place and
time, one distinctive camera idea, light sources, performance intentions, sound perspective and the
final changed image. Use the DIRECTOR TOOLKIT. Explicit requests, assigned source traits, locked
camera/performance/audio and named locations stay authoritative; do not replace identities, assigned
wardrobe or required outcomes; no principal character or subplot added only for spectacle (characters
the story needs are welcome). A faithful edit
authorises no new dialogue, stunts, violence, intimacy or supernatural events. An open brief
authorises genre-appropriate events and dialogue. Where request and sources are silent, invent boldly.

Order the prose: who/where (stable traits, positions) → setting with depth planes, materials and
sourced light → chronological micro-beats (initiator, direction, pace, contact, reaction, resulting
state) → shot size, height, angle, lens/focus and one dominant movement → ambience, Foley, dialogue,
music → one short descriptive continuity closer (no command verbs).

DIRECT VISUAL PROSE: describe finished footage, never a request to a model. Do not use Create,
Generate, Show, Depict, Make, Use, Maintain, Preserve, Ensure, Keep, Avoid, "Do not", "should" or
"focus on". Start with the active tagged subject or visible setting; state preservation as fact
("Her white hair and purple beanie remain consistent"). Replace mood words (intense, raw, frantic)
with distance, posture, gaze, hands, pace, contact, weight, breath, reaction. Never "they engage in".
Cuts only where the structure permits them; locked cameras stay locked. Same four-key JSON as base;
`visual_evidence` stays a short audit.
""".strip()

    H3_NATIVE_PROMPT_RULES = """
MINIMAX H3 NATIVE GRAMMAR (official guide)
Plain-text sections, exact headings, no Markdown bold, blank lines between sections and shots:

subject_definitions:
integrated_multimodal_description:
overall_soundscape:
non_diegetic_music:

- subject_definitions: each persistent person/object/environment/style as <Subject N>, bound to its
  source the official way: `<Subject 1> is the woman in <Picture 1>, ...`, then only the stable
  distinguishing traits (identity, wardrobe, object). Labels keep one meaning in every section. No
  subjects: `N/A`.
- integrated_multimodal_description: open with one or two English sentences setting style and
  initial composition, then `[Shot 1]` (never a timestamp). Later shots: `[Shot 2] At 00:SS.mmm, ...`
  with strictly increasing times inside the clip; every shot lasts at least 2 seconds, the first cut
  comes after 2 seconds and the last cut at least 2 seconds before the end. A cut must bring NEW
  information (subject, space, state, viewpoint or time); a small angle or distance change is camera
  motion instead. Keep positions explicit in multi-subject coverage; props and contacts must be
  visible in frame.
- A cut never changes the physical state: every shot after a cut restates, for each visible subject,
  where they are and their level exactly as the previous shot left them (still seated on the sofa,
  still standing at the bar, kneeling, lying), what they hold and any garment already changed. A
  posture changes only on screen inside a shot (she stands up), never across a cut.
- Camera motion is written as natural action with type, amplitude and speed ("pushes in with small
  amplitude at slow speed toward the folded letter"); omit amplitude/speed when medium/normal; no
  stacked camera labels at the end of sentences.
- Dialogue, written once, here: identify the speaker on first appearance with voice details outside
  the tag (pitch, timbre, pace, accent), then `<Subject N> (S1) says: <d>[Language] exact words</d>`.
  Speaker IDs (S1), (S2)... are assigned in the order of actual vocal events, stay stable across shots,
  and are compound for simultaneous speech, e.g. (S1,S2). Characters who never vocalise get no ID.
  Inside <d> only the language tag and the exact words; supplied lines stay verbatim, untranslated.
  While one subject speaks, the others have closed mouths and active listening behaviour: in every shot
  with speech, name who speaks and state the other visible faces' lips closed. `<cutoff>`
  when speech is cut by the end of the clip. A line that continues over a cut is written in two
  parts, one per shot, each with `<scenetrans>` at the connecting point, and the shot says the voice
  "continues seamlessly across the cut". Voiceover as "says in an off-screen voiceover" with the
  visible lips stated closed.
- overall_soundscape: 1–4 sentences of ambience, physical action sounds and non-verbal human sound;
  no dialogue or diegetic music; `N/A` only if total silence was requested.
- non_diegetic_music: 1–3 sentences of instrumentation, tempo, rhythm and dynamic development, never
  its emotional function or abstract mood words; `N/A` when no score. Diegetic music belongs in the shots.
- Visible on-screen text only if the user supplied the exact words, in English double quotes; no
  captions, logos or UI otherwise.
- End on continuing behaviour, a completed result or a purposeful held beat — not a freeze, dead air
  or a repeat of the opening composition. Positive wording; a targeted constraint only for a likely
  failure (duplicate subjects, voice swap, unstable wardrobe, broken contact).
""".strip()

    NARRATIVE_ARC_RULES = _Base.NARRATIVE_ARC_RULES.replace(
        "changing stakes and payoff", "development and payoff").replace(
        "Reserve time INSIDE the last scene for completion",
        "The requested action may happen in any scene, even the first; reserve time INSIDE the last scene for "
        "its consequence, the story's close")
    assert "development and payoff" in NARRATIVE_ARC_RULES and "even the first" in NARRATIVE_ARC_RULES

    OUTPUT_LANGUAGE_RULE = (
        "OUTPUT LANGUAGE (binding): write every field and all prose in English, even when the user writes the "
        "request in Spanish or any other language. Only the spoken words inside <d>...</d> use the dialogue "
        "language (the selected one, otherwise the language of the user's request). Never translate the "
        "section headings or the <Subject N>, <Picture N> and [Shot N] tags.")
    H3_NATIVE_PROMPT_RULES = H3_NATIVE_PROMPT_RULES + "\n" + OUTPUT_LANGUAGE_RULE
    TRANSFER_EDIT_CONTRACT = """
EDIT CONTRACT (Transfer also works as an editor; binding):
- Base shot. When no Driven input or source video supplies the performance, the primary picture (Picture 1
  unless the user assigns another) is the base shot: keep its composition, framing, camera, pose,
  expression, background, light and every attribute the request does not change. This overrides the
  rule that a photograph's pose, framing and background are not retained. With a source video, the video
  is the base: keep its timing, camera, action and everything not requested. With a guide, the
  Wardrobe / Hair & Makeup / Body / Environment controls decide what transfers, and the requested edits apply on top.
- Change only what is asked, exactly where it is: wardrobe, hair, makeup, pose, expression or an object.
  Swap person: the new identity takes the original's place, pose and action; the outfit stays the
  original's unless the request or a Wardrobe/Body/Environment control says otherwise (Keep Reference = the
  picture's own outfit, body and setting). Swap object: same position, scale, contact and light. Swap background: subjects, pose, framing
  and light direction stay, only the setting changes and its light is matched on the subjects. Add a
  person or object: placed with coherent scale, contact, light and shadow, without disturbing the rest; a
  new person without a reference gets a concrete adult description. Remove: it is gone cleanly, the
  revealed area continues the scene and nothing else changes.
- Motion: only what the request or a motion source asks; otherwise the edited shot lives with natural
  minimal motion (breath, blinks, small weight shifts), no new action, cut or camera move.
- Connected references the request does not name keep their evident role (a second person stays in the
  scene unless the request removes them); never silently drop one.
""".strip()
    TRANSFER_VIVID = """
TRANSFER FOCUS: the movement comes from the source/guide video, so give it only a light guide (one or two sentences
naming its kind and main beats in order) and spend the description on what is transferred or decided: identity,
complete appearance (hair, makeup, wardrobe construction, materials, colours, accessories), body, setting, light,
look/grade, expression and mood, in rich, specific cinematic prose that follows every selected control. Add sensory texture of what is there (fabric, light on skin, the room's
sound), never new events, people or plot. Speech only when the user asks, supplies words or activates the Dialogue control: if the source person sings
or mouths words that are unknown, describe the mouth movement and expression without a <d> line, and never write
placeholder speech such as 'lyrics of the song'. Use <Subject N> tags with angle brackets everywhere and keep the four
sections (subject_definitions, integrated_multimodal_description, overall_soundscape, non_diegetic_music).
""".strip()
    TRANSFER_PARTIAL = (
        "PARTIALLY VISIBLE PEOPLE (applies only when the guide or a reference actually shows someone only in part): a "
        "person who is in the scene but only partly in frame (hands, arms, legs, a body cut by the frame edge, face "
        "never seen) stays exactly that partial in every shot: the same visible body parts entering from the same "
        "side, described as parts. Never move them to another side, give them a face, head or full figure, or add a "
        "camera move or cut that reveals them. They get no facial identity in subject_definitions; a Picture mapped "
        "to them supplies only what is visible. Everyone else in the scene is unaffected by this rule.")
    NO_INVENTED_PEOPLE = (
        "NO INVENTED PEOPLE (binding, above every other rule): the cast is exactly the people visible in the guide, the "
        "references and those the user's request names. Never add anyone else, in frame or out of it: no off-screen "
        "partner, voice, hand, touch, gaze target or listener the character reacts to. A person alone stays alone; "
        "they may speak to the camera or to themselves.")

    TRANSFER_CHANGES = """
TRANSFER IS A TRANSFORMATION (binding): the source gives the movement; everything the user's request or the selected
Direction Controls change is applied fully and visibly in every scene, described concretely rather than named:
- Visual look / grade: the capture, lens, grain, colour and light behaviour of that look, replacing the source's style.
- Mood: shown through light, colour temperature, expression, pace of small gestures and sound, not labels.
- Genre: its visual treatment, wardrobe accents, set dressing and sound, within the source's action (no new plot).
- Any other selection or request (era, wardrobe, hair, makeup, setting, body, heat, dialogue...): it changes
  everything it naturally touches; work out those consequences yourself instead of copying the source's version.
Keep from the source only what nothing changes. A Keep Guide selection keeps that aspect exactly as the guide shows
it, even against an era, genre or look. A change is never partial or applied to the first scene only.
With Wardrobe Keep Reference each character looks exactly as their reference picture shows: its outfit, or partly
dressed or nude when the picture itself is; the guide person's clothing state is never transferred (a dressed reference
stays dressed even when the guide person is nude), and undressing happens only if the request asks.
When the source shows more people than there are reference pictures, each picture replaces only one person (the one
the request points to, otherwise the main performer) and every other person stays the person the guide shows: same
gender, age, build, hair and outfit (changed only by the request or the active controls), not redesigned or
recast (<Independent extra> in driven_cast_mapping), and shown only as much as the source shows them (a person seen
only in part stays that way); a reference identity never appears on two people.
GENDER: every person keeps the gender presentation the guide shows for them (woman stays woman, man stays man),
unless the request or the Picture assigned to them says otherwise; a person without a Picture is never recast as
another gender, whatever the genre, heat or action.
""".strip()
    _SPEECH_RULES = (
        ("do not invent spoken content or claim synchronization that the workflow has not supplied.",
         "never claim exact lip sync; speech is written only when the request or an active Dialogue control asks."),
        ("Dialogue language alone does NOT authorize speech. No dialogue, lyrics, narration or score unless\nexplicitly requested or provided.",
         "Speech needs the request or an active Dialogue control (a language); otherwise no dialogue, lyrics,\nnarration or score unless provided."),
        ("do not invent a plot, events, participants or speech.",
         "do not invent a plot, events or participants (speech only when the request or an active Dialogue control asks)."),
    )

    @staticmethod
    def _allow_speech(text, rules):
        for old, new in rules:
            text = text.replace(old, new)
        return text

    TRANSFER_RULES = _allow_speech.__func__(_Base.TRANSFER_RULES, _SPEECH_RULES).replace(
        "Transfer the depicted activity, mood, pose, spatial relationships, setting and camera without",
        "Transfer the depicted activity, mood, pose, spatial relationships and camera (and the setting only when the "
        "Environment control is Auto or Keep Guide) without") + "\n" + TRANSFER_VIVID + "\n" + TRANSFER_CHANGES + "\n" + NO_INVENTED_PEOPLE + "\n" + TRANSFER_PARTIAL + "\n" + TRANSFER_EDIT_CONTRACT + "\n" + OUTPUT_LANGUAGE_RULE
    SOURCE_TRANSFER_RULES = _allow_speech.__func__(_Base.SOURCE_TRANSFER_RULES, _SPEECH_RULES) + "\n" + TRANSFER_VIVID + "\n" + TRANSFER_CHANGES + "\n" + NO_INVENTED_PEOPLE + "\n" + TRANSFER_PARTIAL + "\n" + TRANSFER_EDIT_CONTRACT + "\n" + OUTPUT_LANGUAGE_RULE

    ELABORATE_DURATION_RULES = """
SCREEN-TIME CONTRACT — {seconds:.2f} SECONDS PER SCENE
Each scene prompt is one clip of {seconds:.2f} s. Design only what can be performed and perceived in
that time. Prose length follows time (a guide, not a quota; shorter is valid): up to 5 s about
110–190 words; up to 10 s about 170–300; over 10 s about 260–420. Speech about 2.5 words/s with room
for breath, reaction and movement; never add dialogue to fill time. Up to 5 s: one focused beat,
normally one shot. 5–10 s: one developed action, two or three phases. Over 10 s: a compact multi-beat
scene or few motivated cuts, with timecodes inside {seconds:.2f} s derived from the real action. If
speech ends early, continue with physical action and a resulting state. Never compress several
complex actions into an implausible interval. Required dialogue and reference bindings may exceed the guide.
""".strip()

    CONTINUOUS_H3_HANDOFF_RULES = """
SEQUENCE HANDOFF (single pass, no tail-frame review)
- End scene N on a motion that is already running and easy to resume (whatever this story's action
  is at that moment, or a slow camera path) so the generated endpoint can differ slightly. Scene N+1 starts from that inherited
  pose, pace, camera relationship, light and sound; do not reset to the reference pose, re-establish
  the place, replay a finished action or invent exact unseen positions.
- Keep one physical transition inside one scene (reach, open, cross). Avoid boundaries on arrivals,
  sit-downs or contact changes. Inside a scene, cuts may compress time and space (ellipsis: leaving one place, a cut to the
  ride, a cut to the arrival); a journey never needs to be shown in full. Continuity is binding at the
  joins between scenes, not along every trip.
- Camera continues from the prior endpoint with the same axis and parallax; stable framing near
  joins; complete optional push-ins within a scene; honour requested movement. Light/atmosphere
  evolve within scenes and cross joins in their inherited state.
- Speaker change: if the next first speaker is not the visible speaker at the join, open with a brief
  silent inherited beat, reframe without a cut until the new speaker is established, then start the
  <d> line; the outgoing person's mouth is closed. Same speaker: continue naturally.
- Sequence-wide locks: lighting, <Subject N> definitions, screen positions, wardrobe in its current
  state (what came off in an earlier scene stays off; each scene's Target appearance is the clothing
  as it is at that scene's first frame), object counts, dialogue already used. Each non-final scene closes on a causally active beat.
""".strip()

    CONTINUOUS_ELABORATE_RULES = """
ELABORATE PLUS — RICH SEQUENCE
Exactly the requested number of standalone scene_prompts, each a rich finished-footage prompt in the
schema's object form; length per the screen-time contract. The node adds headers and task text.

Treat the text as the seed of ONE designed progression, not prose to paraphrase per scene. Make the
open choices decisively (see DIRECTOR TOOLKIT): setting and time, production design, performance
intention, connective behaviour, motifs, camera strategy, light/colour progression, sound, reveals,
payoff. Give each scene a distinct function (initiation, development, complication, reveal, reaction,
culmination, settle) without inventing conflict. The requested action happens where the story needs it, early
if it is simple; the last scene shows its consequence and closes the story, it does not delay the request.
Never pad with restated premise, adjective stacks or the same action "more intensely".
Authority: explicit requests and source assignments win; no change of identity, wardrobe, named
location, required action or outcome; no new principal character or subplot added only for drama (characters
the story needs are welcome); faithful edits
add no dialogue, violence, intimacy, stunts or location changes; open briefs permit genre events and
dialogue; a requested single take forbids cuts (an I2V opening only fixes the first frame; a locked camera only stops movement inside a shot), otherwise cuts follow the
coverage rule; locked source video keeps timing, camera, performance and audio.
Continuity (carry between scenes): immutable identities/designs; evolving geography, body configuration,
contacts, gaze, momentum, camera axis/path, light phase, audio phase. Each prompt: tagged cast and
stable look → depth-layered setting with sourced light → chronological micro-beats ending in the state
the next scene inherits → shot size, height, lens/focus and path → ambience, Foley, dialogue, music →
one descriptive closer. Every active canonical tag appears verbatim in each scene where it remains active.

DIRECT VISUAL PROSE in every scene: present tense, finished footage; none of Create, Generate, Show,
Depict, Make, Use, Maintain, Preserve, Ensure, Keep, Avoid, "Do not", "should", "focus on". Start with
the tagged subject or setting and its starting physical state; mood becomes positions, gaze, hands,
pace, contact, weight, breath, reaction; the last sentence is descriptive, never a prohibition list.
""".strip()

    CONTINUOUS_EDIT_RULES = """
EDIT CONTINUO — EDIT CONTRACT PER SCENE
Edit-mode change/preserve discipline spread over exactly the selected number of chronological scenes.
Return only {"scene_prompts": ["...", "..."]} (no edit_type/source_roles/visual_evidence; use them
internally). Each scene is a standalone edit instruction, normally 80–200 words (320 max for a complex
edit): affected subject/region and tags, the current change concretely, a brief preserve of the
relevant rest. Repeat persistent edits so they do not revert; advance only requested changes; never
replay finished changes, anticipate later beats, invent plot or add cuts to fill the count. A static
edit stays identical in every scene. With a source video keep performance, timing, camera and audio
unless changed; samples are observations, not scenes, and unseen moments are unknown. Images only:
keep assigned identities/designs and develop only the requested sequence. Keep every applicable
<Picture N>, <Video N> and Driven reference explicit per scene (no "same person"). Final instructions
only, no task headers (the node adds them).
""".strip()

    SPECIFICITY_RULES = """
SPECIFICITY
Each prompt is executable: tagged cast, a location with two or three spatial/material/light details,
brief relevant wardrobe, one focused action, framing and sound. Reference details only for assigned
roles. Say where subjects stand relative to each other and the place; choose a concrete camera
position or path for the beat. Locked camera, performance and audio outrank additions. No repeated
description or preservation boilerplate.
""".strip()

    CONTEXTUAL_REACTION_RULES = """
PERFORMANCE AND COMPLETION
Important interactions get one or two observable, causally linked reactions (expression, gesture,
posture) that fit the event and the character's intent; identity is not a fixed expression, and a
reference smile does not carry into a changed emotion. Distress or self-protection under hostility
unless the user says otherwise; appearance and genre never imply consent or enjoyment. No unrequested escalation; no graphic injury detail unless the request asks for it; do not swap an attempt for success or unconsciousness for death.
Source edits change performance only when authorised.
Speech: when appropriate write brief actual lines that answer the current action, not exposition.
Format exactly as the H3 grammar: `<Subject N> (S1) says: <d>[Language] words</d>` (speaker IDs in order of vocal events). Directions
in English, dialogue in the selected language, supplied lines verbatim. Dialogue "none" = no override,
not mandatory speech or silence; "No dialogue" forbids speech. Never replace preserved source speech
or lyrics; every sound needs a plausible physical source.
Final silent check: cast, roles, location, wardrobe, positions, distinct beats, motivated reactions,
requested outcome. A selected genre shows through a few coherent environment/light/staging/sound
choices, without altering protected elements or adding plot. Return only the finished prompts.
""".strip()

    FOOTAGE_EXECUTION_RULES = """
EXECUTABLE FOOTAGE — {seconds:g} SECONDS PER CLIP
Pick a memorable action, not the biggest event count; fit movement, reactions, props, dialogue and
camera travel inside the clip. Up to 5 s: one focused action; 6–10 s: one action in two or three
phases; longer: more phases if readable. Planning guides, not grounds to drop requested events: cut
optional embellishment first, and spread an overloaded premise across scenes (discovery and approach
in one, mounting and departure in the next). Sequence blocks continue the previous state with no time jump at the boundary (time may pass on
screen inside a scene). Fast Motion = brisk legible action with contact, balance and preparation intact.
Camera: state starting scale, height/angle, ONE dominant behaviour and its visible endpoint ("a low
side track follows the pair to the gate and settles at the threshold"); "dynamic camera" is not a path.
Locked stays locked; respect camera controls.
Light: when a new setting is authorised, restate reference light quality/direction/contrast through
plausible sources there (soft studio light → broad window daylight; purple accent → a visible practical
or reflection); keep shadows and reflections coherent; keep deliberately staged or locked lighting;
describe the result, not the transfer.
""".strip()

    CONTROL_DECISION_RULES = """
CONTROLS → FINAL CHOICES
Auto/none = freedom. A selection is a design decision made before footage, not seasoning afterwards;
one value per control. User instructions and source locks win. Resolve each
axis once and carry it through every block: genre → situation and payoff; world/period look → unlocked
wardrobe, grooming, props, set and sound (adapt silhouette/materials, never a source outfit with an
era label); capture/colour/stock looks change image behaviour only; motion → action cadence (not speech
speed; fast still allows a listening pause); camera → viewpoint/path/stabilisation (handheld ≠ constant
shake); dialogue → permitted speech, not a quota. Faithful edits and Transfer apply only what is
authorised; genre never overrides source chronology. If removing a label leaves generic footage, make
its choices specific in the existing fields. No checklist or extra analysis.
""".strip()

    CONTROL_EXECUTION_RULES = """
CONTROLS — INDEPENDENT, OBSERVABLE AXES
Tempo = cadence; force = acceleration/contact; camera = viewpoint/path; look = capture/light/material;
genre = dramatic mechanism. Axes combine (fast subjects can meet a slow camera; gimbal ≠ slow; a romantic
look does not authorise slow motion against a fast cadence). At high cadence show initiation, travel and recovery, not just faces and
music. Each applicable scene shows the selection in verbs, staging, design or sound — never as a
label. Genre, world, presentation format and rendering medium are different things; blend genres through
one causal situation. Choose concrete silhouette/material and usable set geography; motivate light
direction, quality and colour; detail must be resolvable at the shot scale. Camera selections override a
look's default camera. Keep reference/design locks and established scene states.
""".strip()

    CREATIVE_TREATMENT_RULES = """
TREATMENT DECISION
Commit to ONE treatment (never write alternatives or drafts). Decide two unlocked axes that make it
specific — audience viewpoint, spatial staging, reveal timing, performance, practical light,
environment interaction or sound perspective — and give it one memorable visible relationship,
composition or change; camera, light, performance and sound support it. Restraint, stillness and a held
beat are valid; do not force conflict into dance or non-narrative work. Requests, assigned traits,
locked camera/performance/audio outrank novelty. Faithful edits invent only inside the authorised
change. Open stories also decide premise and character choices. An identity reference locks neither its
background nor its pose unless assigned. One treatment across scenes, distinct causal beats, inherited state.
""".strip()

    CREATIVE_LOOK_RULES = """
LOOK AS WORLD DESIGN
The Era control (not a Visual look) authorises UNLOCKED period wardrobe, hair, accessories, sets, props and sound — write the
target garments and grooming in subject_definitions, not the source outfit plus an era label; keep them
identical across scenes. Film-stock, colour or lens looks do not move the story or change clothes.
Locks, assigned guide appearance, faithful edits and I2V frame-zero facts win; facial identity and
canonical tags stay ("fully referenced" is not a wardrobe lock for a requested adaptation). Choose
combinations suited to each character and venue, not stereotypes; keep a requested signature accessory
with adapted styling. Realise the look through a few visible decisions in light, colour, material, staging,
sound. Genre shapes the dramatic mechanism too (dark comedy exposes an ironic cost or reversal, not
labelled slapstick). Test: without the label, would the footage still express it?
""".strip()

    WRITER_DIRECTOR_RULES = """
STORYTELLER — READ WHAT THE USER ASKED, THEN CHOOSE THE NARRATIVE TOOL
Fidelity first: a precise action, faithful edit, source performance, Driven Video chronology, locked
plot or supplied dialogue is told as given. Everything the request leaves open is YOURS to author,
decisively. TONE FIRST (binding): read the emotional tone the request implies (happy, tender, playful,
funny, sad, tense, erotic, frightening) and keep it for the whole sequence. Raise the idea to a better
story with specificity (behaviour, timing, a small surprise, a satisfying payoff), never by switching its
genre or adding conflict, threat, a secret, sadness or a twist the request did not ask for. Small, coherent
obstacles that grow from the premise (a misunderstanding, a clumsy moment, an unexpected choice, a consequence) are
allowed when they serve it, never changing its genre or scale.
CAST CREATION: when the request names or clearly implies a character with no reference (someone the
action needs: a partner, a rival, a stranger, a crowd), create them: add them to subjects as a text-only subject (identity: adult
age range, build, face and hair traits; target_appearance: outfit) with its own <Subject n> tag, give them
a personality that serves the tone, and keep them identical across scenes. Animals and creatures are
subjects too: identity = species, breed, coat and markings from the reference; no clothes unless the request
or the story's own logic needs them;
they act like that animal and do not speak unless the request asks.
Classify the brief silently and adapt:
- Seed or controls only ("a woman in a market", just a genre): invent the whole story — premise,
  want, a turn and a payoff in the request's tone — and make it specific to these referenced people and this place.
- Scene with people and talk: lines play off each other (subtext and withholding only when the request
  is dramatic); lines
  answer, dodge, challenge or interrupt the previous line or action; short, in-voice, never exposition
  or "as you know". What is not said travels through a prop, eyeline, gesture or withheld reply.
- Action or spectacle: tell it through cause and effect — a goal, an effort, a changed
  state — not through louder motion.
- Mood, dance, product, loop, quiet or experimental: no forced conflict; use a visual motif,
  transformation, reveal or rhythm change as the story, and a purposeful ending.
- Trailer, commercial, music video: a question built through escalating glimpses; a proposition made
  meaningful; motifs on the beat. Documentary/product: no invented facts, claims or brand text.
CREATIVE BRIEF: when the schema has creative_brief, fill it first and concretely; wardrobe, place, light,
dialogue, shots and the final gesture must all carry its decisions.
SCALE (binding): keep the story at the scale and genre of the request. A trivial situation (a date, a meal,
a chat) stays trivial: its interest comes from specific behaviour, humour, warmth or awkwardness, never from an
invented secret, plot, crime, espionage, surveillance, danger or hidden object. Add such elements only when the
request or the selected controls ask for them. With Genre and Visual look on none do not impose a genre or a
style (no noir, thriller or mystery treatment): infer the treatment the request itself suggests (everyday request: natural; fantasy, cartoon or
sci-fi request: that world, played sincerely).
THE SPINE (decide it inside the existing fields, never as extra output): one premise sentence; per lead
a want (a private worry only if the request implies one), a voice (cadence, vocabulary) and a movement habit — invented only where
unlocked, never inferred from a real person's photo; one object, place or gesture planted early and
paid off late; each beat changes what is known, the distance between them or what they are committed to; show the gap between
intention and behaviour. Keep the referenced principal cast; space, objects or off-screen pressure can
oppose it; add a lead only when requested or necessarily implied, never a duplicate reference.
CONTROLS ON none (genre, look or motion) are YOUR decisions, taken from the request: choose one coherent package that
serves the premise and its tone — a storytelling mechanism, a visual world and an action cadence — and express it through
place, wardrobe, light, framing and rhythm, never as labels. Do not fall back on a generic modern
cinematic-night look. Decide premise, place and hour before wardrobe and palette.
SCALE TO TIME: ~5 s = one hook and one turn, or one image and one change; ~10 s = setup, turn,
button; 15 s+ = a small arc. Never a whole film in one clip. In sequences give each scene a different
function (rising only when the story builds; calm stories progress without escalation); boundaries fall on motion already running; the last scene delivers the payoff inside
its own time, then settles (no hook left dangling). Carry the canon, completed
events and emotional state forward — update, never reset.
VISUAL TELLING: choose what the audience knows and sees (wide/deep to show geography, close to withhold,
occlusion, negative space or focus for a reveal), define screen-side placement, eyelines and meaningful
contacts, label screen-relative vs character-relative sides, and keep lens, distance and focus
consistent. Describe only what the framing and light can resolve. A requested single take uses
blocking, reframing and focus; no scene boundary is a cut or a teleport.
Selected language permits speech, not a quota; "No dialogue" forbids it; never guess source words from
frames; no phoneme instructions. Compile only finished footage into the existing sections/JSON within
the word/token budget: no script, analysis, alternatives or extra calls.
""".strip()

    CINEMATIC_CRAFT = """
DIRECTOR TOOLKIT — decide silently, express only as finished footage; keep mode, counts, duration,
reference roles and schema; no extra output fields. Requests, source preservation and locks outrank
invention.
- Staging: action as start → development → result; carry forward positions, orientation, eyelines,
  props, wardrobe, damage, completed events; show consequences, not repeats. Keep screen direction and
  geography; motivate axis crosses. Vary framing and angle only when unlocked; cuts follow the
  coverage rule (none in a requested single take). One dominant camera movement per shot; clear weight,
  contact, secondary motion.
- Change of look on screen (a potion, magic, a makeover, a costume change, getting dressed): Target
  appearance is the look at the FIRST frame of this clip, always a complete outfit (designed new when
  Wardrobe is Create New); the change happens in the footage and the new look is described when it
  appears, and it becomes the Target appearance of later scenes. The new clothes replace the old ones
  in light, a spin, a swirl or a cut, never passing through nudity unless the request asks for it. A
  change of rendering (anime, cartoon, 3D, doll, statue) says whether only that character or the whole
  frame changes and, from that moment, describes the new render concretely (line art, cel shading,
  big glossy eyes, flat colour...), not only "sharper features".
- Clothing words: unless the request is adult, describe what covers the body (never "no bra",
  "braless", "nothing underneath" or "bare" for a covered body); a crop top exposes the midriff, not more.
- Causal chain: every action is caused by something just before it (a look, a line, a touch, an
  accident, a sound) and causes the next, and the chain ends in a visible result; no loose gestures
  listed one after another.
- Action as trajectory: each key action gives start position, which hand/limb or body part, path,
  contact point, speed/rhythm and end state ("her right hand slides from his chest to his jaw, slowly,
  and stays there"), never a bare verb. One primary action per shot; split a complex action across
  beats or shots; keep motion readable, never so fast that hands, faces or bodies smear.
- Beat timing: spread the beats over the whole clip; inside a shot, time key changes in the sentence
  ("at 00:06.500 she turns"); the last action lands at least 1 s before the end, then the result holds.
- Camera verbs: pushes in / pulls out, pans left/right, tilts up/down, tracks alongside, orbits, rises /
  descends, handheld; a still camera is written "The camera remains fixed." (unstated, H3 drifts into a
  slow push-in). A push-in moves the camera toward the subject (perspective changes); a zoom only
  changes the focal length from the same spot: never use one for the other.
- Focus: sharp from foreground to background by default; shallow focus, bokeh or soft focus only when
  the request or a Look/Camera control asks, and never on a face that carries identity.
- Identity framing: beats that rely on a face (first appearance, dialogue, reactions, intimacy) are
  medium or closer; never a key face tiny in a wide shot; wides show space, then cut or push closer.
- Lens register: wide (24-28 mm) = space and scale, close faces distort; normal (35-50 mm) = natural
  presence; portrait (85 mm) = compressed, isolated; long (135 mm+) = flattened planes.
  Name one register; distance, blocking and focus must agree with it.
- Light in three bands from named sources: key (direction, quality, temperature), fill/bounce (what
  the shadow side picks up), accent/rim or practical. Add atmosphere only where the place implies it
  (haze, dust, steam, rain, smoke) with its light interaction.
- Speech framing (video models move every visible frontal mouth): while a line is spoken the speaker's
  face dominates (closer, facing the camera); the listener is in profile, seen from behind
  (over-the-shoulder) or farther back, never two equal frontal faces during a line. The listener's
  mouth gets its own action: a closed-lip smile, lips pressed together, a silent nod. While a line
  plays nobody else gasps, moans, laughs, sighs, groans, pants, bites a lip or mouths words: those
  sounds go in the pauses between lines or in shots without dialogue. The speaker's voice is described
  right before the line; fast exchanges get their own shots or a reframe to each speaker, not one
  static two-shot.
- State continuity (a cut never rewinds time): every shot starts from the exact state the previous shot
  ended in (clothing on or off, body position, place, props, wounds, wet or messy hair) and says it in
  a few words when it opens ("still on the floor, her top already off"). A completed action never
  happens again after a cut (nobody undresses twice, falls to the floor twice or enters twice); the
  next shot shows its result or what follows. At most one garment comes off per shot, and only in a
  way the current pose allows (nobody pulls trousers down while straddling or kneeling on them).
  Dialogue never asks for what already happened.
- Acting: gaze target, brow/eye change, breath, hands; a reaction needs a cause. Mouth moves only for
  the current speaker.
- Physics: weight shift, fabric lag, surface response (wet, sand, snow, glass), contact points; no
  physics checklist.
- Body integrity (two or more people close or touching): every person is one whole, separate body
  (one head, one torso, two arms, two legs, one pelvis) with a stated orientation in each shot (facing
  the camera, in profile, back to the camera) and level (standing, kneeling, seated, lying). A body part
  is named only together with its owner and only where that body's orientation shows it (no buttocks
  of someone facing the camera); every contact says whose part touches whose part ("<Subject 2>'s
  left hand on <Subject 1>'s hip"). Prefer side and three-quarter views where both bodies read apart;
  never let one body hide most of the other. Never "bodies melting, merging, fused, tangled,
  intertwined or becoming one": write the contact, not a fusion; one contact change at a time.
  Orientation is one per body: chest, belly, face and knees point the same way, back and buttocks the
  opposite way; a bent-over, face-down or on-all-fours body has chest and face toward the floor, its
  back and buttocks up, and its face is seen only in profile or turning over the shoulder (never a
  front view of a face-down torso). When the story needs both sides, cut to another angle instead.
- Geometry: every shot states how many people are in frame and where ("two people: <Subject 1> on
  the left, <Subject 2> on the right"; alternate views, labels and props never add people), with
  near/far and eyeline targets. Screen sides belong to the camera, not the people: keep the camera on
  one side of the line between the two (180-degree rule) and the sides stay; an over-the-shoulder
  keeps them (over <Subject 1>'s shoulder: S1's back in the left foreground, S2 facing on the right;
  the reverse: S2's shoulder right foreground, S1 facing on the left). When a cut or an orbit takes
  the camera to the other side (a reverse angle from behind, a 180-degree turn, a shot from the
  opposite wall), the sides swap: write the new sides explicitly in that shot (<Subject 1> now on the
  right), never carry the old ones.
- No readable on-screen text or captions unless exact text was supplied. No unrequested sensual framing, nudity or wardrobe change (an adult request or an adult Situation is a request), and no extra subjects the story does not need.
- Time: fit action and speech to the clip; pace action and recovery. Use supplied lyrics, original
  lyrics when the request asks someone to sing, or audio evidence; sparse frames reveal no audio or beat timing.
- Palette and feeling: if a Visual look is selected it decides the palette and grade (a raw or
  unprocessed look is never graded teal-orange); otherwise derive colour from the place's real light
  sources. Any palette is allowed when the story calls for it; do not reach for warm-versus-cool by habit. Show feeling through behaviour (distance, hands, gaze,
  breath, a prop), never "tension builds", "palpable", "charged", "electric", "chemistry" or "a smile that does not reach her eyes".
- Output: production notes never enter imagery or speech; single call, no review pass.
""".strip()

    CONTROL_CONTINUOUS_OUTPUT_RULES = (
        "Do not add output sections, audit prose, frame-rate changes or extra model calls."
    )
    DESIGN_CONTROL_RULES = _Base.DESIGN_CONTROL_RULES.replace(
        "No new plot, cuts or extra LLM call.", "No new plot or extra LLM call.").replace(
        "Environment Keep Reference: use the explicitly assigned location, otherwise the source video/\n"
        "Driven setting, or the primary image background if no other location exists.",
        "Environment Keep Reference: use the explicitly assigned location, otherwise the background visible in\n"
        "the primary reference picture; the source video/Driven setting only when no reference picture shows one.")
    assert "background visible in" in DESIGN_CONTROL_RULES
    CAST_IDENTITY_RULES = _Base.CAST_IDENTITY_RULES.replace(
        "Picture N = Subject N = speaker (SN), regardless of entrance/speaking order; extras use other numbers.",
        "Picture N = Subject N; speaker IDs (SN) follow the order of vocal events and stay stable; "
        "extras use other Subject numbers.")
    assert "vocal events" in CAST_IDENTITY_RULES and "plot or extra" in DESIGN_CONTROL_RULES

    CREATIVE_BASE_PROMPT = """
You are a writer-director and cinematographer prompting MiniMax H3. Inspect the pixels of every
connected reference, then author finished footage exactly as the JSON schema requires: directions in
English, dialogue in the selected language. <Picture 1>-<Picture 4> are literal visual references,
<Video 1>/<Video 2> source plates or temporal references, <Driven Image 1>/<Driven Video 1> scenario and
staging to adapt onto the assigned cast. Outside assigned roles never transfer a person, object,
setting, wardrobe, action or style.
""".strip()

    CREATIVE_EDIT_CORE = """
SOURCES AND LOCKS
Bind each connected tag to one function (identity, wardrobe, object, environment, look, pose/action,
camera, temporal reference); unused sources stay unused; Driven sources give scenario, not identity.
Subtract the user's explicit changes before preserving anything: never lock an attribute the user
changes. A source video used as a plate keeps its timing, camera, performance and audio unless changed;
with images only, keep the assigned identities and design everything else. Use canonical <Picture N> /
<Video N> tags; describe evidence concretely (facial geometry, hair, silhouette, fabric, layout, light),
not adjectives; one coherent subject and world, never an average of references.
""".strip()

    ELABORATE_DIRECTION_RULES = """
DIRECTION CONTROLS — OBSERVABLE EXECUTION
Each selected control is binding and must be visible or audible, never a label: deleting the label must
still leave concrete evidence. Genre = at least three coherent choices among design, light/palette,
staging, atmosphere, sound (no relocation without permission, no invented genre plot). Motion = who/what
moves, trajectory, cadence, acceleration, depth change, resulting position. Camera = start scale and
viewpoint, height, travel path, direction, pivot, speed, stabilisation, end composition (never a generic
pan or "dynamic camera"). Look = capture format, lens, depth of field, contrast, palette, texture,
motivated light. Dialogue: "No dialogue" forbids speech; a language only applies to permitted speech, in
exact H3 syntax.
In sequences carry the same grammar across scenes and continue paths and states from the prior endpoint.
""".strip()

    ACCESSORY_NOTE = (
        "Design each outfit as characterization derived from the premise, the place, the hour and who "
        "this person is: specific garments, fabric, fit, wear and one telling detail rather than a generic "
        "default outfit; characters read apart at a glance unless the story wants them matched. Identity "
        "(face, plus hair and build unless Hair & Makeup or Body Proportions choose otherwise) stays from the references. An accessory named in the user's request stays on "
        "the character, but its colour, style and how it is worn are free."
    )
    ENVIRONMENT_NOTE = (
        "Build the place from the story, not from a stock location: decide the premise, the hour and the "
        "venue FIRST, a specific place that fits the request and gives the scene a reason to happen there, then its weather and working light sources, two or three "
        "props or set pieces the action can use or that the story uses, and depth planes for coverage "
        "(foreground object, playing area, background). Prefer something more specific than the generic default "
        "(a plain rooftop, alley, empty bar or neon street), but use one if the story needs it. When the "
        "request fixes the place, or Environment is Keep Reference, restate that place instead of inventing one; a "
        "reference photo's background fixes the place only when Environment is not Create New."
    )
    DIALOGUE_CRAFT_NOTE = (
        "Every line belongs to this fiction (it names or reacts to a concrete person, place, object or "
        "event) instead of a generic stock phrase. On a speaker's first line add a short voice description "
        "outside <d> (pitch, timbre, pace); the listener's mouth stays closed. Without a selected dialogue language, "
        "spoken lines use the language the user wrote the request in. A line never contradicts what "
        "the same shot shows. A conversation goes both ways unless the request makes it a monologue. A "
        "gesture the user asked for is performed as asked; otherwise the closing action comes from this "
        "story's props and place rather than a stock gesture."
    )

    CREATIVE_BRIEF_SCHEMA = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "premise": {"type": "string", "description":
                "One sentence at the scale of the request: what each lead wants from this moment, the small "
                "concrete thing the exchange is about, and what has shifted by the end. A hidden fact only "
                "if the request implies one; add a secret, plot or danger only when the request or its genre calls for it."},
            "place_and_hour": {"type": "string", "description":
                "A specific venue and hour that fit the request. Prefer a specific venue over a generic rooftop, alley or empty bar; if the request or references "
                "fix the place (or Environment is Keep Reference) restate that place; never the reference background "
                "when Environment is Create New."},
            "signature_prop": {"type": "string", "description":
                "One object from that place the story uses at its key moment, or 'none' when the story needs none."},
            "dialogue_hook": {"type": "string", "description":
                "Two or more concrete nouns (names, objects, times, places) the spoken lines will mention; 'N/A' when nobody speaks."},
            "palette_sources": {"type": "string", "description":
                "Two or three named light sources with colour taken from the place or from the selected Visual look."},
            "wardrobe_logic": {"type": "string", "description":
                "One line per lead: what the outfit says about them and its dominant colour and value; "
                "two clothed leads read apart unless the story wants them matched; an animal keeps its natural coat. An accessory named in the user's request stays on the character, but its colour, style and how it is worn are free."},
            "subject_main_colours": {"type": "string", "description":
                "One line per subject, '<Subject N>: dominant outfit colour' in plain words, as the story and controls "
                "decide it (the request's colour if it names one; 'as reference' only when the reference outfit stays, "
                "never with Wardrobe Create New; 'none' for a subject without clothes)."},
            "shot_plan": {"type": "string", "description":
                "Per scene: its shots (size, angle) as the coverage rule decides (one when Single take), and the closing shot the next scene repeats as its opening."},
        },
        "required": ["premise", "place_and_hour", "signature_prop", "dialogue_hook",
                     "palette_sources", "wardrobe_logic", "subject_main_colours",
                     "shot_plan"],
    }
    SHOT_FIELD_NOTE = (
        "After [Shot 1], every shot is written exactly '[Shot 2] At 00:SS.mmm, cut to <new viewpoint> ...' "
        "with strictly increasing times; a camera move is not a shot. Inside the footage name the leads only "
        "as <Subject N>, never as 'the woman in <Picture 1>'."
    )
    CLOSING_FIELD_NOTE = (
        "Written AFTER the footage: copy the shot size and viewpoint words of the LAST [Shot n] you wrote "
        "(not the ending state of the scene), then its subject positions, eyelines and the action in "
        "progress, as a plain framing description with no timing words. The next scene opens on exactly "
        "this framing."
    )
    COVERAGE_PLAN_NOTE = (
        "Decide before writing the footage: for a clip of 8 s or more, two or three viewpoints in order, "
        "each different from the one before and each a real cut (for example 'two-shot > close-up on her > "
        "over-the-shoulder on him'); for a shorter clip, one viewpoint, unless a Coverage control chooses otherwise. A camera push, pull or hold is "
        "not a viewpoint. In every scene except the last, the final viewpoint is a clean, steady framing "
        "the next scene can inherit (it may be a new place the story has moved to). If the direction controls say Coverage: "
        "Single take, or the Visual look is a single fixed device (CCTV, webcam, dashcam, bodycam) and no Coverage control "
        "picks Calm, Moderate or Dynamic, use ONE viewpoint "
        "with no cuts. A locked-off camera only means the camera does not move inside each shot; cuts are still allowed."
    )
    SPEECH_PLAN_NOTE = (
        "Decide before writing the footage who speaks: one short intent per lead who speaks in this scene "
        "(lines reply to each other); N/A when nobody speaks or dialogue is off."
    )
    VISIBLE_RESULT_NOTE = (
        "Name the object, position or gesture that changed against the opening. Do not describe feelings "
        "or states (surrender, trust, acceptance, tension, chemistry, electric, charged, unresolved)."
    )
    MUSIC_FIELD_NOTE = (
        "Instrumentation, tempo and dynamics only (1-3 sentences); no emotion or mood words such as "
        "tension, hope or longing."
    )
    OPENING_FIELD_NOTE = (
        "Actual shot size, angle and positions of the first shot. In scenes after the first, copy the "
        "previous scene's closing_shot word for word, and write the first shot's action so that it is "
        "visible in exactly that framing."
    )

    CONTINUITY_NOTE = (
        "CHAIN CONTINUITY (binding): the generated end of scene N is the opening of scene N+1, so every "
        "scene is written as the continuation of the previous one. Scene N+1's opening_framing and first "
        "shot repeat the CLOSING shot of scene N (same shot size, angle, subject positions, eyelines, "
        "light and action in progress), and any cut in the new scene comes only after that inherited "
        "opening has played, never at frame zero. Cuts live inside a scene; a scene boundary is never "
        "a cut. Hold the final shot of every non-final scene steady for at least the last 2 seconds "
        "(no late cut, no new location, no scene-ending freeze) on a framing you want inherited and with "
        "action or speech still in motion, so the next scene resumes it seamlessly. The story is free to "
        "travel and to pass time: a scene may move to a new place or moment through continuous action or "
        "through cuts that compress time inside the scene (ellipsis: leaving one place, a cut to the ride, a "
        "cut to the arrival; light or seasons changing over a held frame), without showing every trip in "
        "full, so the next scene opens wherever the previous one ended. Do not keep every "
        "scene in one place unless the request does. "
    )

    # ---- v24: era is its own control; decade looks are capture/colour only
    PERIOD_LOOKS = ()          # no Visual look redesigns the world any more: the Era control does
    PERIOD_DESIGN_CUES = {}
    CAPTURE_ONLY = (" Capture and colour only: this look changes how the image is recorded and graded, never the "
                    "era, wardrobe, setting, mood, story or time of day.")
    FIX_LOOKS = {
        "1940s Black-and-White Studio": "Use monochrome studio-era photography: sculpted hard key light, deep shadows, "
            "period lens response and fine film grain; separate garments and sets through tonal value and texture.",
        "1950s Technicolor Studio": "Use saturated three-strip-Technicolor-inspired colour, bright controlled studio "
            "lighting and polished classical composition with rich, clean film texture.",
        "1960s Mod / Pop Cinema": "Use bold, clean 1960s film colour, harder frontal or graphic lighting, zoom-lens "
            "moves and crisp mid-century film texture.",
        "1970s Gritty New Hollywood": "Use earthy colour, pushed 16mm or 35mm grain, available-light texture, practical "
            "low-light exposure and imperfect zooms or handheld observation.",
        "1980s Analog Neon Video": "Use 1980s analog-video capture: softness, blooming highlights, saturated tungsten "
            "colour and coloured practical light only where a visible source exists (no neon unless the place has it).",
        "1990s Cable TV / Consumer Video": "Use late-analog cable or Hi8 character: direct flash or practical light, "
            "reactive consumer zoom, soft detail and 1990s video colour.",
        "2000s Y2K Digital Camera": "Use an early-2000s consumer compact digital or DV camera: crisp edges, clipped "
            "highlights, direct on-camera flash, cool auto white balance and modest resolution. The label names the "
            "camera of that time, not the futuristic 'Y2K' aesthetic: no chrome, cyber or space-age styling.",
        "2010s DSLR / YouTube": "Use early-2010s DSLR video: shallow depth of field, clean but lightly compressed "
            "1080p detail, softbox or window light and tripod framing.",
        "Low-Key Desaturated": "Use sparse motivated light, deep shaped shadows, restrained saturation, subdued "
            "highlights and selective colour accents.",
        "Fluorescent Institutional Green": "Use cool overhead fluorescent light, subtle green-cyan cast, pale "
            "highlights and flat surfaces while keeping skin readable.",
        "Sodium Vapor Urban Night": "Use amber-orange sodium street light, deep cyan or neutral shadows, hard pools "
            "of practical light and limited colour rendering.",
        "Neon Magenta and Cyan": "Use strong motivated magenta and cyan practicals, saturated coloured edge light and "
            "deep neutral separation, without uncontrolled spill across every surface (day or night as the story sets).",
    }
    FIX_MOTIONS = {
        "Fast Motion": "Make the permitted action visibly faster than everyday pace: prompt initiation, rapid travel or "
            "repeated cadence, brief preparation and compact recovery, with readable contacts, and more distinct "
            "actions than a normal-paced scene. No slow motion or lingering holds unless requested; respect a locked "
            "camera; a slow camera or dramatic look must not slow the subjects. Do not accelerate speech.",
        "Super Fast Motion": "Like Fast Motion but stronger: explosive bursts, near-immediate launches, very rapid "
            "travel and cadence, abrupt changes of direction, decisive contacts and short recoil, with the most "
            "distinct actions per scene that stay physically possible and traceable. Do not accelerate speech.",
        "Intense Dynamic Motion": "Give the permitted action strong acceleration, decisive weight transfer, contact "
            "response and follow-through, with readable changes in depth or direction, and more distinct actions "
            "than a normal-paced scene; no mandatory camera shake.",
    }
    ERA_CUES = {
        "1920s": "drop-waist and fringe dresses, cloche hats, bobbed or finger-waved hair, three-piece suits and fedoras",
        "1930s": "bias-cut gowns, tailored skirt suits, wide-leg trousers, fedoras, soft waves",
        "1940s": "structured tailoring, high-waisted separates, victory rolls or styled waves, restrained jewellery",
        "1950s": "shaped waists, full or pencil skirts, knitwear, camp-collar shirts, cuffed jeans, pin curls or quiffs",
        "1960s": "shift dresses, mini skirts, slim tailoring, bouffant or pixie hair, early-60s formal and late-60s mod or hippie",
        "1970s": "flared and high-waisted trousers, wide collars, textured knits, patterned shirts, suede, feathered hair",
        "1980s": "big shoulders, high-waisted denim, layered sportswear, bold accessories, volume hair",
        "1990s": "relaxed or mom jeans, slip dresses, flannel, cargos, layered tees, minimal makeup",
        "2000s": "ultra low-rise jeans and skirts, cropped and fitted tops, halters, velour tracksuits, trucker hats, frosted lips",
        "2010s": "skinny jeans, athleisure, fitted casual layers, contemporary tailoring, contoured makeup",
    }
    HISTORIC_CUES = {
        "Stone Age": "hide and fur garments, woven plant fibre, bone, tooth and shell ornaments, ochre body paint, "
            "stone and bone tools, fire, caves, hide tents or wooden huts",
        "Ancient Egypt": "pleated linen kilts and sheath dresses, broad beaded collars, gold and lapis jewellery, "
            "kohl-lined eyes, wigs, sandals, mudbrick and painted stone, oil lamps",
        "Classical Antiquity (Greece / Rome)": "tunics, chitons, peplos, togas and stolas, wool cloaks, sandals, "
            "bronze and gold jewellery, marble, frescoes, baths, markets, oil lamps",
        "Medieval": "layered wool and linen tunics, gowns and cloaks, hoods, veils and wimples, leather belts and "
            "boots, stone castles, timber villages, candles, torches and open hearths",
        "Feudal Japan": "kimono, hakama and obi by rank, tabi and geta, topknots or swept-up hair, wooden houses "
            "with shoji screens and tatami, paper lanterns",
        "Pre-Columbian Americas": "dyed cotton and wool tunics, wraps and capes, woven patterned textiles, feather, "
            "jade, shell or gold adornment, stone temples and terraces, markets, hearth fire",
        "Renaissance": "fitted bodices, slashed and puffed sleeves, velvet and brocade, doublets and hose, caps and "
            "hairnets, palazzos, workshops, candlelight",
        "18th Century (Baroque / Rococo)": "powdered wigs, frock coats and waistcoats, breeches, panniered gowns "
            "and stays, lace, buckled shoes, candlelit salons, carriages",
        "Victorian": "corsets, crinolines or bustles, frock coats, top hats and bonnets, high collars, gas lamps, "
            "cobbled streets, early industrial machines",
        "Belle Époque / Edwardian": "S-bend corseted silhouettes, large hats, high lace collars, three-piece suits "
            "and boaters, early motorcars, early electric light",
    }
    HISTORIC_NOTE = ("Era {era}: the story's world is set in that period. Every unlocked character wears what people "
                     "of their rank, age and region actually wore then (garments, materials, colours, hair, grooming, "
                     "adornment); options to consider, not a checklist: {cues}. Architecture, furniture, tools, light "
                     "sources, transport and objects belong to that period, with nothing invented later; materials "
                     "and silhouettes are faithful but lived-in and worn, not museum-clean. Spoken lines use the "
                     "selected language with no modern slang or references to things that did not exist yet. No "
                     "theme-party costume and no fantasy elements unless the genre or request asks. Visual looks only "
                     "set capture and colour; Keep Reference, written wardrobe and I2V facts win.")
    ERA_NOTE = ("Era {era}: the story's world is the {era}. Every unlocked character wears the mainstream fashion of "
                "that decade as people of their age and social world actually wore it (cut, rise, fit, fabrics, colours, "
                "hair, makeup, accessories); options to consider, not a checklist: {cues}. Give each character a "
                "different coherent outfit. Sets, furniture, vehicles, technology and consumer objects belong to that "
                "decade, with nothing invented later. No futurism, no theme-party costume and no subculture unless the "
                "request asks. Visual looks only set capture and colour; Keep Reference, written wardrobe and I2V facts win.")
    PRESENT_NOTE = ("Era Present day: contemporary world, current mainstream fashion, objects and technology; no "
                    "period styling unless the request asks.")
    NSFW_LADDER = ("None", "Flirt", "Sensual", "Heated", "Sex", "Hardcore", "Extreme")
    LEVEL_ZERO = ("NSFW None: no NSFW content yet: attraction, glances, flirting, closeness and touch over clothes; "
                  "everyone stays dressed.")

    @classmethod
    def nsfw_schedule(cls, curve, target, scenes):
        """Per-scene NSFW levels (0-4) for a progression curve, computed for any scene count."""
        import math
        top = cls.NSFW_LADDER.index(target) if target in cls.NSFW_LADDER else 0
        n = max(1, int(scenes))
        def at(i):
            f = i / (n - 1) if n > 1 else 1.0
            if curve == "slow":
                return int(math.floor(top * f ** 2.5 + 0.5))
            if curve == "gradual":
                return int(math.floor(1 + (top - 1) * f + 0.5)) if top > 1 else top
            if curve == "reverse":  # from Hardcore (or Extreme) down to the selected level, always the end
                peak = max(cls.NSFW_LADDER.index("Hardcore"), top)
                return int(math.floor(peak - (peak - top) * f + 0.5))
            return top  # normal, fast
        return [at(i) for i in range(n)]

    ARC_SHAPES = {
        "slow": "a slow burn: calm and contained for most of the time, the peak only at the very end",
        "normal": "one steady level of tension and energy from start to end",
        "gradual": "an even climb from calm to the peak, reached at the end",
        "fast": "up quickly, near the peak before the middle, and kept there to the end",
        "very fast": ("almost no lead-in: a 1-2 s start, the peak reached within about the first fifth of the time, then "
                      "kept, each beat topping the last"),
        "instant": ("straight to the point: open in the middle of the action already at the peak (no introduction, "
                    "arrival, greeting or build-up; the first frame is the action in progress) and sustain it to the "
                    "end, varying it inside the peak (a new position, angle, rhythm, contact or move) so it never "
                    "repeats one moment"),
        "reverse": "opens at the peak and settles step by step into a calm, resolved ending",
    }
    ARC_STEPS = ("calm", "low", "rising", "high", "peak")

    @classmethod
    def arc_note(cls, curve, direction_text, scenes):
        """Progression for any story, SFW or NSFW: the shape of the arc in time, applied to what is active."""
        curve = str(curve or "").strip().lower()
        if curve not in cls.ARC_SHAPES:
            return []
        text = str(direction_text or "")
        drives = ["the story's tension and energy (stakes, conflict or desire, pace of action and cutting, emotion "
                  "from contained to open, camera from calm and distant to close and alive, sound and music building)"]
        if re.search(r"(?im)^Violence:\s*\S", text):
            drives.append("the Violence level (from threat and small hits up to the selected level at the peak)")
        if re.search(r"(?im)^Intensity:\s*\S", text):
            drives.append("the NSFW heat (planned below)")
        n = max(1, int(scenes))
        if n == 1:
            plan = "shot by shot inside this clip"
        else:
            import math
            def step(i):
                f = i / (n - 1)
                if curve == "normal":
                    return "steady"
                value = {"slow": 4 * f ** 2.5, "gradual": 4 * f, "fast": 4 if i else 3, "very fast": 4,
                         "instant": 4, "reverse": 4 - 4 * f}[curve]
                label = cls.ARC_STEPS[int(math.floor(value + 0.5))]
                if curve == "instant":
                    return "peak from the first frame"
                if curve == "very fast" and i == 0:
                    return "peak within its first seconds"
                return label
            plan = "scene by scene: " + "; ".join(f"scene {i + 1} = {step(i)}" for i in range(n))
        return [f"PROGRESSION {curve.capitalize()} (binding, SFW or NSFW): {cls.ARC_SHAPES[curve]}; {plan}. It drives "
                + " and ".join(drives) + ". Each step is visible and earned by what happens (a new obstacle, a "
                "revelation, a closer contact, a decision), never announced; the clip or the sequence still ends "
                "complete. It changes how strongly things happen over time, never the cast, the place or the premise."
                + (" In Transfer it only shades the guide's own performance, sound and grading over time, without "
                   "changing its actions, cuts or timing." if _CLIP.get("mode") == "transfer" else "")]

    @classmethod
    def progression_note(cls, curve, target, scenes):
        if curve not in ("slow", "normal", "gradual", "fast", "very fast", "instant", "reverse") or target not in cls.NSFW_LADDER[1:]:
            return []
        n = max(1, int(scenes))
        if n == 1:
            within = {
                "slow": f"most of the clip stays at None or Flirt; {target} is reached only in the final shot",
                "normal": f"{target} from the first shot to the last",
                "gradual": f"rise shot by shot from Flirt to {target}, reached by the last shot",
                "fast": f"a brief lead-in, {target} reached before the middle of the clip and kept",
                "very fast": f"a 1-2 s lead-in at most, {target} reached within the first fifth of the clip and kept",
                "instant": (f"{target} from the very first frame: the clip opens with the act already in progress (no "
                            "approach, kissing first or undressing unless that is the level itself), sustained to the end "
                            "with changes of position, rhythm and angle"),
                "reverse": (f"open at Hardcore, then come down shot by shot to {target} by the last shot"
                            if target not in ("Hardcore", "Extreme") else f"{target} from the first shot to the last"),
            }[curve]
            plan = f"NSFW heat along the Progression {curve.capitalize()} (one scene, binding): {within}."
            used = {"slow": {0, 1}, "gradual": {1}, "reverse": {cls.NSFW_LADDER.index("Hardcore")} if target != "Extreme"
                    else set()}.get(curve, set())
        else:
            levels = cls.nsfw_schedule(curve, target, n)
            parts = []
            for i, lvl in enumerate(levels):
                label = cls.NSFW_LADDER[lvl]
                if curve == "slow" and i == n - 1:
                    label = f"builds to {target} only in its last shots"
                elif curve == "fast" and i == 0:
                    label = f"brief lead-in, {target} before its middle"
                elif curve == "very fast" and i == 0:
                    label = f"{target} within its first seconds"
                elif curve == "instant":
                    label = f"{target} from its first frame, already in progress"
                parts.append(f"scene {i + 1} = {label}")
            plan = (f"NSFW heat along the Progression {curve.capitalize()} (binding, one level per scene; the sequence always ends at "
                    f"the selected level, {target}): " + "; ".join(parts)
                    + ". Each scene stays at its own level as defined below, no earlier and no later; lower levels in "
                      "early scenes are the plan, not softening, and the change between scenes is visible and earned.")
            used = set(levels)
        notes = [plan]
        table = cls.CONTROL_SEMANTICS["Intensity"]
        for lvl in sorted(used):
            if lvl == 0:
                notes.append(cls.LEVEL_ZERO)
            elif cls.NSFW_LADDER[lvl] != target:
                notes.append(table[cls.NSFW_LADDER[lvl].lower()])
        return notes

    VERY_HIGH_NOTE = ("Action density Very High (overrides any motion-style wording about not adding events): the "
                      "most distinct physical beats the clip can hold, at an explosive pace (about 3-4 in 5 s, 6-8 in "
                      "10 s), each visibly different and physically possible in its time; never filler.")
    LANGUAGE_NOTE = ("Dialogue language {lang} (binding): every word inside <d> is written in {lang}, tagged [{lang}], "
                     "including short replies, names of things and exclamations; never English inside <d> unless the "
                     "language is English. Directions outside <d> stay in English.")
    GENRE_MECHANISMS = {
        **_Base.GENRE_MECHANISMS,
        "POV / First-Person": "The camera is the eyes of one character: everything is seen from their position and the "
            "others look into the lens and act toward the viewer. With an adult request the viewer is the intimate partner "
            "and the other performer's actions are directed at the lens.",
        "Cosplay / Roleplay": "Characters play a costumed persona or scenario (original, or the one the request names): "
            "costume, props and in-character behaviour drive the scene. With an adult request the costume and the role are "
            "part of the seduction and the reveal.",
        "Webcam / Cam Show": "One fixed camera in a personal space: the performer talks to unseen viewers, reacts to "
            "them and builds a show. With an adult request it is an adult cam show that escalates as the request asks.",
        "Boudoir / Glamour Shoot": "A styled photoshoot: deliberate poses, shaped soft light, confident presence toward "
            "the camera, outfit changes and a photographer's rhythm. With an adult request, lingerie or nudity in posed "
            "glamour as far as the request asks.",
        "Home Video / Amateur": "Recorded by the people themselves: handheld or propped phone, natural light, imperfect "
            "framing, spontaneous and personal behaviour. With an adult request, an intimate homemade recording as far "
            "as the request asks.",
        "Romance": "Attraction develops through a specific shared moment and lands in a closer intimacy or a clear "
            "choice; hesitation or obstacles only when the story wants them. Generic flirting alone is not an arc.",
        "Gothic Horror": "Let architecture, the past of the place and a specific forbidden threshold shape the dread; "
            "period detail and light support the threat instead of substituting for it.",
        "Slasher Horror": "Use pursuit geography, limited escape and a threatening reveal to create suspense; how much "
            "injury is shown follows the Violence level.",
        "Body Horror": "Develop an unsettling change in form, agency or bodily control with a clear cause and reaction; "
            "how explicit it is follows the Violence level.",
    }

    # ---- v34: Driven
    DRIVEN_PRIORITY_NOTE = (
        "DRIVEN PRIORITY (binding while a Driven input is connected): the Driven blueprint is the source of the "
        "scenario (activity, motion, staging, camera). Wardrobe, body and setting come from the SOURCE OF EACH "
        "ASPECT rules when they are given, otherwise from the observed Driven roles and setting; the craft notes "
        "about inventing a venue, an outfit or a palette only fill what is left open. Explicit requests and "
        "selected controls (Era, NSFW, Visual look) still apply on their own axes. {wardrobe}")
    DRIVEN_CAST_WITH_REFS = ("With character references, the people seen in Driven are never extra cast: each "
                             "prominent role is performed by a referenced person; CAST CREATION covers only "
                             "characters the user's request itself names.")
    DRIVEN_CAST_NO_REFS = ("No character references are connected: the prominent Driven roles ARE the principal cast. "
                           "Define each as a text-only adult subject with an invented face and hair that fit the "
                           "role (never a copied real face), carrying the observed wardrobe, body and pose; rules "
                           "about preserving reference identities do not apply, and this is not adding characters.")

    PLACEHOLDER_REQUEST = "Describe the exact edit and assign each connected source its role."

    @classmethod
    def empty_request_brief(cls, mode, has_driven, has_refs, direction_text, has_source=False):
        """The request to use when the user wrote none: Driven leads, otherwise the controls, otherwise the
        references, otherwise a free original story. Never an error."""
        mode = str(mode or "").strip().lower()
        controls = [line for line in str(direction_text or "").splitlines() if ":" in line and line.strip()]
        faithful = mode in ("transfer", "compact", "edit", "continuous edit")
        if has_driven:
            if faithful:
                return ("No written request. Transfer the Driven activity, motion and camera onto the connected "
                        "references (without references, its roles are the cast), with the Driven place as the base; "
                        "no new plot, people or events." + (" Apply every selected Direction Control fully and "
                        "visibly to that material (look, era, wardrobe, setting, mood...)." if controls else ""))
            return ("No written request. The Driven scenario is the premise: recast its activity with the connected "
                    "characters (without references, the Driven roles are the cast) and develop it into the best "
                    "story it supports" + (", realizing every selected Direction Control." if controls else "."))
        if controls and faithful and (has_refs or has_source):
            return ("No written request. Keep the connected sources' content and action, with no new events or "
                    "people, and apply every selected Direction Control fully and visibly to them (look, era, wardrobe, "
                    "setting, mood...).")
        if controls:
            return ("No written request. The selected Direction Controls are the complete brief: invent one original, "
                    "coherent concept that realizes every selected control"
                    + (", built around the connected references and only the people they show (add nobody)."
                       if has_refs else ", with characters you create."))
        if has_refs:
            if faithful:
                return ("No written edit request. Write a faithful, polished production prompt of the connected sources "
                        "as they are, improving only light, camera and motion clarity; change nothing else.")
            return ("No written request. Invent one original, coherent story built around the connected references "
                    "(their people, objects or places), in a tone that fits what they show.")
        return ("No written request and no sources. Invent one original, coherent, visually striking short story "
                "with characters you create.")

    @staticmethod
    def _render_transfer_cast(cast, blueprint):
        """Original renderer, tolerant of missing or malformed fields from weaker models."""
        subjects = []
        for index, person in enumerate(cast if isinstance(cast, list) else [], 1):
            if not isinstance(person, dict):
                continue
            get = lambda key: str(person.get(key) or "").strip()
            owner = get("wardrobe_source")
            if re.search(r"\b(?:nude|naked|topless|bottomless|unclothed|wearing nothing)\b", get("reference_wardrobe"),
                         re.IGNORECASE) and not re.search(r"\bno(?:t)? (?:nude|naked)|\bnot visible\b", get("reference_wardrobe"),
                                                          re.IGNORECASE):
                _CLIP["reference_nude"] = True
            if _CLIP.get("wardrobe_follow") and get("requested_wardrobe").lower() not in ("", "n/a"):
                owner = "direction_controls"  # guide garments restyled in the reference style
            if (_CLIP.get("task") == "add" and owner == "performance" and get("identity_source") not in ("", "<Picture 1>")
                    and get("reference_wardrobe").lower() not in ("", "n/a")):
                owner = "reference"  # Add: a new person keeps their own outfit, not the video person's
            if (_CLIP.get("task") == "edit" and _CLIP.get("wardrobe_keep")
                    and get("requested_wardrobe").lower() not in ("", "n/a")):
                owner = "direction_controls"  # Edit: the outfit of Picture 2 onward replaces the video's
            elif (_CLIP.get("wardrobe_keep") and owner not in ("user_request", "reference")
                    and get("reference_wardrobe").lower() not in ("", "n/a")):
                owner = "reference"  # Keep Reference wins over the performer's clothes, never over the written request
            if owner == "reference":
                wardrobe = get("reference_wardrobe")
                if _CLIP.get("restyle_outfit"):  # the reference clothes, in the selected cut
                    wardrobe = (get("requested_wardrobe") if get("requested_wardrobe").lower() not in ("", "n/a")
                                else f"{wardrobe}, restyled to {_CLIP['restyle_outfit']} (same colours and fabrics, "
                                "new cut)")
            elif owner == "performance":
                wardrobe = get("performance_wardrobe")
                if get("performance_source").startswith("<Driven "):
                    role = next((r for r in (blueprint or {}).get("roles", [])
                                 if isinstance(r, dict) and r.get("role") == get("performance_role")), None)
                    if role is not None and str(role.get("wardrobe") or "").strip():
                        wardrobe = str(role["wardrobe"]).strip()
            else:
                wardrobe = get("requested_wardrobe")
            tag = get("subject_tag") or f"<Subject {index}>"
            number = re.search(r"\d+", tag)
            tag = f"<Subject {number.group(0) if number else index}>"
            source = get("identity_source") or f"<Picture {index}>"
            if _CLIP.get("task") == "edit":
                source = "<Picture 1>"  # Edit: every person is a person of the video
            definition = tag + " " + get("identity_details") + " Identity referenced from " + source + "."
            if wardrobe.lower() not in ("", "n/a"):
                definition += " Wardrobe: " + wardrobe.rstrip(".") + "."
            subjects.append(re.sub(r"\s{2,}", " ", definition))
        return "\n".join(subjects)

    @staticmethod
    def _render_transfer_scene(scene, source_tags):
        """Original renderer, tolerant: an unknown source video falls back to the first connected one."""
        scene = scene if isinstance(scene, dict) else {"detailed_description": str(scene)}
        get = lambda key: str(scene.get(key) or "").strip()
        source = get("source_video")
        if source not in source_tags:
            source = source_tags[0] if source_tags else source
        instruction = get("edit_instruction")
        performance = get("detailed_description")
        subjects = get("subject_definitions") or "N/A"
        if source and source not in subjects:
            subjects += "\n" + source + " is the source video for the target video edit."
        task = "[video editing + reference generation]" if re.search(
            r"<Picture\s+\d+>", subjects + " " + instruction) else "[video editing]"
        return "\n\n".join((
            "subject_definitions:\n" + subjects,
            "summary:\n" + task + " The target video is an edited version of " + source + ". " + instruction,
            "retention_analysis:\n" + (get("retention_analysis") or "N/A"),
            "detailed_description:\n" + performance,
            "overall_soundscape:\n" + (get("overall_soundscape") or "N/A"),
            "non_diegetic_music:\n" + (get("non_diegetic_music") or "N/A"),
        ))

    @staticmethod
    def _axis_choice(direction_text, label):
        match = re.search(rf"(?im)^{label}:\s*(.+)$", str(direction_text or ""))
        value = match.group(1).strip().lower() if match else ""
        return ("guide" if value.startswith("keep guide") else "follow" if value.startswith("guide garments")
                else "keep" if value.startswith("keep")
                else "new" if value.startswith(("create", "creative")) else "auto")

    @classmethod
    def source_axes(cls, direction_text):
        """(effective Driven Strength, binding per-aspect note) from the Wardrobe/Body/Environment controls."""
        wardrobe = cls._axis_choice(direction_text, "Wardrobe")
        body = cls._axis_choice(direction_text, "Body proportions")
        place = cls._axis_choice(direction_text, "Environment")
        hair = cls._axis_choice(direction_text, "Hair & Makeup")
        if wardrobe == "follow" and _CLIP.get("mode") != "transfer":
            wardrobe = "keep"  # outside Transfer there is no guide performance to follow
        strength = 1 if wardrobe in ("keep", "follow") else 2 if body == "keep" else 3
        text = {
            "wardrobe": {"guide": "Wardrobe: exactly the clothes of the mapped role in the guide (garments, colours, fit, "
                                  "accessories), kept as they are even against an era, genre or look.",
                         "keep": "Wardrobe: exactly the clothing state of each character's own reference picture (its outfit, or nude if it is nude), not the performer's.",
                         "follow": "Wardrobe: the same garments as the mapped guide person (pieces, layers, cut, length, "
                                   "fit, how they move and come off), restyled in the style of the character's own "
                                   "reference picture (colours, fabrics, patterns, finish, accessories).",
                         "new": "Wardrobe: newly designed; neither the reference outfit nor the performer's clothes.",
                         "auto": "Wardrobe: the clothes of the mapped role in the video/Driven source."},
            "body": {"guide": "Body: exactly the body of the mapped role in the guide (height, weight, bust or chest, waist, "
                              "hips, buttocks, limbs, muscle tone), on the reference face; never idealized.",
                     "keep": "Body: exactly the body of each reference picture (height, weight, bust or chest, waist, hips, "
                             "buttocks, limbs, muscle tone), not the performer's body; never idealized.",
                     "new": "Body: a coherent adapted adult build, written explicitly.",
                     "auto": "Body: the build of the mapped role in the source, on the reference face."},
            "place": {"guide": "Setting: exactly the guide's location (layout, furniture, surfaces, light), kept as it is "
                               "even against an era, genre or look.",
                      "keep": "Setting: the place visible in the primary reference picture; the source's location is NOT "
                              "used, and the observed action is adapted to that place and its own furniture and surfaces.",
                      "new": "Setting: a newly designed place; neither the reference background nor the source location.",
                      "auto": "Setting: the source's location."},
        }
        hair_text = {"keep": "Hair and makeup: those of each character's own reference picture.",
                     "guide": "Hair and makeup: exactly those of the mapped role in the guide.",
                     "new": "Hair and makeup: newly designed from the request and controls; not the reference's nor the guide's.",
                     "auto": ""}
        if _CLIP.get("mode") == "transfer":
            hair_text["auto"] = ("Hair and makeup: free; restyled by the request and active controls (era, genre, NSFW, "
                                 "look), from the reference or the guide as a base.")
        if _CLIP.get("mode") == "transfer":  # Transfer: Auto is the director's free choice; Keep Guide keeps the guide
            text["wardrobe"]["auto"] = ("Wardrobe: free; choose what best serves the request and the active controls "
                                        "(the guide's clothes restyled, the reference outfit or a new design).")
            text["body"]["auto"] = "Body: free; the reference build or the guide's build, whichever fits the result best."
            text["place"]["auto"] = ("Setting: free; the guide's place transformed by the controls, or another place that "
                                     "serves the request better, keeping the action feasible.")
        if _CLIP.get("task"):  # Edit / Add / Remove: what no control sets stays exactly as the guide shows it
            for key in text:
                text[key]["auto"] = text[key]["guide"]
            hair_text["auto"] = hair_text["guide"]
        if _CLIP.get("task") == "add":  # the video's people stay as they are; added people bring their own look
            text["wardrobe"]["auto"] = ("Wardrobe: the video's people keep exactly their clothes in the guide; each "
                                        "added person wears the outfit of their own picture.")
            text["body"]["auto"] = ("Body: the video's people keep their build; each added person keeps the build of "
                                    "their own picture.")
            hair_text["auto"] = ("Hair and makeup: the video's people keep theirs; each added person keeps those of "
                                 "their own picture.")
        if _CLIP.get("task") == "edit":  # Picture 1 is the video: 'reference' means Picture 2 onward
            text["wardrobe"]["keep"] = (
                "Wardrobe: the outfit shown in <Picture 2> onward, worn by the video's people in place of theirs and "
                "going through the video's own clothing changes (same moments, order and final coverage).")
            text["wardrobe"]["follow"] = ("Wardrobe: the video's own garments, restyled in the style of <Picture 2> "
                                          "onward (colours, fabrics, patterns, accessories).")
            text["body"]["keep"] = "Body: the build shown in <Picture 2> onward, on the video's people."
            text["place"]["keep"] = "Setting: the place shown in <Picture 2> onward, with the video's action adapted to it."
            hair_text["keep"] = "Hair and makeup: those shown in <Picture 2> onward, on the video's people."
        note = ("SOURCE OF EACH ASPECT (binding): face and identity always come "
                "from the reference pictures; motion, action and choreography always come from the video/Driven source. "
                + text["wardrobe"][wardrobe] + " " + text["body"][body] + " " + text["place"][place]
                + (" " + hair_text[hair] if hair_text[hair] else ""))
        return strength, note

    @classmethod
    def driven_note(cls, strength, has_refs):
        try:
            strength = int(strength)
        except (TypeError, ValueError):
            strength = 3
        wardrobe = "" if has_refs else "Without references the observed styling of each role is used. "
        return cls.DRIVEN_PRIORITY_NOTE.format(wardrobe=wardrobe) + (
            cls.DRIVEN_CAST_WITH_REFS if has_refs else cls.DRIVEN_CAST_NO_REFS)

    _DRIVEN_FIELDS = ("activity", "setting", "lighting", "visual_style", "mood", "camera", "props")
    _ROLE_FIELDS = ("role", "wardrobe", "hair", "body_proportions", "pose", "performance", "spatial_relationships")

    @classmethod
    def _normalize_blueprint(cls, data):
        """Coerce any parsed blueprint into the exact shape the director expects; never raises."""
        data = data if isinstance(data, dict) else {}
        blueprint = {field: str(data.get(field) or "") for field in cls._DRIVEN_FIELDS}
        roles = []
        for index, role in enumerate(data.get("roles") if isinstance(data.get("roles"), list) else []):
            if not isinstance(role, dict):
                continue
            entry = {field: str(role.get(field) or "") for field in cls._ROLE_FIELDS}
            entry["role"] = entry["role"].strip() or f"person {index + 1}"
            roles.append(entry)
        seen = {}
        for entry in roles:  # role names must be unique for the cast mapping
            count = seen.get(entry["role"], 0)
            seen[entry["role"]] = count + 1
            if count:
                entry["role"] = f"{entry['role']} {count + 1}"
        blueprint["roles"] = roles
        return blueprint

    PARTIAL_ANALYSIS_NOTE = (
        "GENDER OF EACH ROLE (required; this is not identity): every role field starts with the person's visible "
        "gender presentation (woman, man, or unclear) so that nobody is recast as another gender. "
        "PARTIALLY VISIBLE PEOPLE: a person who is in the scene but only partly in frame (hands, arms, legs, torso, a "
        "body cut by the frame edge, face never seen) is still a role, but its role field must start with "
        "'PARTIAL:' and name exactly the visible parts and the frame edge they enter from; never describe a face, hair or full figure "
        "you did not see. EMOTIONAL BEATS: in each role's performance, record in order how expression, energy and "
        "attitude change across the footage, as seen in face and body.")
    PARTIAL_RE = re.compile(r"^\s*PARTIAL\b|(?:face|head) (?:is )?(?:never |not )"
                            r"(?:visible|shown|seen|in frame)|only (?:his|her|their|the) (?:hands?|arms?|legs?|feet|torso)",
                            re.IGNORECASE)
    @classmethod
    def _describe_driven(cls, llm, driver_content, request_text, max_tokens, seed):
        """Driven analysis that never fails on a malformed reply: lenient parsing, one retry, then an empty
        blueprint (the director continues from the request and references)."""
        from . import story_director as _sd
        cached = _CLIP.get("blueprint_cache")
        if cached is not None:  # retries inside one execution reuse the analysis: no second paid call
            return cls._mask_blueprint(json.loads(json.dumps(cached[0]))), cached[1], cached[2]
        original = _Base._describe_driven.__func__
        captured = {}

        def fake_request(llm_model, payload):
            captured["payload"] = payload
            raise _DrivenCapture()
        _install_provider_dispatch()
        real_request = _sd._external_llm_request  # the dispatcher: the running execution's hook still applies
        token = _HOOK_VAR.set(fake_request)
        try:
            original(cls, llm, driver_content, request_text, max_tokens, seed)
        except _DrivenCapture:
            pass
        finally:
            _HOOK_VAR.reset(token)
        payload = captured["payload"]
        payload["max_tokens"] = max(900, min(1800, int(max_tokens)))
        if _CLIP.get("mode") == "transfer" and payload.get("messages") and isinstance(payload["messages"][0].get("content"), str):
            payload["messages"][0]["content"] += "\n\n" + cls.PARTIAL_ANALYSIS_NOTE
        usage, last_error = {}, None
        for attempt in range(2):
            result = real_request(llm, payload)
            usage = result.get("usage") or {} if isinstance(result, dict) else {}
            try:
                raw = _sd._completion_content(result, payload, llm, "Driven visual analysis")
            except RuntimeError as error:  # empty reply or refusal: retry once, then continue without it
                if "no usable content" not in str(error) and "no completion choices" not in str(error):
                    raise
                last_error = error
                print("[H3 Director] Driven analysis returned no content; retrying once")
                continue
            if not isinstance(raw, str):
                return cls._normalize_blueprint(raw), usage, payload
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
            start = raw.find("{")
            body = raw[start:] if start >= 0 else raw
            for candidate in [body, *cls._json_candidates(body)]:
                try:
                    data = json.loads(candidate, strict=False)
                except ValueError as error:
                    last_error = error
                    continue
                if isinstance(data, dict) and any(v not in (None, "", [], {}, "N/A") for v in data.values()):
                    _CLIP["driven_note"] = "" if attempt == 0 else " · guide analysis retried"
                    normalized = cls._normalize_blueprint(data)
                    _CLIP["blueprint_cache"] = (json.loads(json.dumps(normalized)), usage, payload)
                    return cls._mask_blueprint(normalized), usage, payload
            payload = dict(payload, max_tokens=min(2400, int(payload["max_tokens"] * 1.5)))
            print(f"[H3 Director] Driven analysis unreadable ({last_error}); retrying once")
        print("[H3 Director] Driven analysis unusable twice; continuing from the request and references")
        _CLIP["driven_note"] = (" · ⚠ guide analysis unusable: movement fidelity lost, scenario from the request, "
                                "references and controls")
        _CLIP["blueprint_cache"] = (cls._normalize_blueprint({}), usage, payload)  # retries do not pay for it again
        return cls._mask_blueprint(cls._normalize_blueprint({})), usage, payload

    @staticmethod
    def _mask_blueprint(blueprint):
        """Remove from the Driven blueprint every aspect a Direction Control assigns elsewhere, so the director
        cannot copy the source's setting, clothes or body against the user's choice."""
        flags = _CLIP.get("mask", {})
        kept = flags.get("keep_guide") or {}
        cls_partial = H3CompactMultimodalEditDirectorFix.PARTIAL_RE.search
        if flags.get("setting"):
            for field in ("setting", "lighting", "props"):
                blueprint[field] = "(not used: the Environment control decides the setting)"
        if flags.get("restyle") is not None:  # Transfer: the source is the base, every active choice re-renders it
            chosen = "; ".join(flags["restyle"]) or "none (only the user's request)"
            blueprint["transformation"] = (
                "The fields above are the SOURCE BASE: keep its action, kind of place, spatial layout, framing and camera, "
                "but re-render every visual attribute (setting, props, lighting, wardrobe, hair, makeup, style, mood, "
                "sound) through the user's request and these active Direction Controls: " + chosen + ". Each choice "
                "changes everything it naturally touches, in every scene, never partially; fields marked KEEP GUIDE "
                "stay exactly as observed.")
        if kept.get("Environment"):
            for field in ("setting", "props", "lighting"):
                if str(blueprint.get(field, "")).strip():
                    blueprint[field] = f"KEEP GUIDE (exactly as observed): {blueprint[field]}"
        if flags.get("look"):
            blueprint["visual_style"] = "(not used: the selected Visual look decides the image style)"
        if flags.get("mood"):
            blueprint["mood"] = "(not used: the selected Genre decides the mood)"
        follows_changes = _CLIP.get("wardrobe_follow") or (_CLIP.get("task") == "edit" and _CLIP.get("wardrobe_keep"))
        if flags.get("wardrobe") and _CLIP.get("mode") == "transfer" and not follows_changes:  # no source nudity leak
            nude = re.compile(r"\b(?:completely |fully |partially |partly )?(?:nude|naked|topless|bottomless|unclothed|"
                              r"undressed|bare[- ]chested|in the nude|without (?:any )?cloth(?:es|ing))\b,?\s*", re.IGNORECASE)
            for field in ("activity", "mood", "visual_style", "props"):
                blueprint[field] = nude.sub("", str(blueprint.get(field, "")))
            for role in blueprint.get("roles", []):
                for field in ("role", "pose", "performance", "spatial_relationships"):
                    role[field] = nude.sub("", str(role.get(field, "")))
        if _CLIP.get("mode") == "transfer":  # people seen only in part are marked so they stay that way
            partial = [role for role in blueprint.get("roles", [])
                       if cls_partial(" ".join(str(role.get(k, "")) for k in ("role", "pose", "spatial_relationships")))]
            for role in partial:
                role["visibility"] = ("PARTIAL ONLY: keep exactly these visible body parts and frame side; no face, "
                                      "no full figure, no shot that reveals this person")
        for role in blueprint.get("roles", []):
            for label, field in (("Wardrobe", "wardrobe"), ("Body proportions", "body_proportions"), ("Hair & Makeup", "hair")):
                if kept.get(label) and str(role.get(field, "")).strip():
                    role[field] = f"KEEP GUIDE (exactly as observed): {role[field]}"
            if (_CLIP.get("wardrobe_follow") and _CLIP.get("mode") == "transfer" and str(role.get("wardrobe", "")).strip()
                    and not str(role.get("wardrobe", "")).startswith(("KEEP GUIDE", "GARMENTS TO REPLICATE"))):
                role["wardrobe"] = ("GARMENTS TO REPLICATE (same pieces, cut and behaviour, restyled in the reference "
                                    "style): " + str(role["wardrobe"]).strip())
            elif flags.get("wardrobe") and (_CLIP.get("task") == "edit" and _CLIP.get("wardrobe_keep")) and _CLIP.get(
                    "mode") == "transfer" and str(role.get("wardrobe", "")).strip():
                role["wardrobe"] = ("CLOTHING CHANGES ONLY (the outfit itself is the reference picture's; follow when "
                                    "and how these clothes are opened, removed or put on): " + str(role["wardrobe"]).strip())
            elif flags.get("wardrobe"):
                role["wardrobe"] = "(not used: the Wardrobe control decides the outfit)"
                if flags.get("hair") == "legacy":
                    role["hair"] = "(not used: the reference hair is kept)"
            if flags.get("hair") == "control":
                role["hair"] = "(not used: the Hair & Makeup control decides hair and makeup)"
            if flags.get("body"):
                role["body_proportions"] = "(not used: the Body Proportions control decides the build)"
        return blueprint

    VIDEO_TASK_BASE = (
        "VIDEO TASK {task} (binding, above the other Transfer rules): <Picture 1> is a frame of the guide video itself, "
        "not a new identity. Every person of the guide stays the same person (face, body, hair, outfit, gender) with "
        "identity_source <Picture 1>, identity_details saying which person of that frame it is (position, look), and "
        "driven_cast_mapping maps every guide role to <Picture 1>. The guide video is the base: its timing, action, "
        "camera, framing, place, light and sound stay exactly as they are. 'Transfer is a transformation' does not "
        "apply: only what this task, the user's words and the selected controls change is changed, exactly where it "
        "is, and every aspect no control sets stays as the guide shows it.")
    VIDEO_TASK_NOTES = {
        "edit": "EDIT: change only what is asked (clothes, hair, makeup, colours, an object, the light, the place...) on "
                "the people and things of the video. A <Picture 2>, <Picture 3> or <Picture 4> is the source for that "
                "change (a garment, accessory, hairstyle, object, place or style to copy), never a new person, unless "
                "the words say otherwise; it gets no transfer_cast entry and is named where it is used (for example "
                "'the dress from <Picture 2>'). A person shown in such a picture lends only that aspect, never their "
                "face, skin tone, hair, body or identity (unless a Keep Reference control takes hair or build from it): "
                "every Subject is a person of the video, with identity referenced from <Picture 1>, and the person "
                "in <Picture 2> onward never gets a Subject tag.",
        "add": "ADD: everyone in the video stays and keeps their own action. Each person in <Picture 2> onward joins the "
               "scene as a new person (their own identity, and their own outfit unless the words or Wardrobe say "
               "otherwise), placed where the words say, otherwise in free space of the frame, at coherent scale, light, "
               "shadow and contact. They get their own action that fits the scene and its rhythm, never a mirror copy "
               "of someone, and interact only as the words ask. Each added person has a transfer_cast entry with their "
               "Picture as identity_source; their performance_role is only a schema formality (the nearest role) and "
               "never gives them that person's place, clothes, body or action: their own action is written in the "
               "description, and their wardrobe_source is 'reference' unless the words or a control dress them "
               "otherwise. The people of the video keep identity_source <Picture 1> and never take the look of an added "
               "person. A Picture that shows an object adds that object instead.",
        "remove": "REMOVE: the person or thing the words name is gone from every shot, with no trace, shadow, reflection "
                  "or voice, and the area behind it continues the place naturally; nobody else changes. A removed "
                  "person gets no transfer_cast entry and no Subject tag; whoever touched, looked at or spoke to them "
                  "adapts that moment to the empty space (looks away, gestures alone) without new events or people. "
                  "If the words do not say what to remove, remove nothing.",
    }

    EDIT_KEEP_ASPECTS = {"Wardrobe": "outfit (garments, colours, accessories)", "Hair & Makeup": "hairstyle, hair colour and makeup",
                    "Body proportions": "build and proportions", "Environment": "place"}
    EDIT_KEEP_NOTE = (
        "{label} Keep Reference in Video task Edit (binding): <Picture 1> is the video itself, so the reference is "
        "<Picture 2> onward: the {what} shown there is applied to the video's people (each Picture to the person the "
        "words assign it to; otherwise the main person, or everyone when it fits), replacing theirs in every shot, "
        "while their identity, action and everything else stay the video's. In transfer_cast, wardrobe_source is "
        "'direction_controls' and requested_wardrobe describes that outfit concretely, naming its Picture.")
    EDIT_WARDROBE_CHANGES = (
        "The video's own clothing changes still happen with the new outfit: whatever the person opens, takes off or "
        "puts on in the video, they do with the equivalent garment of that outfit, at the same moments and in the "
        "same order, ending just as covered or uncovered as in the video (down to underwear or nudity when the video "
        "goes there); when the outfit has no such garment, the gesture uncovers nothing the video does not. The "
        "user's own words still win.")

    LORA_NOTE = (
        "SELECTED LoRA {names} (binding): the video model has a LoRA loaded for this, and its name says what it is "
        "for (an action, position, body detail, movement or style). It is a point the story must reach, not the whole "
        "story: tell an interesting story from the request, the characters and the controls (a situation, intent, "
        "tension, a development) that arrives at that action, at the start, in the middle or toward the end as the "
        "story and the request call for, early enough that it holds a real stretch of screen time (about a "
        "third of the footage or more). Before it, the scene has a life of its own; once reached, the action is "
        "written plainly and concretely with the words its name implies (the position, the body parts involved, the "
        "motion and its rhythm, the camera angle that shows it) so the LoRA engages, and the story keeps moving "
        "through it (reactions, changes, a turn or a closing state). With an empty or open request the story is built "
        "toward it between the people present. It never adds people: an act that needs a partner uses one only when "
        "the request or the references provide one; otherwise it is adapted to the people present. Its content "
        "sets the heat it needs (the request can set it lower or higher), between consenting "
        "adults. In Transfer the guide's choreography stays: the LoRA only sharpens how the matching action is "
        "written.")
    # a match starts where a word starts (file names use _ - . and spaces), so Floral, Analog or Document never match
    _LORA_EXPLICIT = re.compile(r"(?i)(?<![a-z])(?:doggy|cowgirl|missionary|blow ?job|fellatio|cunnilingus|"
                                r"oral(?![a-z])|anal(?![a-z])|sex(?![a-z])|fuck|foreskin|penetrat|cum(?:shot)?(?![a-z])|"
                                r"creampie|hand ?job|foot ?job|tit ?job|boob ?job|paizuri|(?<![0-9])69(?![0-9])|deep ?throat|squirt|"
                                r"masturbat|fingering|riding|spooning|scissoring|threesome|gangbang|bukkake|orgasm|pegging|"
                                r"fisting|rimjob|lick(?!ing lips)|dildo|wank|pussy|penis|vagina|bone(?![a-z])|ahegao|suck)")
    _LORA_STYLE = re.compile(r"(?i)\bstyle\b|\blook\b|cinematic|cinema|grade\b|\bvhs\b|minidv|television|\banime\b|"
                             r"pixel|\bps2\b|animation|digital art")
    LORA_STYLE_NOTE = (
        "SELECTED STYLE LoRA {names} (binding): the video model has a LoRA loaded for this visual style. Apply it like "
        "a Visual look to all the footage: the capture, texture, palette, lighting and rendering its name describes, "
        "concrete in the first sentence of every scene and consistent throughout. It changes how the footage looks, "
        "not the story, the actions, the people, the era or the heat (a period film style does not move the "
        "story to that period unless Era or the request does); it blends with a selected Visual look and wins over "
        "it where they clash.")

    @classmethod
    def lora_is_style(cls, name):  # 'VHS 80s Style' is a look; 'Doggy Style' is an act
        return bool(cls._LORA_STYLE.search(name)) and not cls._LORA_EXPLICIT.search(name)

    @staticmethod
    def lora_names(direction_text):
        return [m.split(" | ")[0].strip() for m in re.findall(r"(?im)^LoRA:\s*(.+)$", str(direction_text or ""))
                if m.split(" | ")[0].strip()]

    @staticmethod
    def lora_notes(direction_text):
        """{name: note} for LoRAs that carry a description ('LoRA: Name | what it shows')."""
        return {name.strip(): note.strip() for name, note in
                re.findall(r"(?im)^LoRA:\s*([^|\n]+?)\s*\|\s*(.+)$", str(direction_text or "")) if note.strip()}

    EXTRA_SLOT = "<Independent extra>"

    @classmethod
    def _extra_slot(cls, role_names, pictures):
        """Transfer with more source people than pictures: the remaining people stay separate supporting people."""
        return ([cls.EXTRA_SLOT] if _CLIP.get("mode") == "transfer" and not _CLIP.get("task")
                and len(role_names) > len(pictures) else [])

    @classmethod
    def _fix_driven_mapping(cls, mapping, role_names, pictures):
        """Repair the role -> Picture map instead of failing: keep valid entries, assign missing roles; in Transfer a
        picture is never given to two roles (the extra people stay independent)."""
        sources = [f"<Picture {index}>" for index, _ in pictures]
        if _CLIP.get("task") and "<Picture 1>" in sources:  # Edit / Add / Remove: the guide people are Picture 1
            return [{"driver_role": role, "identity_source": "<Picture 1>"} for role in role_names]
        extra = cls._extra_slot(role_names, pictures)
        fixed, used_roles, taken = [], set(), []
        for entry in mapping if isinstance(mapping, list) else []:
            if (isinstance(entry, dict) and entry.get("driver_role") in role_names
                    and entry.get("driver_role") not in used_roles and entry.get("identity_source") in sources + extra):
                source = entry["identity_source"]
                if extra and source in taken:
                    source = extra[0]  # the same identity on a second person would clone the lead
                fixed.append({"driver_role": entry["driver_role"], "identity_source": source})
                used_roles.add(entry["driver_role"])
                taken.append(source)
        for role in role_names:
            if role in used_roles or not sources:
                continue
            free = [source for source in sources if source not in taken]
            source = free[0] if free else (extra[0] if extra else sources[len(fixed) % len(sources)])
            fixed.append({"driver_role": role, "identity_source": source})
            taken.append(source)
        return fixed

    CONTROL_SEMANTICS = {
        "Intensity": {
            "flirt": "NSFW Flirt (erotic intensity 1 of 6, innocent): playful, coy charm: glances, smiles, light teasing words, cute or shy poses, a little closeness; nothing sexual.",
            "sensual": "NSFW Sensual (2 of 6): slow, warm sensuality: lingering looks, soft suggestive words, sensual poses and slow body movement, caressing (one's own body, or a partner's when there is one), kissing when there is a partner; no sex acts.",
            "heated": "NSFW Heated (3 of 6): bold, aggressive desire: provocative poses, hip rolls and grinding, hands gripping and groping (one's own body or a partner's), lip biting, breathless dirty talk, undressing play as far as the Nudity rule allows; no penetration or oral acts.",
            "sex": "NSFW Sex (4 of 6): openly sexual: movements and poses that are sexual in themselves (riding and thrusting motions, spread and arched poses), touching and self-pleasure, sex acts when there is a partner, moaning and explicit sexual talk, written in direct words.",
            "hardcore": "NSFW Hardcore (5 of 6): raw and explicit: explicit sexual action sustained with no softening, vulgar dirty talk, hard rhythm and strong physical response (sweat, trembling, loud moans), nothing implied, faded or cut away.",
            "extreme": "NSFW Extreme (6 of 6): total release: filthy, vulgar and dominant, a dark and transgressive edge in attitude, words and movement, every moment charged at maximum intensity, the most extreme poses and rhythm, raw unfiltered dialogue. Intensity, never cruelty: every adult visibly consents and wants it; no coercion; pain or harm only when the request asks for it.",
        },
        "Violence": {
            "mild": "Violence Mild: conflict and danger are real but violence happens mostly off-screen or in quick, bloodless moments.",
            "moderate": "Violence Moderate: fights and injuries are visible on screen, with little blood and no gore.",
            "strong": "Violence Strong: wounds and blood are shown plainly; impacts land and their consequences stay on screen.",
            "graphic": "Violence Graphic: gore, wounds and impact shown in close, sustained detail, no cut-away.",
            "extreme": "Violence Extreme: extreme gore and body horror in close, sustained detail (open wounds, dismemberment, viscera) with relentless dread and no cut-away.",
        },
        "Dialogue density": {
            "sparse": "Dialogue density Sparse: a few short lines in total; scenes without speech are fine.",
            "balanced": "Dialogue density Balanced: each lead speaks once or twice per scene.",
            "dense": "Dialogue density Dense: many short turns that answer each other, never long speeches; speech still fits about 2.5 words per second with room for action.",
        },
    }
    ANATOMY_NOTE = (
        "ANATOMY FIDELITY (video models render unnamed intimate anatomy as smooth, doll-like or a vague bulge): "
        "in every shot where a nude or partly nude body is visible, name the intimate anatomy explicitly and "
        "anatomically in that shot's footage text, with its state, position toward the camera and any contact: "
        "breasts with nipples and areolae; for women the vulva and labia, with pubic hair or bare skin; for men "
        "the penis (flaccid or erect) and scrotum/testicles, with pubic hair; buttocks and anus when the angle "
        "shows them. Restate it in wide shots, during movement, position changes and after every cut, not only "
        "in close-ups; in a wide or full shot give it at the scale of the framing, small but correctly placed and "
        "anatomically correct, never a blur, a lump or an extra shape. When only part of a body is bare, name the bare "
        "anatomy and say what still covers the rest. Never leave a nude body undescribed between the legs or call it "
        "smooth, featureless or implied. Keep each character's stable details (pubic hair style, size, skin tone) identical across shots "
        "and scenes by writing them in that character's target appearance when nudity is part of the story. This "
        "applies to EVERY person whose clothing leaves the genitals or breasts exposed (pants or underwear pulled "
        "down, an open fly, a top pulled aside), even when the action focuses on "
        "someone else: name what is exposed in the shots that show it; if the action does not need that person "
        "exposed, keep their clothing on instead of leaving it half-down and undescribed.")
    COVERED_ANATOMY_NOTE = (
        "COVERED ANATOMY (video models invent bulges, nipples or genital shapes under clothing when it is left "
        "undescribed, above all during movement): while an intimate area is covered (clothing, underwear, swimwear, a "
        "towel, a sheet, hands, hair or the angle), say in that shot what covers it and how it sits: breasts as natural "
        "contours under the named garment, nipples not showing unless the request and the fabric (sheer, wet, thin) "
        "show them; the crotch of women and men as the garment's smooth, natural fit or drape, with no bulge or "
        "outline unless the request and the story call for one, and then described exactly. Restate the covering when "
        "bodies bend, straddle, kneel, lie down, stretch or press together, and after every cut, so it stays in place "
        "until the footage itself removes it; a garment coming off is an on-screen action, and what it uncovers is "
        "then named exactly.")
    _ADULT_REQUEST = re.compile(
        r"\bnsfw\b|\+\s?18|\b18\+|\bxxx\b|er[oó]tic|\bporn|\bsex\b|\bsexo\b(?!\s*(?:masculino|femenino))|"
        r"sexual|cachond|hard ?core|\bfollar|\bcoger\b|expl[ií]cit|desnud|\bnude\b|\bnaked\b|"
        r"(?:muy|bien|super) caliente", re.IGNORECASE)
    _SUGGESTIVE_REQUEST = re.compile(
        r"\bsexy\b|sensual|seduc|picante|coquet|provocativ|atrevid|sugerente|subid[oa] de tono|"
        r"(?:algo|historia|escena) caliente|\bhot\b|\bsteamy\b|\bflirt", re.IGNORECASE)
    SUGGESTIVE_NOTE = (
        "SUGGESTIVE REQUEST (words like 'sexy', 'sensual', 'seductive'): play it sensual and flirtatious through "
        "confident movement, looks, attitude and touch over the clothes. Clothing stays fully on and in place: "
        "underwear is always worn, and nothing intimate is exposed (no nipples, genitals or bare buttocks, no framing "
        "under clothing); no sex acts. It goes further only when the request says so.")
    ADULT_FREE_NOTE = (
        "ADULT REQUEST (the words ask for adult content): you choose how far it goes, as the story and the request's "
        "tone call for, from seduction to fully explicit sex; whatever the words name (nudity, an act, a garment off) "
        "happens exactly as named. Every character is a consenting adult (18+) and consent stays visible. Clothes come "
        "off on screen when the story gets there, never from the first frame unless the request says so.")
    REQUEST_READING_NOTE = (
        "READING THE REQUEST (binding): the user's words are the only controls for heat, skin, pace, camera, cuts, "
        "progression and violence. Translate them faithfully, neither weaker nor stronger: 'desnuda/naked' means "
        "naked; 'sexo/sex' means explicit sex; 'rápido/fast' "
        "means more story in less time and earlier payoff; 'un solo plano/one shot' means no cuts; 'que vaya subiendo' "
        "means a rising progression; 'sexy/sensual' means flirtatious, caresses and tension with the clothes on; 'violento' means visible violence. Violence never happens during intimacy: no struggle "
        "for control, pinned wrists, marks or injuries between intimate partners unless the request asks for consensual "
        "rough play. Where the words say nothing, decide what "
        "serves this story best; with no erotic words at all, the story stays non-erotic unless an adult Situation is "
        "selected.")
    _NUDITY_REQUEST = re.compile(
        r"desnud|\bnude|naked|sin ropa|topless|undress|nackt|\bsex(?:o|ual|uales)?\b|porn|\btetas|\bpechos|\bsenos|\bpene\b|vagina|"
        r"genital|lingerie|lencer[ií]a|ropa interior|underwear|\bbra\b|sost[eé]n|brasier|panties|bragas|calzon|tanga|"
        r"thong|b[oó]xer|bikini|shirtless|sin (?:camisa|camiseta|blusa|brasier|sost[eé]n|pantal[oó]n|falda)|"
        r"semi ?desnud|half[- ]naked|se quita|quit[aá]ndose|takes? off|strips?\b|nalgas|butt\b|buttocks|escote|"
        r"cleavage|transparen|see[- ]through|sheer", re.IGNORECASE)
    _SING_REQUEST = re.compile(
        r"\bcant(?:a|an|ar|ando|e|o|ó|aba)\b|canci[oó]n|\bcoro\b|estribillo|\bletra\b|karaoke|\bsing(?:s|ing)?\b|"
        r"\bsang\b|\bsong\b|\bchorus\b|\blyrics?\b|\brap(?:s|ping|ea|ear|eando)?\b|serenat|\bchant(?:e|er|ez)\b|"
        r"\bsingt\b|\bcanta(?:re)?\b", re.IGNORECASE)
    @classmethod
    def _sings(cls, request):
        """A song is requested, not negated ('no canta', 'without singing')."""
        for m in cls._SING_REQUEST.finditer(request):
            if not re.search(r"(?:\bno|\bsin|\bwithout|\bnot|n't|\bnunca|\bni)\s+(?:\w+\s+)?$", request[:m.start()],
                             re.IGNORECASE):
                return True
        return False

    SINGING_NOTE = (
        "SINGING (the request asks for a song): the character really sings, not talks, shouts or hypes the crowd. "
        "Write original lyrics in the dialogue language (the selected one, otherwise the request's), in the style the "
        "request names: short rhythmic lines with internal rhyme and a catchy hook that repeats (verse into chorus "
        "when the clip allows), about 2 sung words per second with room for held notes and instrumental bars; never "
        "real songs or real artists. Each sung line is written as `<Subject N> (SN) sings: <d>[Language] lyric</d>`, "
        "preceded by how it is sung (vocal timbre and register, melody rising or falling, held or clipped notes, "
        "ad-libs, breath) and placed on the beat of the moment it happens; the mouth, breath and body move with the "
        "melody. The music the singer performs to is diegetic: write its source and arrangement inside the shots (a live "
        "band, a DJ, a speaker or backing track: style, tempo in BPM, instruments, groove, how it builds, visible "
        "players when they are in frame); overall_soundscape keeps the room, the crowd and Foley; "
        "non_diegetic_music is N/A unless the request also asks for a score. Spoken interjections, if any, use says and stay few.")
    PARTIAL_NUDITY_NOTE = (
        "NUDITY FROM THE USER'S WORDS: uncover exactly what the words ask, nothing more. A "
        "partial request (a top or shirt off, underwear, a bare back, chest or buttocks, see-through or wet clothes, a "
        "garment taken off) stays partial in every shot: the named area is shown and described exactly, everything "
        "else stays covered as COVERED ANATOMY says, and it never grows into full nudity or sex the words did not ask "
        "for; full nudity only when the words ask for it.")
    WARDROBE_FOLLOW_NOTE = (
        "Wardrobe Guide Garments + Reference Style (binding): each character wears the same garments as the guide "
        "person they replace, the same pieces and layers with the same cut, length, fit and way of moving, opening and "
        "coming off, restyled in the style of that character's own reference picture (in Video task Edit, <Picture 2> "
        "onward): its colours, fabrics and textures, patterns, finish, accessories and overall look. Garment by "
        "garment: the guide decides what each piece is, the reference decides how it looks. Every clothing change of "
        "the guide (opening, taking off, putting on) happens with those restyled garments at the same moments, ending "
        "just as covered or uncovered as the guide; nudity that comes from the guide is allowed and the user's own "
        "words still win. In transfer_cast, wardrobe_source is 'direction_controls' and requested_wardrobe lists each "
        "guide garment in the reference style.")
    KEEP_REFERENCE_NOTES = {
        "Wardrobe": "Wardrobe Keep Reference (binding): every character keeps exactly the clothing state of their own "
                    "reference picture (its garments, colours and accessories, or partial or no clothing when the picture "
                    "shows that), whatever the video or guide performer wears and even in a person swap; transfer_cast "
                    "wardrobe_source is 'reference'. A selected Outfit style restyles that same clothing (cut and fit "
                    "only, same pieces, colours and fabrics) as its own rule says.",
        "Environment": "Environment Keep Reference (binding): the setting is the place visible in the primary reference "
                       "picture, not the video or Driven location; adapt the observed action feasibly to that place and "
                       "its own furniture and surfaces.",
        "Hair & Makeup": "Hair & Makeup Keep Reference (binding): each character keeps the hairstyle, hair colour and "
                         "makeup of their own reference picture, whatever the clothes, era or guide do.",
        "Body proportions": "Body Proportions Keep Reference (binding, exact): each character has exactly the body of "
                            "their reference picture, which already shows it to the video model: in the definition write "
                            "it once and neutrally as 'the same figure and proportions as in <Picture N>' with at most one "
                            "plain word for the overall build that matches the picture (slim, average, athletic, petite, "
                            "heavyset...). Never list or stress body parts by size (no thick thighs, full bust, wide "
                            "hips, big buttocks, curvy) and never repeat body adjectives in the footage: actions describe "
                            "movement, not body size. The same body in every shot and scene: never slimmed, enlarged, "
                            "idealized or averaged toward a model body, and the performer's body shape is not "
                            "transferred.",
    }
    HAIR_NEW_NOTE = (
        "Hair & Makeup Create New (binding): design a new hairstyle (cut, length, shape, texture, colour), makeup, "
        "nails, facial hair and grooming for each character from the request, Era, Genre, Situation, Visual look and the request's heat; "
        "the reference picture gives face and identity, never its hairstyle or makeup, and the identity text does not "
        "describe them. Face, age, tattoos and marks stay.")
    KEEP_GUIDE_NOTES = {
        "Hair & Makeup": "Hair & Makeup Keep Guide (binding when a guide image/video is connected): each character has "
                         "exactly the hairstyle, hair colour and makeup of the guide person they replace, with the "
                         "reference face, age and identity.",
        "Wardrobe": "Wardrobe Keep Guide (binding when a guide image/video is connected): each character wears exactly the "
                    "clothes of the guide person they replace (garments, colours, fit, accessories), even against an era, "
                    "genre or look; the reference picture gives identity, not clothing. Write those clothes out concretely; never name the guide.",
        "Environment": "Environment Keep Guide (binding when a guide image/video is connected): the setting is exactly the "
                       "guide's place (layout, furniture, surfaces, light), even against an era, genre or look.",
        "Body proportions": "Body Proportions Keep Guide (binding when a guide image/video is connected, exact): each "
                            "character has exactly the body of the guide person they replace, read precisely and "
                            "written concretely in their definition (height and frame, weight and body fat, shoulder "
                            "width, bust or chest size, waist, hip width, buttocks, thighs, arm and leg thickness, "
                            "muscle tone, posture), with the reference face, age and identity. The same body in every "
                            "shot: never slimmed, enlarged, idealized or averaged.",
    }
    CONTROL_PRIORITY_NOTE = ("Priority: the user's written request, then the selected Direction Controls, then the SOURCE "
                             "OF EACH ASPECT rule, then swap and transfer defaults. A control on Auto adds no rule of "
                             "its own.")
    WARDROBE_NEW_NOTE = (
        "Wardrobe Create New (binding): every visible garment and accessory is newly designed from the request, Era, "
        "Genre, Situation, Visual look and the request's heat; no "
        "item of the reference outfit (top, bottom, outerwear, footwear, headwear, jewellery, bag) survives unless "
        "the user's request names it. The identity text lists face, and hair and body as their own controls allow, never the reference clothes; "
        "lead colours are new choices, never 'as reference'. The reference image gives identity, not clothing.")
    ENVIRONMENT_NEW_NOTE = (
        "Environment Create New (binding): design a new setting from the request, Era, Genre and Visual look; the "
        "backgrounds, rooms, furniture "
        "and props visible in the reference images are not used, recoloured or rearranged. A place named in the "
        "user's request is the only boundary.")
    BODY_NEW_NOTE = (
        "Body Proportions Creative Adaptation (binding): write each adapted character's target build and "
        "proportions explicitly in their definition; the body shape in the reference image is not kept, only the "
        "face, age and identity. Adult, coherent anatomy; no age change.")
    TRANSFER_DIALOGUE_NOTE = (
        "DIALOGUE IN TRANSFER (active Dialogue control; this is a request for speech and overrides any rule that a "
        "language selection is not one): the characters speak, in the selected language, lines written for this "
        "footage. Each line fits what that person is visibly doing at that moment (gaze, contact, effort, "
        "pace), the relationship the source shows (if any), and the transformed world: the era's way of speaking, the genre's "
        "tone, the mood and the NSFW heat. Place speech where the face and timing allow (pauses, breaths, moments "
        "facing another person who is present, or the camera), short enough for the clip; it gives voice to the existing action and never "
        "adds events, people or a plot outside the footage. The conversation stays continuous across scenes. "
        "Without Dialogue density, a few natural lines per scene; a person alone speaks to the camera or to "
        "themselves, never to an invented listener. EMOTION FIRST: read the emotional arc of the "
        "footage (what each moment feels like from the body, face, energy and rhythm, plus the selected genre, mood "
        "and the heat of the request) and let every line ride it: the words and their delivery rise, fall, break and change with "
        "the action, with interjections, laughter, breaths, unfinished or overlapping lines where they fit; "
        "people who are present react audibly or visibly. Each character has their own voice and a want in the moment; lines "
        "react to what just happened and carry subtext, never exposition, narration of the visible action, polite "
        "small talk or generic filler; no line repeats across scenes.")
    TRANSFER_NSFW_NOTE = (
        "NSFW IN TRANSFER (the user chose to charge this footage): the level re-reads the source action at that heat "
        "and replaces the rule against turning a non-intimate story intimate. Same choreography, timing, positions and "
        "camera, performed with that intent: gaze, expression, breath, touch on one's own body or between the people "
        "already present, clothing loosened or removed during the existing movements when the level allows (also over "
        "Wardrobe Keep Reference: the reference outfit is what comes off), more skin and anatomy in frame, camera and "
        "light lingering on it, sound and dialogue at the same heat. Contact between the people present may become "
        "its intimate counterpart in the same positions and rhythm; acts needing a partner or position the source "
        "lacks, new people and new places are not invented. How much skin is shown follows the request. Violence "
        "follows only the request. The selected genres shape how the same action is performed and "
        "filmed, in their adult version. If the request asks for the heat to rise, it rises scene by scene.")
    TRANSFER_SOUND = (
        "SOUND DESIGN (Transfer, binding; replaces every rule asking N/A for unrequested music or unevidenced sound): "
        "the generated audio is the final audio, so fill it completely and coherently for this footage, drawing on "
        "what the video shows, the references, the request and the controls (era: period instruments and sound "
        "world; genre and mood: tone; an adult request: heat; motion style: tempo). overall_soundscape: the place's "
        "ambience as it sounds in the transformed world, layered near and far, plus precise Foley tied to visible "
        "actions at their moment (steps, fabric, contact, breath, objects, impacts), named by concrete source. "
        "non_diegetic_music: an original score written as instrumentation, tempo that locks to the movement's "
        "rhythm, character and how its dynamics follow the action; or diegetic music when the scene has a source "
        "(club, radio, stage, a dancer's track), described where it plays. Use N/A only when raw ambience clearly "
        "serves the footage better (e.g. a security-camera look). Vocal sounds (breath, sighs, laughter, effort, "
        "moans in an adult scene) belong to the soundscape and need no Dialogue control; spoken words still do. "
        "SOUND CONTINUITY: the first scene fixes the sound identity (instruments, tempo, motif, room tone) and "
        "later scenes continue and develop it, never resetting or changing style at a cut unless the request or "
        "the action changes. Mix: music sits under dialogue and key Foley. Never name real songs or artists; no "
        "lyrics unless the user wrote them or asks someone to sing (then the SINGING rule applies). FORMAT: every scene prompt ends with its own two sections, exactly\n"
        "overall_soundscape:\n<ambience and Foley>\n\nnon_diegetic_music:\n<score or diegetic music>\n"
        "Sound and music never go inside integrated_multimodal_description, and are never left N/A by default. "
        "SPOKEN LINES (whenever there is speech): they live in integrated_multimodal_description at the moment they "
        "are said, never in the sound sections. The first time a character speaks, describe the voice outside the tag "
        "(pitch, timbre, age, accent); before every line state its delivery: the emotion and how it sounds (for "
        "example whispered, breathless, laughing, teasing, trembling, shouted, cold), its pace and volume, and the "
        "expression or gesture carrying it, then `<Subject N> (SN) says: <d>[Language] exact words</d>`. A line "
        "without a delivery is a flat line.")
    VIDEO_EDIT_NOTE = (
        "VIDEO EDIT (binding, this overrides any other format or length rule): <Video 1> is the source video and the "
        "target video is an edited version of it; its movement, poses, timing, cuts, camera and soundtrack stay as "
        "filmed. Change only what the user and the active controls ask (a person or character, clothes, hair, an "
        "object added, removed or replaced, the background, weather, time of day, light or style); everything else "
        "is fully_preserved. Each <Picture N> is a frame of the video repainted with the target: it gives the "
        "identity and look of what it shows. Write the six sections subject_definitions, summary, "
        "retention_analysis, detailed_description, overall_soundscape, non_diegetic_music. Describe the target video, "
        "never instructions to the model. subject_definitions: '<Video 1> is the source video for the target video "
        "edit.', then each changed or kept <Subject N> once, head to toe, with the same words every time it is named, "
        "tied to the source person or object it replaces by position and a distinct trait (colour, garment, place in "
        "frame); name the original traits that must disappear (for example 'no longer blonde'); a costume is written "
        "as a costume ('full-body mascot suit, wearer hidden'). summary: one short paragraph. retention_analysis: one "
        "short line per label, fully_preserved, partially_preserved, attribute_transfer or weak_reference, saying "
        "exactly what changes and what stays, never contradicting itself (a changed background is never also "
        "preserved). detailed_description: the style in one or two sentences, then the source's shots in order "
        "([Shot 1], later shots 'At MM:SS.mmm' inside the clip length), each saying who is in frame, where, in which "
        "look, and what changed; the movement itself is the source's, so name it in a few words only, no "
        "choreography. A removed thing says what fills its place (sky, sea, an empty floor); an added object is "
        "anchored to a stable feature ('on the table, left of the lamp'). No dialogue lines and no new sound: the "
        "source audio is reused 1:1.")

    VIDEO_EDIT_TASKS = {
        "swap": ("VIDEO TASK SWAP (binding): the reference pictures recast people of <Video 1>. Each <Picture N> "
                 "replaces one named source person (by position and a visible trait, as the request says, or in the "
                 "order the people appear); that <Subject N> keeps the source person's movement, place and timing "
                 "and takes the identity, hair, body and clothes the pictures and controls give. Everyone not "
                 "replaced, the place, the objects and the light stay fully_preserved unless asked."),
        "edit": ("VIDEO TASK EDIT (binding): the people of <Video 1> stay themselves (identity fully_preserved). "
                 "Change only what the request and the controls ask: clothes, hair, colours, an object swapped, the "
                 "background, weather, time of day, light or style; a <Picture N> shows that changed look or "
                 "object, never a new person."),
        "add": ("VIDEO TASK ADD (binding): everything in <Video 1> stays (people, place, light fully_preserved) and "
                "what the <Picture N> show is added: each new person or object gets its own <Subject N>, a place "
                "anchored to a stable feature of the frame, its size and how it moves with the scene."),
        "remove": ("VIDEO TASK REMOVE (binding): the person or thing the request names disappears from every shot "
                   "of <Video 1> and the prompt says what fills its place (the background behind it, sky, sea, an "
                   "empty floor); everything else stays fully_preserved."),
    }
    @classmethod
    def video_edit_contract(cls):
        """Edit mode, last words of the system prompt: format, task and the exact Picture list."""
        pics = _CLIP.get("edit_pictures") or []
        names = ", ".join(f"<Picture {i}>" for i in pics)
        picture_rule = (f"PICTURES (binding): the only pictures are {names}, one per attached reference image, exactly "
                        "as labelled. The '<Video 1> SAMPLE' frames are frames of <Video 1>, never Pictures; never "
                        "write any other <Picture N>." if pics else
                        "PICTURES (binding): no reference image is attached, so no <Picture N> exists; the "
                        "'<Video 1> SAMPLE' frames are frames of <Video 1>.")
        layout = ("FORMAT (binding): every scene_prompts entry is the full six-section text, each label alone on its "
                  "own line in this order: subject_definitions:, summary:, retention_analysis:, "
                  "detailed_description:, overall_soundscape:, non_diegetic_music:. Never one free paragraph.")
        return "\n\n".join([cls.VIDEO_EDIT_NOTE, cls.VIDEO_EDIT_TASKS.get(_CLIP.get("edit_task") or "swap",
                                                                         cls.VIDEO_EDIT_TASKS["swap"]),
                             picture_rule, layout])

    _PIC_LIST = re.compile(r"<Picture \d+>(?:(?:,\s*|\s+and\s+|\s+or\s+|,\s*and\s+)<Picture \d+>)*")

    @classmethod
    def _valid_pictures(cls, text):
        """Only the connected reference slots are Pictures; anything else was a video sample misnamed."""
        valid = set(_CLIP.get("edit_pictures") or [])

        def fix_list(m):
            nums = []
            for n in re.findall(r"<Picture (\d+)>", m.group(0)):
                if int(n) in valid and int(n) not in nums:
                    nums.append(int(n))
            if not nums:
                return "<Video 1>"
            tags = [f"<Picture {n}>" for n in nums]
            return tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]
        return cls._PIC_LIST.sub(fix_list, text)

    @staticmethod
    def _edit_sections_from_prose(text):
        """A free paragraph (no section labels) becomes the six-section layout."""
        text = re.sub(r"^\s*\[[^\]\n]*\]\s*\n+", "", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+(?=<|[A-Z\[])", text)
        subjects = []
        while sentences and re.match(r"<Subject \d+> is ", sentences[0]):
            subjects.append(sentences.pop(0).strip())
        body = " ".join(sentences).strip()
        if body and "[Shot 1]" not in body:
            body = "[Shot 1] " + body
        retention = ["<Video 1> (movement, camera, cuts, timing and everything not named below): fully_preserved - "
                     "unchanged from the source."]
        for line in subjects:
            who = re.match(r"(<Subject \d+>)", line).group(1)
            pics = list(dict.fromkeys(re.findall(r"<Picture \d+>", line)))
            if pics:
                retention.append(f"{who}: attribute_transfer - identity and look from {' and '.join(pics)}; movement "
                                 "and place from <Video 1>.")
        return ("subject_definitions:\n" + "\n".join(subjects) + "\n\nsummary:\n\n\nretention_analysis:\n"
                + "\n".join(retention) + "\n\ndetailed_description:\n" + body
                + "\n\noverall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A")

    _EDIT_LORA = {"swap": "swap", "edit": "edit", "add": "edit", "remove": "edit"}

    _EDIT_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
                      "overall_soundscape", "non_diegetic_music")

    @classmethod
    def _video_edit_format(cls, text):
        """Edit mode: the official H3 video-editing layout, with the source audio reused."""
        if not isinstance(text, str) or not text.strip():
            return text
        text = cls._valid_pictures(text)
        text = re.sub(r"(?:(?<=[.!?])|^)[ \t]*(?:At \d\d:\d\d\.\d{3},?\s*)?<Subject \d+>\s*[.!?]", "", text,
                      flags=re.MULTILINE)  # what a removed spoken line leaves behind
        if not re.search(r"(?m)^subject_definitions:", text):
            text = cls._edit_sections_from_prose(text)
        text = re.sub(r"^\s*\[[^\]\n]*\]\s*\n+(?=\s*subject_definitions:)", "", text)  # the tag lives in summary
        pictures = bool(re.search(r"<Picture \d+>", text))
        tag = "[video editing + reference generation + audio reuse]" if pictures else "[video editing + audio reuse]"

        def section(name, body_fn):
            nonlocal text
            m = re.search(rf"(?ms)^{name}:[ \t]*\n?(.*?)(?=^(?:{'|'.join(cls._EDIT_SECTIONS)}):|\Z)", text)
            if m:
                body = body_fn(m.group(1).strip())
                text = text[:m.start()] + f"{name}:\n{body}\n\n" + text[m.end():]
            else:
                text = text.rstrip() + f"\n\n{name}:\n{body_fn('')}"

        def subjects(body):
            lines = [x for x in body.splitlines() if x.strip()
                     and not re.match(r"\s*<(?:Video|Audio) 1> is ", x)]
            return "\n".join(["<Video 1> is the source video for the target video edit.",
                              "<Audio 1> is the synchronized audio track of <Video 1> and is reused in the target video."]
                             + lines)

        def summary(body):
            body = re.sub(r"^\s*\[[^\]]*\]\s*", "", body)
            body = re.sub(r"(?i)^the target video is an edited version of <Video 1>\.?\s*", "", body)
            return f"{tag} The target video is an edited version of <Video 1>. {body}".strip()

        def retention(body):
            lines = [x for x in body.splitlines() if x.strip() and not x.strip().startswith("<Audio 1>")
                     and x.strip().upper() != "N/A"]
            return "\n".join(lines + ["<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's "
                                      "complete final audio track."])

        section("subject_definitions", subjects)
        section("detailed_description", lambda body: re.sub(  # a tag left behind by a removed line
            r"[ \t]*<(?:Subject|Picture) \d+>[ \t]*(?=\n|$)", "", body))
        section("summary", summary)
        section("retention_analysis", retention)
        section("overall_soundscape", lambda body: "<Audio 1> is reused 1:1: the original ambience and sound "
                                                   "effects of <Video 1>, unchanged.")
        section("non_diegetic_music", lambda body: "<Audio 1> is reused 1:1: any original music of <Video 1> "
                                                   "continues unchanged.")
        order = []
        for name in cls._EDIT_SECTIONS:  # official order
            m = re.search(rf"(?ms)^{name}:[ \t]*\n?(.*?)(?=^(?:{'|'.join(cls._EDIT_SECTIONS)}):|\Z)", text)
            if m:
                order.append(f"{name}:\n{m.group(1).strip()}")
        return "\n\n".join(order)

    TRANSFER_BASIC_NOTE = (
        "BASIC TRANSFER OUTPUT (binding, this overrides every length, choreography, pose, timing and detail rule "
        "above): the video already carries the movement, poses, actions, expressions, timing and camera, so none "
        "of that is described. The prompt only says WHO (each reference person or object and whom/what it "
        "replaces), WEARING WHAT, WHERE (place and its objects) and the light, as the request and the controls "
        "say (kept or changed). subject_definitions: one short line per subject (Picture tag, identity, outfit in "
        "a few words; body only when a control asks). integrated_multimodal_description: one or two short "
        "sentences for the place, its objects and the light, then one short line per shot that only names who is "
        "in frame, in which position (left, right, center) and in which outfit; at most three words for what they "
        "do. No poses, gestures, choreography, facial expressions, camera moves, lens or timing detail, no "
        "adjective stacks. About 60 to 120 words per scene in all.")
    TRANSFER_KEEP_AUDIO = (
        "KEEP SOURCE AUDIO (binding): the source video's original audio is the final audio. Write N/A in "
        "overall_soundscape and in non_diegetic_music and write no <d> dialogue lines; mouths, singing and "
        "reactions simply follow the source performance. All visual rules apply unchanged.")
    _TRANSFER_AUDIO_SWAPS = (
        ("Speech needs the request or an active Dialogue control (a language); otherwise no dialogue, lyrics,\n"
         "narration or score unless provided. Sparse video frames reveal no audio; never fabricate a soundtrack.\n"
         "Use non_diegetic_music: N/A when no score is requested. Describe only source-supported ambience\n"
         "and physical Foley in overall_soundscape, or N/A when no sound is supported/requested.",
         "Speech needs the request or an active Dialogue control (a language); no narration or lyrics unless\n"
         "provided. Sound and music follow the SOUND DESIGN rule."),
        ("Do not add dialogue merely for a language setting.\nOnly requested/provided dialogue, narration or music is "
         "permitted. Use N/A for unsupported sound\nand unrequested music.",
         "An active Dialogue language asks for dialogue; sound and music follow the SOUND DESIGN rule."),
        ("No dialogue requests silence; a language choice governs requested speech only, not permission to invent it. ",
         "No dialogue requests silence; an active language choice asks for dialogue written for this footage. "),
        ("Style any requested music consistently but do not add a score merely for a genre/look. ",
         "Music and sound follow the SOUND DESIGN rule, consistent across blocks. "),
        ("do not restart the requested musical track or invent new dialogue to fill a new block.",
         "do not restart the musical track; dialogue continues the conversation instead of filling time."),
        ("A selected language does not permit a greeting, speech or vocal performance; do not add any unless "
         "explicitly requested. Use N/A where sound/music is not evidenced or requested.",
         "An active Dialogue language asks for speech written for this footage; sound and music follow the SOUND "
         "DESIGN rule."),
    )

    @staticmethod
    def _clean_engine_text(text):
        """The original engine still speaks of a 'Driven Strength' (1/2/3) with its own appearance policies; that
        control no longer exists (Wardrobe / Hair & Makeup / Body / Environment decide), so its wording would
        contradict the selected controls. Rewritten for every director call; the guide analysis is untouched."""
        target = "SOURCE OF EACH ASPECT" if "SOURCE OF EACH ASPECT" in text else "the Direction Controls"
        text = re.sub(r"Apply Driven Strength consistently:\n1 [^\n]*\n2 [^\n]*\n3 [^\n]*\n",
                      f"Each aspect (outfit, hair and makeup, body, setting) follows {target}.\n", text)
        lines, kept, skipping = text.split("\n"), [], False
        for line in lines:  # drop a "DRIVEN STRENGTH n — ..." heading and its own paragraph, never the next section
            if re.match(r"DRIVEN STRENGTH [123] — ", line):
                skipping = True
                continue
            if skipping and (not line.strip() or re.match(r"[A-Z][A-Z0-9 /&'-]{5,}(?: — |:|$)", line)):
                skipping = False
            if not skipping:
                kept.append(line)
        text = "\n".join(kept)
        text = re.sub(r"DRIVEN STRENGTH [123]:[^\n]*", "", text)
        text = re.sub(r"(?:the (?:selected |existing )?)?Driven Strength(?: policy| defaults)?", target, text)
        text = text.replace("at EVERY strength", "always").replace("The selected strength never cancels", "Nothing cancels")
        text = text.replace("Transfer the exact visible garment design, hair color/length/style, body proportions, pose, "
                            "activity, lighting and setting while retaining reference facial identity. ",
                            f"Transfer the pose, activity, lighting and setting, and garments, hair and body only as {target} "
                            "allows, while retaining reference facial identity. ")
        return text

    @classmethod
    def _transfer_audio_system(cls, text):
        """Final Transfer system prompt: creative sound by default, or the source audio kept untouched."""
        if _CLIP.get("keep_audio"):
            return text + "\n\n" + cls.TRANSFER_KEEP_AUDIO + (("\n\n" + cls.TRANSFER_BASIC_NOTE)
                                                                if _CLIP.get("basic") else "")
        for old, new in cls._TRANSFER_AUDIO_SWAPS:
            text = text.replace(old, new)
        return text + "\n\n" + cls.TRANSFER_SOUND

    TRANSFER_PACE_NOTE = (
        "OTHER CONTROLS IN TRANSFER: Motion style shapes how the camera behaves and how the observed movements are "
        "performed (style, energy), not new choreography or more events.")
    NUDITY_NOTES = {
        "clothed": "Nudity Clothed (binding, above the NSFW level): everyone stays dressed in every shot; breasts, "
                   "nipples, bare buttocks and genitals are never exposed. Every action, however intense, happens through "
                   "or around clothing that stays on and covers (bodies pressed together, grinding and thrusting through "
                   "fabric, hands over or under clothing without exposing anything, clothing shifted only out of frame); "
                   "say in each shot what covers each body.",
        "underwear": "Nudity Underwear (binding): at most lingerie or underwear, which may be revealed by undressing on "
                     "screen; bras, panties and briefs stay on, so no nipples, fully bare buttocks or genitals; nothing beyond "
                     "underwear is ever shown.",
        "topless": "Nudity Topless / Partial (binding): breasts or one other area may be bare (a top off, a garment "
                   "pulled aside), the rest stays covered; genitals stay covered or out of frame.",
        "nude": "Nudity Nude: full nudity on screen (breasts, buttocks, full bodies); genitals may be seen but are not "
                "the subject of close framing.",
        "explicit": "Nudity Explicit: full nudity with genitals visible and shown in close anatomical detail, held in "
                    "frame.",
    }
    NUDITY_DEFAULT = {"flirt": "clothed", "sensual": "underwear", "heated": "topless", "sex": "nude",
                      "hardcore": "explicit", "extreme": "explicit"}
    NUDITY_SHORT = {"clothed": "everyone dressed, nothing exposed", "underwear": "at most underwear",
                    "topless": "breasts or one area bare, genitals covered", "nude": "full nudity",
                    "explicit": "nudity with genitals in close detail"}

    @classmethod
    def nudity_key(cls, direction_text):
        """The Nudity rule in force: the Nudity control, otherwise the NSFW level's default; '' when neither is set."""
        text = str(direction_text or "")
        value = (re.search(r"(?im)^Nudity:\s*(.+)$", text) or [None, ""])[1].strip().lower()
        if value:
            return next((key for key in cls.NUDITY_NOTES if value.startswith(key)), "")
        level = (re.search(r"(?im)^Intensity:\s*(\w+)", text) or [None, ""])[1].strip().lower()
        return cls.NUDITY_DEFAULT.get(level, "")

    INTENSITY_SCOPE = ("The NSFW level is the user's own request for that much sexual action: it overrides every "
                       "default rule against sensual framing, escalation or explicit detail (and against nudity and "
                       "undressing up to what the Nudity rule allows), and the footage must reach that level instead "
                       "of softening it. The level never "
                       "evaporates. When the request leaves the kind of story open (it only names the characters, their "
                       "relationship, a place or a mood), the level decides it: the premise is an intimate, erotic story "
                       "with the characters that reaches the level (along the Progression when one is set), with "
                       "era, genre, look and motion shaping how it plays. A story the user explicitly "
                       "framed as non-sexual keeps its kind and takes only the sensuality that fits it. Violence and gore "
                       "are set only by the Violence control, never by this level. The level never adds people: the "
                       "cast is the people in the references and guide plus whoever the request names or clearly asks "
                       "for. With one person, the level is reached solo (undressing, self-touch and solo pleasure at the "
                       "level's explicitness, toward the camera or the viewer); a partner appears only when the request "
                       "brings one.")
    ADULT_GENRE_NOTE = ("ADULT VERSION OF EVERY GENRE (an NSFW level is active): each selected genre keeps its own "
                        "mechanism, world and style but plays as its adult version at that level, together with the "
                        "selected look, motion, era and other controls: romance becomes erotic romance, comedy erotic "
                        "comedy, horror erotic horror, a creator, vlog or ASMR format becomes adult creator content, a "
                        "music video a sensual performance, and so on for any genre. The intimacy grows from the genre's "
                        "own situation instead of replacing it.")
    NSFW_SHORT = {"flirt": "innocent, playful flirting",
                  "sensual": "slow sensuality, caresses, soft suggestive words",
                  "heated": "bold, aggressive desire: grinding, groping, dirty talk",
                  "sex": "openly sexual movement, poses, touch and talk",
                  "hardcore": "raw, explicit and vulgar, sustained",
                  "extreme": "total release: filthy, dominant, dark edge, maximum intensity"}
    NSFW_CAST_NOTE = ("The NSFW level measures how erotic the actions, poses, movements, gestures, attitude and dialogue are, "
                      "from innocent to total release; it never decides who is in the scene. It is played with the cast "
                      "present: one person reaches it alone (poses, self-touch, self-pleasure from Sex up, dirty talk to "
                      "the camera or the viewer), two people with each other; a partner never appears for it.")
    VIOLENCE_SHORT = {"mild": "real danger, violence mostly off-screen",
                      "moderate": "a visible fight or injury on screen",
                      "strong": "wounds and blood shown plainly",
                      "graphic": "gore and impact in close detail",
                      "extreme": "extreme gore, sustained"}

    @classmethod
    def control_items(cls, direction_text):
        """Every active Direction Control as one concrete, visible requirement (any combination, NSFW or not)."""
        text = str(direction_text or "")
        get = lambda label: (re.search(rf"(?im)^{re.escape(label)}:\s*(.+)$", text) or [None, ""])[1].strip()
        items = []
        for genre in [g.strip() for g in get("Genre").split(" + ") if g.strip()]:
            items.append(f"Genre {genre}: its own situation and mechanism, recognizable without the label")
        pace = get("Story pace").lower()
        if pace in cls.STORY_PACE_NOTES and _CLIP.get("mode") != "transfer":
            items.append(f"Story pace {get('Story pace')}: "
                         + {"slow": "one savoured moment, no jumps", "fast": "several events or places, compressed by cuts",
                            "very fast": "a quick montage of short events and places"}[pace])
        situation = get("Situation")
        if situation.lower() in cls.SITUATION_CUES:
            items.append(f"Situation {situation}: its place, roles, props and turn, recognizable without the label")
        for style in [m.strip() for m in get("Motion style").split(" + ") if m.strip()]:
            items.append(f"Motion style {style}: visible in how the bodies and the camera move")
        for look in [v.strip() for v in get("Visual look").split(" + ") if v.strip()]:
            items.append(f"Visual look {look}: its capture, grain, colour and light")
        level = get("Intensity").capitalize()
        if level.lower() in cls.NSFW_SHORT:
            items.append(f"NSFW {level}: {cls.NSFW_SHORT[level.lower()]}"
                         + (" (along the Progression plan)" if get("Progression") else "")
                         + (", in the adult version of the selected genre" if get("Genre") else ""))
        bare = cls.nudity_key(text)
        if bare and re.search(r"(?im)^Nudity:\s*\S", text):
            items.append(f"Nudity {get('Nudity')}: {cls.NUDITY_SHORT[bare]}")
        violence = get("Violence")
        if violence.lower() in cls.VIOLENCE_SHORT:
            items.append(f"Violence {violence}: {cls.VIOLENCE_SHORT[violence.lower()]}")
        curve = get("Progression").lower()
        if curve in cls.ARC_SHAPES:
            items.append(f"Progression {curve.capitalize()}: {cls.ARC_SHAPES[curve]}")
        if _CLIP.get("mode") == "transfer" and _CLIP.get("task"):
            items.insert(0, {"edit": "Video task Edit: the video's own people and footage, changed only where asked",
                             "add": "Video task Add: the video's people unchanged, plus the people of Picture 2 onward",
                             "remove": "Video task Remove: the named person or thing gone cleanly, nothing else changed",
                             }[_CLIP["task"]])
        aspects = {"Wardrobe": "outfit", "Environment": "place", "Hair & Makeup": "hairstyle and makeup",
                   "Body proportions": "build"}
        for label, what in aspects.items():
            value = get(label).lower()
            if value.startswith("create new") or value.startswith("creative"):
                items.append(f"{label} {get(label)}: a new {what}, nothing taken from the reference pictures")
            elif value.startswith("keep reference") and _CLIP.get("task") == "edit" and _CLIP.get("mode") == "transfer":
                items.append(f"{label} Keep Reference (Edit): the {what} of Picture 2 onward on the video's people"
                             + (", opened or taken off exactly when and as far as the video's clothes are"
                                if label == "Wardrobe" else ""))
            elif value.startswith("guide garments") and _CLIP.get("mode") == "transfer":
                items.append(f"{label} Guide Garments + Reference Style: the guide's garments in the reference's "
                             "style, changing exactly when and as far as the guide's clothes do")
            elif value.startswith("keep reference"):
                items.append(f"{label} Keep Reference: the {what} of the reference picture")
            elif value.startswith("keep guide"):
                items.append(f"{label} Keep Guide: exactly the {what} shown in the guide")
        if get("Era"):
            items.append(f"Era {get('Era')}: period fashion, objects and sets")
        if get("Outfit style") and re.search(r"(?im)^Wardrobe:\s*Keep Reference", text):
            items.append(f"Outfit style {get('Outfit style')}: the reference clothing restyled in that cut, same pieces "
                         "and colours")
        elif get("Outfit style") and not re.search(r"(?im)^Wardrobe:\s*Keep Guide", text):
            items.append(f"Outfit style {get('Outfit style')}: every designed outfit in that style")
        for name in cls.lora_names(text):
            items.append(f"LoRA {name}: " + ("that visual style in every scene" if cls.lora_is_style(name) else
                                             "the story reaches that action and shows it plainly for a real stretch"))
        if _CLIP.get("i2v") and cls._DESIGN_CONTROLS.search(text):
            items.insert(0, "I2V: <Picture 1> is the exact opening frame (its person, outfit, hair, place and light); "
                            "the design controls below reach that character and place only through visible change after it")
        dialogue = get("Dialogue")
        if dialogue.lower() == "no dialogue":
            items.append("No dialogue: nobody speaks")
        elif dialogue and dialogue.lower() not in ("none", "auto") and not (
                _CLIP.get("keep_audio") and (_CLIP.get("mode") == "transfer" or _CLIP.get("video_edit"))):
            items.append(f"Dialogue in {dialogue}" + (f", density {get('Dialogue density')}" if get("Dialogue density") else ""))
        chain = get("Action chain").lower()
        if chain in cls.ACTION_CHAIN_NOTES and _CLIP.get("mode") != "transfer":
            items.append(f"Action chain {get('Action chain')}: "
                         + {"short": "one action and its result", "long": "4-5 actions, each caused by the one before",
                            "very long": "a fast domino chain of actions"}[chain] + ", ending in a visible result")
        for label in ("Action density", "Coverage"):
            if get(label):
                items.append(f"{label} {get(label)}")
        return items

    I2V_CONTROLS_NOTE = (
        "I2V OPENING FRAME AND THE DESIGN CONTROLS (binding): frame zero is exactly <Picture 1>: its person, outfit, "
        "hair, body, place and light. Wardrobe, Hair & Makeup or Environment Create New, Body Creative Adaptation, "
        "Keep Guide and Era apply in full to everything not visible at frame zero (other characters, later places), "
        "and reach the <Picture 1> character and place only through visible on-screen change after the opening "
        "(changing or removing clothes, restyling, moving to a new place, the world around changing), never in the "
        "first shot: the opening holds <Picture 1> as it is and the change comes later, gradually; a change that does "
        "not fit the rest of the clip starts here and completes in the next scene instead of being rushed. In a sequence, "
        "scene 1 makes that change and later scenes keep the new design.")
    I2V_ALIGNMENT = ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
                     "referenced.")
    GUIDE_SILENT_NOTE = (
        "GUIDE = IMAGE/VIDEO TO PROMPT (binding): read the guide image/video as inspiration and write what it shows "
        "(action, pose, staging, camera, place, light, and clothes/hair/body where the controls take them from it) "
        "into the prompt as your own concrete description, always combined with the reference pictures (faces and "
        "identity, plus every aspect set to Keep Reference) and the user's request, which wins. The video model never "
        "sees the guide: the prompt never mentions a guide, guide image, guide video, driver, Driven, blueprint or "
        "source, and never says 'as shown/as seen' about it.")

    @classmethod
    def guide_look_note(cls):
        """Keep Guide in Normal/Compact: the guide's observed look as concrete text the director must write."""
        keep = (_CLIP.get("mask") or {}).get("keep_guide") or {}
        cached = _CLIP.get("blueprint_cache")
        blueprint = cached[0] if isinstance(cached, tuple) and cached and isinstance(cached[0], dict) else {}
        roles = blueprint.get("roles") if isinstance(blueprint.get("roles"), list) else []
        fields = [("Wardrobe", "wardrobe", "wears"), ("Hair & Makeup", "hair", "hair and makeup"),
                  ("Body proportions", "body_proportions", "body")]
        lines = []
        for role in roles:
            if not isinstance(role, dict):
                continue
            parts = [f"{word}: {str(role.get(key) or '').strip()}" for label, key, word in fields
                     if keep.get(label) and str(role.get(key) or "").strip()]
            if parts:
                lines.append(f"the person who plays the '{str(role.get('role') or 'person').strip()}' role — " + "; ".join(parts))
        place = "; ".join(v for v in (str(blueprint.get(k) or "").strip() for k in ("setting", "props", "lighting")) if v)
        if keep.get("Environment") and place:
            lines.append("the setting — " + place)
        if not lines:
            return ""
        return ("KEEP GUIDE LOOK (binding, these replace the reference pictures' clothes/hair/body/place on these "
                "aspects only; face and identity stay the reference's): write exactly this, concretely, in each "
                "character's Target appearance / Body proportions and in the setting paragraph, in every scene: "
                + " | ".join(lines) + ". Never describe the reference picture's outfit for an aspect listed here.")

    I2V_OPENING_NOTE = (
        "IMAGE-TO-VIDEO FIRST FRAME (binding): [Shot 1] opens exactly on <Picture 1> as it is (the same people, "
        "clothing, hair, colours, place, framing and light) and stays in that place and look for the first shot. "
        "Begin integrated_multimodal_description by stating what is visible in <Picture 1> at 0.00 seconds, then the "
        "motion that grows from it. No jump to another place, outfit or composition at the start; any change asked by "
        "the request, the controls or the story happens later in the clip, visibly and gradually. THE PLACE: unless the "
        "request or Environment Create New asks for another place, the setting of the whole clip is the place visible "
        "in <Picture 1>, described as it really is (a plain studio backdrop stays a plain backdrop, with its own light); "
        "never invent rooms, furniture, windows or light sources the picture does not show; the environment and "
        "visual_style fields describe that place. The code adds the alignment line: never write it in any field.")
    _DESIGN_CONTROLS = re.compile(r"(?im)^(?:(?:Wardrobe|Environment|Hair & Makeup):\s*(?:Create New|Keep Guide)|"
                                  r"Body proportions:\s*(?:Creative|Keep Guide)|Era:\s*\S)")

    @classmethod
    def control_checklist(cls, direction_text):
        """A short, last-read list of the active controls: what the footage must visibly show."""
        items = cls.control_items(direction_text)
        if _CLIP.get("mode") and _CLIP.get("mode") != "transfer" and not _CLIP.get("video_edit"):
            items.append("Ending: the last (or only) clip finishes the story and holds its outcome in the final "
                         "seconds, never cut mid-action or mid-line")
        return ("FINAL CHECK BEFORE ANSWERING (the selected Direction Controls; the footage must visibly show each): "
                + "; ".join(items) + ".") if items else ""

    _NUDITY_WORDS = re.compile(r"\b(?:nude|naked|topless|bare (?:breasts?|chest|body|skin|buttocks)|nipples?|areolae?|"
                               r"undress\w*|strips? (?:off|down|naked)|unclothed)\b", re.IGNORECASE)
    _SENSUAL_WORDS = re.compile(r"\b(?:kiss\w*|caress\w*|strok\w*|embrac\w*|nuzzl\w*|touch\w*|fondl\w*|lick\w*)\b",
                                re.IGNORECASE)
    _HEATED_WORDS = re.compile(r"\b(?:grind\w*|straddl\w*|grop\w*|making out|makes? out|dry[- ]hump\w*|undress\w*|"
                               r"hip rolls?|rolls? (?:her|his) hips|lip[- ]bit\w*|bites? (?:her|his) lip|dirty talk|"
                               r"unbutton\w*|unzip\w*|strips?\b|pins? (?:her|him)|writh\w*|fondl\w*)\b", re.IGNORECASE)
    _UNDERWEAR_WORDS = re.compile(r"\b(?:lingerie|underwear|bra|panties|thong|negligee|corset|garter)\b", re.IGNORECASE)
    _SEX_WORDS = re.compile(r"\b(?:sex|penetrat\w*|thrust\w*|straddl\w*|grind\w*|masturbat\w*|orgasm\w*|climax\w*|"
                            r"vulva|labia|penis|clitoris|oral|fingering|riding)\b", re.IGNORECASE)
    _VIOLENCE_WORDS = re.compile(r"\b(?:punch\w*|hits?|strikes?|slaps?|kicks?|fights?|fighting|wound\w*|blood\w*|"
                                 r"bleed\w*|stab\w*|shoot\w*|shot at|gun\w*|knife|blade|injur\w*|bruis\w*|attack\w*|"
                                 r"struggl\w*|grabs? (?:her|him|them) by|shov\w*|choke\w*|gore)\b", re.IGNORECASE)

    _EXPOSED = re.compile(
        r"(?:pants|trousers|jeans|shorts|boxers|briefs|underwear|sweatpants|slacks|panties|knickers|thong)\b[^.;\n]{0,30}?"
        r"\b(?:(?:pulled|pushed|tugged|yanked|slid|slipped) (?:down|off|aside|low)|dropped|lowered|"
        r"(?:down )?(?:to|around) (?:his |her |the )?(?:mid-?)?(?:thighs|knees|ankles|shins|calves))"
        r"|\bfly (?:is )?(?:open|unzipped)|naked from the waist down|\bbottomless\b|\bno underwear\b|without underwear",
        re.IGNORECASE)
    _MALE_WORD = re.compile(r"\b(?:he|his|him|man|men|guy|boyfriend|husband)\b", re.IGNORECASE)
    _FEMALE_WORD = re.compile(r"\b(?:she|her|hers|woman|women|girl|girlfriend|wife)\b", re.IGNORECASE)
    _MALE_PARTS = re.compile(r"\b(?:penis|cock|shaft|erection|erect|flaccid|scrotum|testicles|glans|foreskin|genitals?)\b",
                             re.IGNORECASE)
    _FEMALE_PARTS = re.compile(r"\b(?:vulva|labia|pussy|clitoris|vagina|pubic|genitals?)\b", re.IGNORECASE)

    @classmethod
    def undescribed_exposure(cls, prompt):
        """People whose lower body is exposed by their clothing while their anatomy is never named (video models then
        render a smooth lump). Returns 'man' and/or 'woman'."""
        text = re.sub(r"<d>.*?</d>", " ", str(prompt), flags=re.S)
        found = set()
        if not (cls._NUDITY_WORDS.search(text) or cls._SEX_WORDS.search(text) or cls._MALE_PARTS.search(text)
                or cls._FEMALE_PARTS.search(text)):
            return []  # no intimate scene: a lowered garment is not checked
        for match in cls._EXPOSED.finditer(text):
            before = text[max(0, match.start() - 250):match.start()]
            gendered = list(re.finditer(r"\b(?:he|his|him|man|guy|she|her|woman|girl)\b", before, re.IGNORECASE))
            who = gendered[-1].group(0).lower() if gendered else ""
            if who in ("he", "his", "him", "man", "guy") and not cls._MALE_PARTS.search(text):
                found.add("man")
            elif who in ("she", "her", "woman", "girl") and not cls._FEMALE_PARTS.search(text):
                found.add("woman")
        return sorted(found)

    _GENITALS = re.compile(r"\b(?:vulva|labia|clitoris|pussy|penis|cock|scrotum|testicles|genitals?|anus)\b", re.IGNORECASE)
    _BREASTS_BARE = re.compile(r"\b(?:topless|nipples?|areolae?|bare (?:breasts?|chest)|naked|nude|unclothed)\b",
                               re.IGNORECASE)

    @classmethod
    def nudity_excess(cls, direction_text, prompt):
        """More skin than a selected Nudity control allows (only when the control itself is set)."""
        text = str(direction_text or "")
        if not re.search(r"(?im)^Nudity:\s*\S", text):
            return ""
        out = cls._NEGATED.sub(" ", re.sub(r"<d>.*?</d>", " ", str(prompt), flags=re.S))
        bare = cls.nudity_key(text)
        if bare == "clothed" and (cls._BREASTS_BARE.search(out) or cls._GENITALS.search(out)
                                  or re.search(r"\bbare (?:buttocks|ass)\b", out, re.IGNORECASE)):
            return "Clothed"
        if bare == "underwear" and (cls._BREASTS_BARE.search(out) or cls._GENITALS.search(out)):
            return "Underwear"
        if bare == "topless" and cls._GENITALS.search(out):
            return "Topless / Partial"
        return ""

    @classmethod
    def missing_controls(cls, direction_text, prompt):
        """Selected NSFW / Violence levels that the delivered prompt does not reach at all (lexical check)."""
        text = str(direction_text or "")
        out = re.sub(r"<d>.*?</d>", " ", str(prompt), flags=re.S)
        out = cls._NEGATED.sub(" ", out)  # 'not naked', 'without blood': a negated word never counts as reached
        missing = []
        level = (re.search(r"(?im)^Intensity:\s*(\w+)", text) or [None, ""])[1].capitalize()
        if level == "Sensual" and not (cls._SENSUAL_WORDS.search(out) or cls._SEX_WORDS.search(out)):
            missing.append(f"NSFW {level} ({cls.NSFW_SHORT['sensual']})")
        elif level == "Heated" and not (cls._HEATED_WORDS.search(out) or cls._SEX_WORDS.search(out)):
            missing.append(f"NSFW {level} ({cls.NSFW_SHORT['heated']})")
        elif level in ("Sex", "Hardcore", "Extreme") and not cls._SEX_WORDS.search(out):
            missing.append(f"NSFW {level} ({cls.NSFW_SHORT[level.lower()]})")
        bare = cls.nudity_key(text)
        bare_name = (re.search(r"(?im)^Nudity:\s*(.+)$", text) or [None, bare.capitalize()])[1].strip()
        if bare == "underwear" and not (cls._UNDERWEAR_WORDS.search(out) or cls._NUDITY_WORDS.search(out)):
            missing.append(f"NSFW Nudity {bare_name} ({cls.NUDITY_SHORT[bare]})")
        elif bare in ("topless", "nude") and not cls._NUDITY_WORDS.search(out):
            missing.append(f"NSFW Nudity {bare_name} ({cls.NUDITY_SHORT[bare]})")
        elif bare == "explicit" and not (cls._MALE_PARTS.search(out) or cls._FEMALE_PARTS.search(out)):
            missing.append(f"NSFW Nudity {bare_name} ({cls.NUDITY_SHORT[bare]})")
        if level in ("", "Sex", "Hardcore", "Extreme"):  # an explicit act LoRA must reach the footage
            for name in cls.lora_names(text):
                if (cls._LORA_EXPLICIT.search(name) and not cls.lora_is_style(name)
                        and not (cls._NUDITY_WORDS.search(out) or cls._SEX_WORDS.search(out))):
                    missing.append(f"NSFW LoRA {name} (the act its name says, shown plainly on screen)")
        violence = (re.search(r"(?im)^Violence:\s*(\w+)", text) or [None, ""])[1].lower()
        if violence in ("moderate", "strong", "graphic", "extreme") and not cls._VIOLENCE_WORDS.search(out):
            missing.append(f"Violence {violence.capitalize()} ({cls.VIOLENCE_SHORT[violence]})")
        return missing

    _NEGATED = re.compile(r"\b(?:not(?! only)|no(?! longer| sooner)|never|without|nor|instead of|rather than)\b"
                          r"[^.,;:!?\n]{0,40}", re.IGNORECASE)
    _REQUEST_LIMIT = {
        "NSFW": re.compile(r"\b(?:sin|no|nada de|ni|without|not?|never|sem|ohne|senza|sans)\s+(?:\w+\s+){0,2}?"
                           r"(?:desnud\w*|nud\w*|naked|sexo|sex\w*|expl[ií]cit\w*|explicit\w*|porn\w*|genital\w*)",
                           re.IGNORECASE),
        "Violence": re.compile(r"\b(?:sin|no|nada de|ni|without|not?|never|sem|ohne|senza|sans)\s+(?:\w+\s+){0,2}?"
                               r"(?:sangre|blood\w*|violen\w*|gore|golpes|heridas|injur\w*|wounds?)", re.IGNORECASE),
    }

    @classmethod
    def request_limits(cls, request):
        """Controls whose level the user's own words turn down ('sin desnudez', 'no blood'): the words win."""
        return {name for name, pattern in cls._REQUEST_LIMIT.items() if pattern.search(str(request or ""))}

    VIOLENCE_SCOPE = ("The Violence level is the user's own request for that much violence: the footage must reach it "
                      "instead of softening it, and an open request makes conflict and violence part of the story. It "
                      "never turns into sexual violence: violence never happens during intimacy and nobody is hurt by sexual contact; "
                      "with an NSFW level the fight and the intimate moments are separate beats, and intimacy stays "
                      "visibly consensual: no struggle for control, wrestling, pinned wrists, marks or injuries between "
                      "intimate partners unless the request itself asks for consensual rough play; the violence goes to "
                      "rivals, threats or other moments of the story.")
    STORY_PACE_NOTES = {
        "slow": ("Story pace Slow (binding): one moment savoured, no jumps in place or time; the clip lives inside a "
                 "single small event, with room for looks, pauses, details and reactions."),
        "fast": ("Story pace Fast (binding): the story covers more ground than usual: about 2-3 distinct events, places "
                 "or moments in a 15 s clip (fewer in shorter clips, more in longer ones), each one moving the story on "
                 "(the bill is paid, cut, they already walk down the street). Jump through cuts that compress time and "
                 "space (ellipsis), never showing every trip; dialogue keeps only the lines that move the story, short "
                 "and answered fast. Each new place or moment is set in one glance (light, sound, a telling prop). The "
                 "central event the request, situation or controls promise lands early, before the middle of the clip, "
                 "and the rest develops it further; in a sequence it lands in the first scene and the later scenes go on "
                 "from there with new events and places at the intensity the controls set, never saving it for the end."),
        "very fast": ("Story pace Very Fast (binding): almost a montage: about 4-5 short events, places or time jumps in "
                      "a 15 s clip (scaled to the length), one clear action each, linked by cuts on movement or match "
                      "cuts; dialogue reduced to single lines or none, sound and music carrying the flow. It still reads "
                      "as one story with a clear order and a landing, never random fragments. The central event the request, "
                      "situation or controls promise arrives in the first seconds; in a sequence, in the first scene, "
                      "and every later scene moves past it to new events at the intensity the controls set."),
    }
    COMPACT_NOTE = (
        "COMPACT OUTPUT (binding, this overrides the word ranges above): every scene prompt is about half the usual "
        "length. Keep in full: the shot markers and times, who is in frame and where, each action with its start and "
        "result, the clothing and position state at each cut, the named anatomy and covering, and every dialogue line "
        "with its speaker and short voice note. Cut: adjective stacks, repeated description, long light, lens and "
        "atmosphere passages (one short sentence for look and place), and anything the reference pictures already "
        "show. overall_soundscape and non_diegetic_music: one short sentence each. Subject definitions: identity and "
        "outfit in a few words.")
    STORY_PACE_SINGLE = ("With Single take there are no cuts: the story moves on through continuous action (they walk out "
                         "and the camera follows them into the next place, light changing over a held frame), never a "
                         "jump.")
    ACTION_CHAIN_NOTES = {
        "short": ("Action chain Short (binding): inside each moment one action and its result, unhurried, with room "
                  "for looks, breath and small reactions."),
        "long": ("Action chain Long (binding): inside each moment about 4-5 linked actions (fewer in 5 s, more in long "
                 "clips), each one caused by the one before and causing the next, ending in a visible result; a chain "
                 "of cause and effect in the same place and time, not a list of gestures and not a jump to another "
                 "place (that is Story pace)."),
        "very long": ("Action chain Very Long (binding): the longest chain of linked actions the clip can hold, a fast "
                      "domino effect where every action triggers the next (about 6-8 links in 10 s), each one legible "
                      "and physically possible, ending in a clear result; never filler and never unrelated gestures."),
    }
    ACTION_DENSITY_NOTE = {
        "high": "Action density High (overrides any motion-style wording about not adding events): more distinct, "
                "legible physical beats than usual at a brisk pace, scaled to the clip (about 2-3 in 5 s, 4-6 in 10 s, "
                "more in longer clips), each one visibly different (a new movement, object, direction or contact) and "
                "physically possible in its time; never filler.",
        "low": "Action density Low: one or two unhurried beats per scene, long holds and small movements.",
    }

    _ENGLISH = frozenset("the you i and is are what it to my me don't not we this that your was do how why "
                         "where can will just know want here there".split())

    _ENGLISH_PROSE = frozenset("the a an and of to in on with her his their its is are at from as into by over "
                               "toward towards while for she he they them it this that".split())
    _FOREIGN_PROSE = frozenset(  # frequent function words of Spanish, Portuguese, French, Italian and German
        "el la los las del que y en con su sus una un por para se lo al es está mientras hacia desde sobre entre "
        "muy pero como cuando ella él le les o do da os em não uma ao pela pelo com sua seu "
        "des du et dans avec sur une est elle il qui pour pas "
        "gli della di che per sono nel nella "
        "der die das und mit ist ein eine sie er nicht auf zu den dem".split()) - {"a", "in", "on", "the", "as", "to", "do", "o", "die", "per", "den", "er"}

    @classmethod
    def _fallback_prompt(cls, kwargs):
        """Last resort when every attempt failed (e.g. a small local model that cannot follow the schema): a basic
        but valid H3 prompt built from the guide analysis, the references and the controls. No extra call."""
        mode = str(kwargs.get("edit_mode", "")).strip().lower()
        sequence = mode in ("transfer", "continuous edit", "elaborate plus")
        try:
            count = max(1, min(12, int(kwargs.get("continuous_scene_count", 3)))) if sequence else 1
        except (TypeError, ValueError):
            count = 1
        refs = [i for i in range(1, 5) if kwargs.get(f"reference_image_{i}") is not None]
        cached = _CLIP.get("blueprint_cache")
        blueprint = (cached[0] if cached else None) or {}

        def clean(value):
            value = re.sub(r"^KEEP GUIDE \(exactly as observed\): ", "", str(value or "").strip())
            return "" if not value or value.startswith("(not used") else value.split(" -> ")[0].strip()
        roles = [role for role in blueprint.get("roles", []) if isinstance(role, dict)]
        subjects = []
        for index in range(max(len(refs), len(roles))):
            role = roles[index] if index < len(roles) else {}
            look = ", ".join(x for x in (clean(role.get("wardrobe")), clean(role.get("hair"))) if x)
            what = clean(role.get("role"))
            if index < len(refs):
                line = (f"<Subject {index + 1}> the person in <Picture {refs[index]}>" + (f", as the {what}" if what else "")
                        + (f", wearing {look}" if look and not _CLIP.get("wardrobe_keep") else "")
                        + f". Identity referenced from <Picture {refs[index]}>.")
            else:
                line = f"<Subject {index + 1}> {what or 'a person'}" + (f", {look}" if look else "") + "."
            subjects.append(line)
        if not subjects:
            subjects = ["<Subject 1> the main character."]
        controls = [line.strip() for line in str(kwargs.get("direction_context") or "").splitlines()
                    if ":" in line and not re.match(r"(?i)\s*(Adults only|Palette: follows|Coverage|Dialogue density|Video task|LoRA)", line)]
        request = str(_CLIP.get("request") or kwargs.get("edit_request") or "").strip()  # the user's own words
        tokens = re.findall(r"[a-z\u00e0-\u00ff']+", request.lower())
        english_words = sum(t in cls._ENGLISH_PROSE for t in tokens)
        foreign_words = sum(t in cls._FOREIGN_PROSE and t not in cls._ENGLISH_PROSE for t in tokens)
        english_request = english_words >= 1 and foreign_words == 0
        parts = [clean(blueprint.get(k)) for k in (("setting", "lighting") if _CLIP.get("basic")
                                                    else ("activity", "setting", "lighting", "camera"))]
        base = " ".join(x.rstrip(".") + "." for x in parts if x)
        if request and request != cls.PLACEHOLDER_REQUEST:
            base = (base + " " + (request if english_request else
                                  f'Original user request (keep its content): "{request}"')).strip()
        if controls:
            base = (base + " Direction: " + "; ".join(controls) + ".").strip()
        if not base:
            base = "<Subject 1> moves naturally through the scene, framed in a steady medium shot."
        sound = "N/A" if _CLIP.get("keep_audio") else "Natural ambience of the place with Foley of the visible movement."
        scenes = []
        for number in range(1, count + 1):
            step = f" Part {number} of {count} of the same continuous action." if count > 1 else ""
            scenes.append("[reference generation]\n\nsubject_definitions:\n" + "\n".join(subjects)
                          + "\n\nintegrated_multimodal_description:\n[Shot 1] " + base + step
                          + "\n\noverall_soundscape:\n" + sound + "\n\nnon_diegetic_music:\nN/A")
        if _CLIP.get("i2v") and scenes:  # the opening frame is Picture 1: official I2V layout
            scenes[0] = cls._to_i2va(cls.I2V_ALIGNMENT + "\n\n" + scenes[0])
        prompt = json.dumps({"scene_prompts": scenes}, ensure_ascii=False) if sequence else scenes[0]
        return prompt, count

    @classmethod
    def _prose_not_english(cls, raw):
        """True only on positive evidence: a prose line (outside <d>) made mostly of function words of another
        language. Terse English (music, sound, framing notes) never triggers it."""
        texts = [raw]
        try:
            data = json.loads(raw) if isinstance(raw, str) and raw.lstrip().startswith("{") else None
            if isinstance(data, dict) and isinstance(data.get("scene_prompts"), list):
                texts = [item for item in data["scene_prompts"] if isinstance(item, str)]
        except ValueError:
            pass
        for text in texts:
            text = re.sub(r"<d>.*?</d>", " ", str(text), flags=re.S)
            text = re.sub(r"<[^>]+>|\[[^\]]+\]", " ", text)
            for line in text.split("\n"):
                tokens = re.findall(r"[a-z\u00e0-\u00ff']+", line.lower())
                if len(tokens) < 25:
                    continue
                foreign = sum(t in cls._FOREIGN_PROSE for t in tokens)
                english = sum(t in cls._ENGLISH_PROSE for t in tokens)
                if foreign / len(tokens) >= 0.15 and foreign > 2 * english:
                    return True
        return False

    @classmethod
    def _language_warning(cls, raw, direction_text):
        """Warn (no extra call) when lines tagged in a non-English language read as English."""
        match = re.search(r"(?im)^Dialogue:\s*(.+)$", str(direction_text or ""))
        lang = match.group(1).strip() if match else ""
        if not lang or lang.lower() in ("english", "no dialogue", "none", "auto"):
            return ""
        bad = 0
        for tag, words in re.findall(r"<d>\[([^\]]+)\]\s*(.*?)</d>", str(raw), flags=re.S):
            tokens = re.findall(r"[a-z']+", words.lower())
            if tokens and sum(t in cls._ENGLISH for t in tokens) / len(tokens) >= 0.3:
                bad += 1
        return f" · ⚠ {bad} dialogue line(s) look English, not {lang}" if bad else ""

    _ADULT_OUTFITS = frozenset({"Lingerie", "Corset / Bustier", "Latex / Leather", "Fishnet / Rave", "Babydoll",
                                "Silk Robe / Kimono"})

    OUTFIT_RESTYLE_NOTE = (
        "Outfit style {styles} with Wardrobe Keep Reference (binding): each character keeps the clothing of their "
        "own reference picture, restyled in that style ({cues}). Keep what makes it recognizable: the same pieces, "
        "colours, fabrics, prints and accessories. Change only the cut to the style: rise, length, fit, neckline and "
        "details (for example the same jeans become low-rise, the same top is cropped, in the same colour and "
        "fabric); the change of cut must be obvious at first glance, never a copy of the picture's outfit. When the style is a complete uniform or costume (maid, nurse, police, flight attendant, military, "
        "cheerleader, cosplay, superhero, pirate...), make that uniform in the colours and materials of the "
        "reference clothing so it visibly comes from it. Restyling never removes clothing and never shows more skin "
        "than the request allows; the user's own words about clothes win. Every character is an adult.")

    KEEP_REFERENCE_RESTYLE = (
        "Wardrobe Keep Reference + Outfit style (binding): the reference picture gives each character's pieces, "
        "colours, fabrics, prints and accessories; the Outfit style gives the cut. The result must look clearly "
        "restyled at first glance, never the picture's outfit copied unchanged: write each restyled garment in Target "
        "appearance with its new cut words (low-rise, cropped, mini, fitted, oversized...), and mark the outfit "
        "partially_preserved (colours and fabrics kept, cut changed) in retention_analysis.")

    @classmethod
    def outfit_rule(cls, styles, direction_text):
        """Outfit style rule for the wardrobe in force: designed clothes, or the reference clothes restyled."""
        styles = [s for s in styles if s]
        if not styles:
            return ""
        text = str(direction_text or "")
        cues = "; ".join(f"{s}: {cls.OUTFIT_CUES.get(s.lower(), s)}" for s in styles)
        if re.search(r"(?im)^Wardrobe:\s*Keep Guide", text):
            return ""
        if re.search(r"(?im)^Wardrobe:\s*Keep Reference", text):
            return cls.OUTFIT_RESTYLE_NOTE.format(styles=" + ".join(styles), cues=cues)
        return cls.OUTFIT_NOTE.format(styles=" + ".join(styles), cues=cues)

    @classmethod
    def _random_picks(cls, kwargs):
        """Controls set to Random (Genre, Situation, Visual look, Outfit style, Era) get a random pick, repeatable with
        the same seed, in Normal / Compact. The request always wins over a pick."""
        ctx = str(kwargs.get("direction_context") or "")
        wanted = (re.search(r"(?im)^Random pick:\s*(.+)$", ctx) or [None, ""])[1]
        if str(kwargs.get("edit_mode", "")).strip().lower() != "elaborate plus":
            if wanted:
                _CLIP["random_ignored"] = True  # Transfer keeps the guide's own design
            return ""
        labels = [label for label in ("Genre", "Situation", "Motion style", "Visual look", "Outfit style", "Era")
                  if label in wanted and not re.search(rf"(?im)^{label}:\s*\S", ctx)]
        if not labels:
            return ""
        import random
        from . import direction_controls_fix as _dc
        request = str(kwargs.get("edit_request") or "")
        adult = bool(cls._ADULT_REQUEST.search(request) or cls._NUDITY_REQUEST.search(request))
        rng = random.Random(int(kwargs.get("seed") or 0))
        pools = {
            "Genre": [g for g in _dc.GENRE_CHOICES if g not in ("POV / First-Person", "Webcam / Cam Show")],
            "Situation": [n for n in _dc.SITUATIONS if adult or n not in _dc.ADULT_SITUATIONS],
            "Motion style": [m for m in _dc.MOTION_CHOICES if m not in _dc.SINGLE_CAMERA_MOTIONS and "POV" not in m],
            "Visual look": [k for k in _dc.LOOK_CHOICES if k not in _dc.SINGLE_CAMERA_LOOKS],
            "Outfit style": [o for o in _dc.OUTFIT_STYLES if adult or o not in cls._ADULT_OUTFITS],
            "Era": [e for e in _dc.ERA_CHOICES if e not in ("Auto", "Present day")],
        }
        lines, picked, extra = [], [], []
        for label in labels:
            choice = rng.choice(pools[label])
            lines.append(f"{label}: {choice}")
            picked.append(f"{label} {choice}")
            if label in ("Visual look", "Motion style"):
                _CLIP["keep_shallow"] = True  # a drawn look or camera keeps its own depth of field
            if label == "Motion style":
                try:
                    from .story_director import MOTION_STYLES
                    spec = cls.FIX_MOTIONS.get(choice) or MOTION_STYLES.get(choice)
                    if spec:
                        extra.append(f"Motion style {choice}: {spec}")
                except Exception:
                    pass
            if label == "Visual look":
                try:
                    from .story_director import VISUAL_LOOKS
                    spec = (cls.FIX_LOOKS[choice] + cls.CAPTURE_ONLY) if choice in cls.FIX_LOOKS else VISUAL_LOOKS.get(choice)
                    if spec:
                        extra.append(f"Visual look {choice}: {spec} It also decides the palette.")
                except Exception:
                    pass
        _CLIP["random_picks"] = picked
        outfit = [line.partition(":")[2].strip() for line in lines if line.startswith("Outfit style:")]
        lines = [line for line in lines if not line.startswith("Outfit style:")]
        if outfit:
            extra.append(cls.outfit_rule(outfit, ctx))
            if re.search(r"(?im)^Wardrobe:\s*Keep Reference", ctx):
                _CLIP["restyle_outfit"] = outfit[0]
        rules = " ".join([e for e in extra if e] + cls.control_notes("\n".join(lines)))
        return ("OPEN PICKS (drawn at random because these controls are set to Random): " + "; ".join(picked)
                + ". Use them when the request leaves that aspect open; when the request already sets its own situation, "
                "place, tone, movement, look, clothes or period, the request wins and that pick is dropped. " + rules)

    @classmethod
    def _control_rules_text(cls, direction_text):
        notes = cls.control_notes(direction_text)
        if _CLIP.get("axes_note"):
            notes.append(_CLIP["axes_note"])
        if _CLIP.get("extra_rule"):
            notes.append(_CLIP["extra_rule"])

        return ("\nDIRECTION CONTROL RULES: " + " ".join(notes)) if notes else ""

    @classmethod
    def control_notes(cls, direction_text):
        """Meaning of the extra Direction Controls lines, as short rules for the system prompt."""
        notes = []
        for raw in str(direction_text or "").splitlines():
            label, _, value = raw.partition(":")
            label, value = label.strip(), value.strip().lower()
            if label in ("Dialogue", "Dialogue density") and _CLIP.get("keep_audio") and (_CLIP.get("mode") == "transfer" or _CLIP.get("video_edit")):
                continue  # the source audio is kept: no dialogue rules
            table = cls.CONTROL_SEMANTICS.get(label)
            if table:
                key = next((k for k in table if value.startswith(k)), None)
                if key:
                    notes.append(table[key])
                    if label == "Intensity":
                        notes.append(cls.NSFW_CAST_NOTE)
                        notes.append(cls.INTENSITY_SCOPE)
                        if re.search(r"(?im)^Genre:\s*\S", str(direction_text or "")):
                            notes.append(cls.ADULT_GENRE_NOTE)
                    elif label == "Violence":
                        notes.append(cls.VIOLENCE_SCOPE)
            elif label == "Action chain":
                if value in cls.ACTION_CHAIN_NOTES and _CLIP.get("mode") != "transfer":
                    notes.append(cls.ACTION_CHAIN_NOTES[value])
            elif label == "Action density":
                if value.startswith("very"):
                    notes.append(cls.VERY_HIGH_NOTE)
                else:
                    key = "high" if value.startswith("high") else "low" if value.startswith("low") else None
                    if key:
                        notes.append(cls.ACTION_DENSITY_NOTE[key])
            elif label == "Story pace":
                if value in cls.STORY_PACE_NOTES and _CLIP.get("mode") != "transfer":
                    notes.append(cls.STORY_PACE_NOTES[value])
                    if value != "slow" and _CLIP.get("single_take"):
                        notes.append(cls.STORY_PACE_SINGLE)
            elif label == "Situation":
                name = raw.partition(":")[2].strip()
                cue = cls.SITUATION_CUES.get(value)
                if cue:
                    notes.append(cls.SITUATION_NOTE.format(name=name, cue=cue.removeprefix("18+: ")))
                    if cue.startswith("18+") and not re.search(r"(?im)^Intensity:\s*\S", str(direction_text or "")):
                        notes.append(cls.SITUATION_ADULT_NOTE)
                    if _CLIP.get("mode") == "transfer":
                        notes.append(cls.SITUATION_TRANSFER_NOTE)
            elif label == "Progression":
                target = next((line.partition(":")[2].strip() for line in str(direction_text).splitlines()
                               if line.partition(":")[0].strip() == "Intensity"), "")
                notes.extend(cls.arc_note(value, direction_text, _CLIP.get("scenes", 1)))
                notes.extend(cls.progression_note(value, target, _CLIP.get("scenes", 1)))
            elif label in cls.KEEP_GUIDE_NOTES and value.startswith("keep guide"):
                notes.append(cls.KEEP_GUIDE_NOTES[label])
            elif (label in cls.EDIT_KEEP_ASPECTS and value.startswith("keep reference") and _CLIP.get("task") == "edit"
                  and _CLIP.get("mode") == "transfer"):
                notes.append(cls.EDIT_KEEP_NOTE.format(label=label, what=cls.EDIT_KEEP_ASPECTS[label]))
                if label == "Wardrobe":
                    notes += [cls.EDIT_WARDROBE_CHANGES, cls.ANATOMY_NOTE, cls.COVERED_ANATOMY_NOTE]
            elif label == "Wardrobe" and value.startswith("guide garments") and _CLIP.get("mode") == "transfer":
                notes += [cls.WARDROBE_FOLLOW_NOTE, cls.ANATOMY_NOTE, cls.COVERED_ANATOMY_NOTE]
            elif label in cls.KEEP_REFERENCE_NOTES and value.startswith("keep reference"):
                if label == "Wardrobe" and (_CLIP.get("restyle_outfit") or re.search(
                        r"(?im)^(?:Outfit style:\s*\S|Random pick:.*Outfit style)", str(direction_text or ""))):
                    notes.append(cls.KEEP_REFERENCE_RESTYLE)
                else:
                    notes.append(cls.KEEP_REFERENCE_NOTES[label])
                if cls.CONTROL_PRIORITY_NOTE not in notes:
                    notes.append(cls.CONTROL_PRIORITY_NOTE)
            elif label == "Hair & Makeup" and value.startswith("create new"):
                notes.append(cls.HAIR_NEW_NOTE)
            elif label == "Wardrobe" and value.startswith("create new"):
                notes.append(cls.WARDROBE_NEW_NOTE)
            elif label == "Environment" and value.startswith("create new"):
                notes.append(cls.ENVIRONMENT_NEW_NOTE)
            elif label == "Body proportions" and value.startswith("creative"):
                notes.append(cls.BODY_NEW_NOTE)
            elif label == "Era":
                era = raw.partition(":")[2].strip()
                if era.lower().startswith("present"):
                    notes.append(cls.PRESENT_NOTE)
                elif era in cls.HISTORIC_CUES:
                    notes.append(cls.HISTORIC_NOTE.format(era=era, cues=cls.HISTORIC_CUES[era]))
                elif era in cls.ERA_CUES:
                    notes.append(cls.ERA_NOTE.format(era=era, cues=cls.ERA_CUES[era]))
            elif label == "Dialogue":
                lang = raw.partition(":")[2].strip()
                if lang and lang.lower() not in ("no dialogue", "none", "auto"):
                    notes.append(cls.LANGUAGE_NOTE.format(lang=lang))
            elif label == "Coverage" and value.startswith("single"):
                notes.append("Coverage Single take: every scene is ONE continuous shot with no cuts and no [Shot 2]; "
                             "the camera may move, but never cut."
                             + (" Each scene is generated from the last frames of the previous one, so it picks up at "
                                "that exact moment, place and camera position: no jump in time or place; the place "
                                "changes only when the characters move and the camera follows them."
                                if _CLIP.get("mode") in ("continuous edit", "elaborate plus") else ""))
            elif label == "Coverage" and value.startswith("dynamic"):
                try:
                    seconds = float(_CLIP.get("seconds") or 10.0)
                except (TypeError, ValueError):
                    seconds = 10.0
                notes.append(cls.DYNAMIC_COVERAGE_NOTE.format(shots=cls.max_shots(seconds)))
            elif label == "Outfit style":
                styles = [s.strip() for s in raw.partition(":")[2].split(" + ") if s.strip()]
                rule = cls.outfit_rule(styles, direction_text)
                if rule:
                    notes.append(rule)
            elif label == "Coverage" and (value.startswith("calm") or value.startswith("moderate")):
                try:
                    seconds = float(_CLIP.get("seconds") or 10.0)
                except (TypeError, ValueError):
                    seconds = 10.0
                shots = cls.max_shots(seconds)
                notes.append(("Coverage Calm (binding): few, unhurried cuts, about one every 10-15 s; long held shots "
                              "where the camera moves inside the shot (push-ins, drifts, pans) instead of cutting"
                              if value.startswith("calm") else
                              "Coverage Moderate (binding): a cut about every 6-8 s, each to a clearly different size or "
                              "angle with a reason (a reaction, a detail, a change of space)")
                             + f"; this clip has {shots} shot{'s' if shots > 1 else ''}"
                             + (" (one continuous shot)" if shots == 1 else "") + "."
                             + (" With the fixed-device look, each cut switches to another fixed device of that kind "
                                "(another security camera, webcam or dashcam), never to a film camera."
                                if "fixed device" in value else ""))
            elif label == "Palette":
                notes.append("Palette follows the selected Visual look, which beats any default colour habit.")
            elif label == "Adults only":
                notes.append("Adults only: every character is an adult (18+); any sexual content is between consenting adults.")
            elif label == "Visual look":
                notes.append("The selected Visual look must be visible in the first sentence of "
                             "integrated_multimodal_description as concrete capture traits (lens, grain, light, colour).")
            elif label == "Motion style":
                notes.append("The selected Motion style must be visible in what the people and camera do in every scene, not only named.")
        if not re.search(r"(?im)^(?:Intensity|Nudity):\s*\S", str(direction_text or "")) and (
                _CLIP.get("reference_nude") or cls._NUDITY_REQUEST.search(str(_CLIP.get("request") or ""))
                or cls._ADULT_REQUEST.search(str(_CLIP.get("request") or ""))):
            notes += [cls.ADULT_FREE_NOTE if cls._ADULT_REQUEST.search(str(_CLIP.get("request") or ""))
                      else cls.PARTIAL_NUDITY_NOTE, cls.ANATOMY_NOTE, cls.COVERED_ANATOMY_NOTE]  # asked in words
        elif (not re.search(r"(?im)^(?:Intensity|Nudity):\s*\S", str(direction_text or ""))
              and cls._SUGGESTIVE_REQUEST.search(str(_CLIP.get("request") or ""))):
            notes += [cls.SUGGESTIVE_NOTE, cls.COVERED_ANATOMY_NOTE]  # sexy, not explicit
        bare = cls.nudity_key(direction_text)
        if bare:  # how much skin: the Nudity control, or the NSFW level's default
            explicit_control = bool(re.search(r"(?im)^Nudity:\s*\S", str(direction_text or "")))
            notes.append(cls.NUDITY_NOTES[bare] + ("" if explicit_control else
                                                   " (the default ceiling for this NSFW level, not the starting outfit: people first wear what the "
                                                   "situation and the place call for and reach this level on screen, by "
                                                   "undressing, when the story gets there; the Nudity control can change it)"))
            notes.append(cls.COVERED_ANATOMY_NOTE)
            if bare in ("topless", "nude", "explicit") and cls.ANATOMY_NOTE not in notes:
                notes.append(cls.ANATOMY_NOTE)
        loras = cls.lora_names(direction_text)
        styles = [name for name in loras if cls.lora_is_style(name)]
        actions = [name for name in loras if name not in styles]
        if actions:
            notes.append(cls.LORA_NOTE.format(names=" and ".join(f"'{name}'" for name in actions)))
            described = cls.lora_notes(direction_text)
            for name in actions:
                if described.get(name):
                    notes.append(f"LoRA '{name}' (what the action looks like WHEN it happens: follow it for who does "
                                 "what, position, body parts, camera angle and rhythm in those moments, adapted to the "
                                 "cast and controls; it is not the whole scene and does not replace the story): "
                                 f"{described[name]}")
            if not re.search(r"(?im)^(?:Intensity|Nudity):\s*\S", str(direction_text or "")) and cls.ANATOMY_NOTE not in notes:
                notes += [cls.ANATOMY_NOTE, cls.COVERED_ANATOMY_NOTE]
        if styles:
            notes.append(cls.LORA_STYLE_NOTE.format(names=" and ".join(f"'{name}'" for name in styles)))
            described = cls.lora_notes(direction_text)
            for name in styles:
                if described.get(name):
                    notes.append(f"Style LoRA '{name}' looks like: {described[name]}")
        if (_CLIP.get("mode") and _CLIP.get("mode") != "transfer"  # Transfer and Edit keep the video's own ending
                and not _CLIP.get("video_edit")):
            notes.append(cls.ENDING_NOTE)
        if (cls._sings(str(_CLIP.get("request") or "")) and not _CLIP.get("keep_audio")
                and not re.search(r"(?im)^Dialogue:\s*No dialogue", str(direction_text or ""))):
            notes.append(cls.SINGING_NOTE)
        if _CLIP.get("i2v") and cls._DESIGN_CONTROLS.search(str(direction_text or "")):
            notes.append(cls.I2V_CONTROLS_NOTE)
        if _CLIP.get("mode") == "transfer" and _CLIP.get("task"):
            notes[:0] = [cls.VIDEO_TASK_BASE.format(task=_CLIP["task"].capitalize()), cls.VIDEO_TASK_NOTES[_CLIP["task"]]]
        if _CLIP.get("mode") == "transfer":  # Transfer: the controls take part in the transferred footage
            text = str(direction_text or "")
            lang = re.search(r"(?im)^Dialogue:\s*(.+)$", text)
            if lang and lang.group(1).strip().lower() not in ("no dialogue", "none", "auto") and not _CLIP.get("keep_audio"):
                notes.append(cls.TRANSFER_DIALOGUE_NOTE)
            if re.search(r"(?im)^Intensity:\s*\S", text):
                notes.append(cls.TRANSFER_NSFW_NOTE)
            if re.search(r"(?im)^Violence:\s*\S", text):
                notes.append("VIOLENCE IN TRANSFER: the level charges the guide's existing physical action (impacts, "
                             "injuries, blood) without inventing fights, events or people.")
            if re.search(r"(?im)^(Motion style|Action density|Coverage):\s*\S", text):
                notes.append(cls.TRANSFER_PACE_NOTE)
        return notes

    @classmethod
    def max_shots(cls, seconds):
        """3 shots up to 15 s, then about one shot per 7 s (20 s: 3, 30 s: 4, 45 s: 6, 60 s: 8); with Coverage
        Dynamic about one shot per 4 s (10 s: 3, 20 s: 5, 30 s: 8, capped at 8)."""
        rate = _CLIP.get("cut_every")
        if rate:  # Coverage Calm / Moderate / Dynamic (or a fast Story pace): one cut per interval started
            import math
            return max(1, min(8, math.ceil(float(seconds) / rate - 1e-9)))
        return 3 if float(seconds) <= 15 else max(3, min(8, round(float(seconds) / 7.5)))

    ENDING_NOTE = (
        "A COMPLETE STORY (binding): however short, the story has a beginning, a turn and an ending that lands. The "
        "only clip, or the last scene of a sequence, closes it inside its own time: the action is finished, the last "
        "line is said and answered or reacted to, a song reaches its final note, and the last 2-3 seconds hold the "
        "outcome as a settled image (a look, a smile, a kiss, an exit, a breath, the aftermath); never end mid-motion, "
        "mid-sentence, on a setup or on a question left open. This applies to the final clip only: every earlier scene "
        "of a sequence completes its own beat (something is achieved, said or changed) and then ends on a movement "
        "already running that the next scene resumes, as the handoff rules say.")
    DYNAMIC_COVERAGE_NOTE = (
        "Coverage Dynamic (binding): lively, edited coverage: a real cut about every 3-5 s (never more than {shots} "
        "shots in this clip), each shot a different size or angle with a purpose: wide two-shot to set the space, "
        "medium and close shots, shot / reverse shot and over-the-shoulder when people talk, reaction shots (often "
        "the listener's face matters more than the line), inserts of hands, objects or details. Cut on a line, a "
        "reaction or a movement; the sound can lead into the next shot (mark speech that runs over the cut with "
        "`<scenetrans>`). People never stand still to talk: they lean "
        "in, gesture, laugh, touch their face, drink, shift, interrupt; every exchange changes something (a turn of "
        "topic, a laugh, a disagreement, a confession, more tension or complicity) and the clip builds to a small "
        "landing instead of 'they keep talking'. Lines are short and answer each other, with pauses filled by "
        "reactions. Between cuts the camera stays alive (a slow push-in on the key line, a rack focus, a light "
        "handheld drift). Each new shot names who is in it by label (<Subject N>), where they are and their posture as the "
        "previous shot left it (still seated, still standing), briefly, without repeating their "
        "description.")
    OUTFIT_CUES = {
        "casual": "relaxed everyday basics (tees, denim, sneakers)", "urban streetwear": "oversized hoodies, cargo pants, sneakers, caps, bold logos",
        "y2k": "early-2000s pop: low-rise, baby tees, rhinestones, butterfly clips, tinted glasses",
        "ultra low-rise": "ultra low-rise jeans, cargo pants, flared trousers or track pants sitting low on the hip bones (a low-rise mini skirt only now and then), cropped tops, exposed midriff",
        "hip-hop": "baggy denim, jerseys, gold chains, sneakers, bucket hats", "skater": "loose tees, cargo shorts, skate shoes, beanies",
        "grunge": "flannel, ripped denim, band tees, combat boots", "punk / rock": "leather jacket, studs, ripped tights, boots",
        "goth": "black lace and velvet, corsets, chokers, dark makeup accents", "alt / e-girl": "pastel and black layers, chains, striped sleeves, platform shoes",
        "boho": "flowing prints, fringe, crochet, earthy layers", "minimalist": "clean lines, neutral colours, few pieces",
        "sophisticated": "tailored, refined fabrics, understated jewelry", "old money / quiet luxury": "cashmere, linen, camel and cream, no logos",
        "cocktail dress": "short elegant dress, heels, clutch", "evening gown": "floor-length gown, satin or sequins, formal jewelry",
        "red carpet glam": "dramatic couture gown or sharp tux, statement jewelry", "business / office": "blazer, blouse or shirt, pencil skirt or trousers",
        "power suit": "sharp tailored suit, strong shoulders, heels or loafers", "yoga": "fitted leggings, sports bra or crop top, barefoot",
        "gym / sportswear": "training shorts or leggings, tank tops, sneakers", "athleisure": "matching sweat sets, leggings, sleek sneakers",
        "running": "running shorts, light tops, running shoes", "tennis": "pleated tennis skirt or shorts, polo, visor",
        "swimwear / bikini": "bikinis, swimsuits, cover-ups, sandals", "cheerleader (adult)": "adult cheer squad uniform, pleated skirt, crop top, pompoms",
        "maid": "classic maid uniform: black dress, white apron, headpiece", "nurse": "nurse uniform or scrubs",
        "police": "police uniform: shirt, badge, belt, cap", "flight attendant": "airline uniform: fitted jacket, scarf, pencil skirt",
        "military": "military-style fatigues or dress uniform", "secretary": "fitted blouse, pencil skirt, glasses, heels",
        "waitress / bartender": "service uniform: apron, vest or fitted black outfit", "idol (k-pop stage)": "coordinated K-pop stage outfits: sparkle, harness details, statement boots",
        "cosplay / anime": "an original anime-inspired costume (not a real character)", "cyberpunk": "neon accents, techwear, vinyl, LED details",
        "steampunk": "Victorian layers, corsets, brass gadgets, goggles", "fantasy / elf": "flowing fantasy garments, leather and leaf details",
        "superhero": "an original hero suit (not a real character)", "pirate": "pirate coats, loose shirts, corsets, boots",
        "lingerie": "lingerie sets: bra, panties, garters, stockings", "corset / bustier": "structured corset or bustier with skirt or trousers",
        "latex / leather": "glossy latex or leather pieces", "clubwear / bodycon": "tight bodycon dresses, mini skirts, heels",
        "fishnet / rave": "fishnets, neon, mesh, rave accessories", "silk robe / kimono": "silk robe or kimono-style wrap",
        "babydoll": "sheer babydoll nightie", "western / cowgirl": "denim, boots, cowboy hats, fringe",
        "festival": "festival looks: crochet, glitter, shorts, statement accessories", "tropical / beachwear": "light linen, sarongs, open shirts, sandals"}
    SITUATION_CUES = {
        'first date': 'a restaurant, bar or café; two people meeting properly for the first time; menus, drinks, phones; nerves, testing questions, the moment the guard drops',
        'blind date': 'a busy restaurant or bar; two strangers set up by friends; finding each other, mistaken identity, a first impression to overcome',
        'meet-cute': 'a public place (street, shop, café); a small accident or mix-up brings two strangers together; spilled drink, dropped bag, the same item grabbed',
        'anniversary dinner': 'a candlelit table at home or a restaurant; a long-time couple; a gift, a toast, an old memory brought back',
        'proposal': 'a meaningful place; one partner plans the surprise; the ring, kneeling, the answer',
        'wedding night': 'a hotel suite or bedroom after the wedding; newlyweds still half in wedding clothes; the dress, champagne, finally alone',
        'ex reunion': 'an unexpected place (party, street, bar); former partners meet again; old wounds, chemistry, an unfinished question',
        'breakup': 'an apartment, car or café; a couple ending things; packed bag, keys returned, the last words',
        'make-up after a fight': 'home after an argument; a couple; cold silence, an apology, the reconciliation',
        'jealousy': 'a party, bar or home; a partner sees the other with someone; looks, a confrontation, reassurance or escalation',
        'caught cheating': "a bedroom, office or hotel; a partner walks in or finds proof; the phone, the stranger's item, denial and fallout",
        'dentist': 'a dental clinic: reclining chair, overhead lamp, instruments, bib; a nervous patient and the dentist; the power gap of the chair, the mouth that cannot speak',
        'doctor check-up': 'an exam room: exam table, stethoscope, gown; patient and doctor; questions, the exam, a diagnosis or surprise',
        'massage / spa': 'a dim spa room: massage table, oils, towels, candles; client and masseur/masseuse; relaxation, touch, a line approached',
        'hair salon': 'a salon chair, mirror, scissors, washing basin; client and stylist; gossip, the reveal in the mirror',
        'tattoo studio': 'a tattoo chair, needle, stencil, inked artist; client and artist; pain, trust, the finished design',
        'personal trainer': 'a gym or home gym; client and trainer; spotting, corrections by touch, pushing the limit',
        'job interview': 'an office: desk, CV, chairs facing; candidate and interviewer; tough questions, a slip, the decision',
        'casting / audition': "a casting room or stage: camera on a tripod, marks on the floor, a director's table; performer and casting team; the scene read, a request, the callback",
        'photo shoot': 'a studio: lights, backdrop, camera; model and photographer; poses directed, the perfect shot',
        'tailor fitting': 'a fitting room or atelier: mirror, measuring tape, pins; client and tailor; measurements, adjustments, closeness',
        'house viewing': 'an empty or staged house; buyer and real-estate agent; room by room, a hidden flaw or a perfect room',
        'delivery at the door': 'a front door and hallway; resident and delivery person; the package, a signature, an invitation inside',
        'plumber / repair visit': 'a kitchen or bathroom; resident and repair worker; the broken pipe or appliance, tools, a mess, payment',
        'room service': 'a hotel room; guest and staff; the cart, the order, the tip',
        'new neighbor': 'a doorway, hallway or shared fence; the newcomer and a neighbor; borrowing something, curiosity, an invitation',
        'landlord visit': 'a rented apartment; tenant and landlord; rent, an inspection, a deal',
        'pool cleaner': 'a backyard pool; homeowner and pool cleaner; sun, water, the skimmer net, watching from the lounger',
        'robbery / hold-up': 'a store, bank or street; robber and victims; the weapon, the bag, the till, a hero or a mistake',
        'heist': 'a vault, museum or casino; a crew with a plan; blueprints, gadgets, the alarm, the getaway',
        'interrogation': 'an interrogation room: table, lamp, one-way mirror; suspect and investigator; pressure, a lie, the confession',
        'traffic stop': 'a roadside at night: car, flashing lights; driver and officer; licence, nerves, what is in the car',
        'hostage standoff': 'a building under siege; captor, hostages, negotiator; demands, a phone line, the breaking point',
        'chase': 'streets, rooftops or roads; pursuer and pursued; obstacles, near misses, the catch or the escape',
        'blackmail': 'a private meeting place; blackmailer and victim; the evidence, the demand, the counter-move',
        'stakeout': 'a parked car or a window; two watchers; coffee, binoculars, boredom, the target appears',
        'undercover': 'a club, office or gang den; an infiltrator among targets; a fake identity, a near exposure',
        'house party': 'a crowded house: music, red cups, kitchen, hallway; friends and strangers; a game, a rumour, a corner away from the crowd',
        'nightclub': 'a club: dance floor, lights, bar, VIP booth; dancers and strangers; eye contact across the room, a dance',
        'bar pickup': 'a bar counter; two strangers; an offered drink, a line, a decision to leave together or not',
        'office party': 'an office after hours: decorations, drinks, desks; colleagues; loosened ties, a secret, a slip in front of the boss',
        'karaoke night': 'a karaoke bar or booth; friends; the microphone, a bold song choice, cheering',
        'game night': 'a living room: board game or cards, snacks; friends or couples; competition, cheating, a dare',
        'bachelor / bachelorette party': 'a party room, club or suite; the guest of honour and friends; sashes, drinks, a surprise act',
        'strip club': 'a strip club: stage, pole, neon, booths; dancers and customers; the stage show, a private request',
        'morning after': 'a bedroom in morning light; two people who spent the night; tangled sheets, coffee, remembering last night',
        'cooking together': 'a kitchen; a couple or friends; ingredients, flour, tasting from the spoon, a mess',
        'moving day': 'a half-empty home: boxes, tape, furniture; movers or a couple; heavy lifting, a found memory',
        'power outage': 'a home or building in the dark; people stuck together; candles, phone lights, closeness',
        'stuck in an elevator': 'a stalled elevator; strangers or rivals; the alarm button, heat, waiting, forced closeness',
        'locked out': 'a doorstep or balcony; someone locked out, a neighbor or partner; keys, climbing in, embarrassment',
        'rainy day in': 'a cozy home while it rains; a couple or friends; blankets, windows streaked with rain, idle play',
        'road trip': 'a car on the open road; friends or a couple; maps, snacks, music, a detour',
        'hitchhiker': "a roadside and a car; driver and hitchhiker; the thumb, the lift, the stranger's story",
        'train compartment': 'a private train compartment; strangers or lovers; passing landscape, the rocking motion, a closed door',
        'flight': 'an airplane cabin; passengers and crew; turbulence, a seat neighbor, the restroom',
        'hotel check-in': 'a hotel lobby and room; guests and receptionist; the key card, a single bed instead of two',
        'beach day': 'a beach: sand, sea, towels, sunscreen; friends or a couple; swimming, sunbathing, sunset',
        'camping': 'a tent in the woods; friends or a couple; campfire, a sleeping bag for two, night noises',
        'gas station': 'a gas station at night; travellers and an attendant; fuel, the shop, a strange encounter',
        'late at the office': 'an empty office at night; two colleagues; deadline, desk lamps, a closed blinds',
        'boss and employee': "a boss's office; boss and employee; a review, a favour, a shift of power",
        'live stream': "a streamer's room: ring light, phone or webcam, chat; the streamer talking to viewers; reactions to chat, a challenge",
        'podcast interview': 'a podcast studio: microphones, headphones; host and guest; a revealing question',
        'backstage': 'a backstage area: mirrors, costumes, crew; performers before or after a show; nerves, celebration',
        'music video shoot': 'a set with lights and playback; artist, dancers, director; takes, choreography, the money shot',
        'gym session': 'a gym: weights, mirrors, machines; gym-goers or a pair; reps, spotting, sweat',
        'yoga class': 'a yoga studio: mats, mirrors; instructor and students; poses, adjustments, breath',
        'boxing training': 'a boxing gym: ring, bags, gloves; boxer and coach or sparring partners; pads, combinations, exhaustion',
        'locker room': 'a locker room: benches, lockers, showers; teammates or gym-goers; changing, towels, banter',
        'haunted house': 'an old house at night; visitors and a presence; creaking doors, a cold spot, the apparition',
        "vampire's visit": 'a bedroom or castle at night; a vampire and a human; the window, the invitation, the bite',
        "witch's spell": 'a cottage or altar; a witch and someone who wants something; candles, potion, a spell with a price',
        'alien encounter': 'a field, road or ship interior at night; humans and a visitor; lights, curiosity, contact',
        'dating app hookup': '18+: a first meeting after matching online; an apartment door or a bar; checking the profile against the person, fast chemistry',
        'one-night stand': '18+: strangers going home together after a night out; the stumble through the door, clothes on the floor',
        'booty call': '18+: a late-night text and a knock on the door; two people who know why they are there',
        'hotel affair': '18+: a hotel room booked in secret; two lovers with other lives; the do-not-disturb sign, stolen time',
        'sugar date': '18+: a luxury dinner or suite; a wealthy partner and a younger adult companion; gifts, an arrangement, mutual terms',
        'escort booking': '18+: an upscale hotel room; a client and a professional companion; the agreed terms, the envelope, the time',
        'seductive neighbor': "18+: a neighbor's door, balcony or shared laundry room; the attraction finally acted on",
        'delivery seduction': '18+: a delivery at the door turns into an invitation inside; the package, the tip, the offer',
        'strip tease': '18+: a performer undresses slowly for one person or an audience; music, a chair, garments dropped one by one',
        'lap dance': '18+: a seated person and a dancer straddling them; music, slow grinding, hands-off rules or not',
        'private dance': '18+: a private room in a club or a bedroom; a personal show for one; a curtain, a closed door',
        'webcam show': "18+: a performer's room on camera: ring light, laptop, live chat and tips; requests from viewers, teasing the lens",
        'content creator shoot': '18+: creators filming their own adult content: phone on a tripod, ring light, a script, retakes, laughs',
        'boudoir photo shoot': '18+: an intimate photo session in lingerie or less: bed, window light, a photographer directing poses',
        'casting couch': '18+: an adult audition: couch, camera, an interviewer; questions, a test scene, both adults agreeing on what comes next',
        'pole dance class': '18+: a pole studio: poles, mirrors; instructor and students; spins, climbs, sensual flow',
        'strip poker': '18+: a card table; friends or a couple; each lost hand costs a garment',
        'truth or dare': '18+: adults in a circle or on a bed; questions and dares that escalate',
        'roleplay night': '18+: a couple playing characters with costumes and an agreed scenario; the act, the breaking of character',
        'body painting': '18+: paint on bare skin; artist and model or a couple; brushes, hands, colour over the body',
        'sex toy unboxing': '18+: a bed or sofa; a person or couple opening and trying new toys; packaging, curiosity, laughter, testing',
        'shower together': '18+: a steamy shower; two people; water, soap, wet skin, the glass wall',
        'hot tub / sauna': '18+: a hot tub or sauna; bodies in heat and steam; towels slipping, closeness',
        'skinny dipping': '18+: a lake, pool or sea at night; clothes left on the shore, naked swimming',
        'car / back seat': '18+: a parked car; two people in the back seat; fogged windows, cramped space, the risk of being seen',
        'elevator quickie': '18+: an elevator between floors; two people with seconds to spare; the stop button, the doors',
        'mile high club': '18+: an airplane restroom or a dark cabin; two passengers; the locked door, the turbulence, the silence',
        'fitting room': '18+: a store fitting room; a couple or a customer and a lover; the curtain, the mirror, staff outside',
        'sex shop visit': '18+: an adult store: shelves of toys and lingerie; a couple or a customer and a clerk; browsing, trying, a suggestion',
        'morning sex': '18+: waking up together in bed; slow, sleepy, sunlight through curtains',
        'honeymoon': '18+: a honeymoon suite or villa; newlyweds; rose petals, the view, nothing else to do',
        'makeup sex': '18+: right after a fight; anger turning into desire',
        'movie night on the couch': '18+: a couch, a blanket, a film nobody watches anymore',
        'massage with happy ending': '18+: a massage table and oils; the massage moving to an erotic finish both agree to',
        'sunbathing / tan lines': '18+: a sun lounger or a private terrace; oil on skin, bikini lines, the strap pulled aside',
        'bdsm dungeon session': '18+: a playroom: restraints, cross, toys; dominant and submissive with an agreed safeword; control, trust, aftercare',
        'domination / submission': '18+: a consensual power exchange; commands, obedience, praise; limits agreed and respected',
        'bondage': '18+: ropes, cuffs or ties on a consenting partner; the tying, the helplessness, the release',
        'cuckold night': '18+: a consenting couple and a third adult; one partner watches by agreement; jealousy as thrill',
        'swingers party': '18+: a private party of consenting couples; masks or wristbands, open rooms, exchanges agreed by everyone',
        'exhibitionist couple': '18+: a couple who enjoy being seen by consenting onlookers or the camera; a window, a balcony, an open door, both aware and wanting it',
        'threesome invitation': '18+: a couple inviting a third adult; the proposal, hesitation, the first touch of three',
        'orgy / sex party': '18+: a private party where many adults join in; a mansion or loft, bodies, music',
        'bachelor party show': '18+: a hired adult performer at a bachelor or bachelorette party; the guest of honour in a chair, the group cheering',
    }
    SITUATION_NOTE = (
        "Situation {name} (binding premise): {cue}. Build the story inside this situation: its place, its roles, its "
        "props and its classic turn, played fresh, never as a label. The written request wins on every detail it gives "
        "(place, who, what happens); the Genre sets its tone; the request decides how far it "
        "goes. The referenced people take the situation's roles; a counterpart the situation needs (the dentist, the "
        "robber, the client, a third person for a group situation) may be added once when no reference covers it, as "
        "an adult. Every character is a consenting adult; in a situation of danger or coercion (robbery, hostage, "
        "interrogation, blackmail, kidnapping) sexual content happens only between consenting adults and never "
        "sexualises a victim or the coercion itself.")
    SITUATION_ADULT_NOTE = (
        "Adult situation: it is erotic by itself. It plays at the heat the request implies, "
        "Sensual at least (seduction, touch, suggestive talk); the request decides the skin. "
        "Consent stays visible (looks, asking, agreement), above all in kink and power-exchange situations.")
    SITUATION_TRANSFER_NOTE = (
        "Situation in Transfer: it only dresses the guide's own action (set dressing, props, wardrobe accents, sound) "
        "without new events, people or plot.")
    OUTFIT_NOTE = (
        "Outfit style {styles} (binding for designed clothing): every character whose outfit is designed in this "
        "prompt (Wardrobe Auto or Create New; with Guide Garments + Reference Style the guide's garments are restyled "
        "in this style instead of the reference's) wears a complete, coherent outfit of that style ({cues}), written "
        "as concrete garments, fabrics, colours, fit and accessories, a different variation for each character; vary the "
        "main pieces (trousers, jeans, shorts, skirts or dresses are equally valid wherever the style allows, never a "
        "skirt by default). Keep Guide outfits are not changed; the user's own words about "
        "clothes win; the Era sets the world while the clothes follow this style; the request decides how much "
        "stays on. Every character is an adult and is styled as one.")

    @classmethod
    def long_take_note(cls, seconds):
        """Clips over 15 s: enough distinct phases for the length, and anchors so nothing drifts."""
        seconds = float(seconds)
        if seconds <= 15:
            return ""
        phases = "3-4" if seconds <= 22 else "4-5" if seconds <= 32 else "5-7" if seconds <= 45 else "6-8"
        if _CLIP.get("dynamic"):
            return (f"LONG CLIP ({seconds:g} s, binding): {phases} beats that each change something, spread over the "
                    "whole clip and landing in the last seconds, never one beat stretched; describe each Subject's full "
                    "look once, at the first clear appearance, then refer to them by label (<Subject N>) and restate "
                    "only what changes. ")
        return (f"LONG CLIP ({seconds:g} s, binding): plan {phases} phases in strict chronological order, each with a "
                "visible change (a new action, position, gesture, contact, camera move or emotional turn) and enough "
                "time to read; spread them across the whole clip, never one beat stretched or repeated to fill time, "
                "and end with a clear closing state in the last seconds. One take: write the phases as one continuous "
                "flow (first..., then..., later..., by the end...) with the camera moving between them as needed. With "
                "cuts: each [Shot n] carries one or two phases. Keep everything stable without "
                "copying descriptions: each Subject's full look is written once, at the first clear appearance; "
                "afterwards use the label (<Subject N>) and state only what changes on screen (a garment removed, "
                "wet hair, a new position). The place and the light stay as first established unless the footage "
                "changes them. The anatomy and covering rules still name what each shot shows. ")

    @classmethod
    def coverage_note(cls, seconds):
        if float(seconds) >= 8.0:
            return cls.CONTINUITY_NOTE + (
                "The request decides how much to cut when it says so (one shot, few cuts, fast editing); a fixed-camera look or motion means a single take. "
                ""
                "Coverage follows information: when people talk, a clip of 8 s or more normally uses "
                "2-3 shots (for example a two-shot, then a single or over-the-shoulder on the speaker or an "
                "insert of a telling prop or gesture, then a reaction), written as [Shot 2] At 00:SS.mmm, "
                "cut to ... inside the description. In every scene except the last, the final shot is a clean, "
                "steady framing that the next scene inherits. "
                "Each cut must add new information (viewpoint, space, "
                "state or time); a pan or reframe is camera motion, not a cut. Use one shot only for an "
                "uninterrupted take, a locked source or a requested single take (a locked-off camera still allows cuts; an I2V "
                "opening only fixes the first frame, cuts may follow it). Never more than "
                + str(cls.max_shots(seconds)) + " shots in a clip, each at least 2 s long, and never write a [Shot n] marker for "
                "something that is not a cut. After every cut the people keep the posture and place the previous shot left "
                "them in (seated stays seated) unless the new shot shows them change it. "
                + cls.long_take_note(seconds)
            )
        return cls.CONTINUITY_NOTE + (
            "The request decides how much to cut when it says so. Otherwise one focused shot; a pan or reframe is camera motion, not a new [Shot n]; add a shot marker "
            "only for a distinct timed beat that adds new information. "
        )

    _LOWER_AFTER_CUT = frozenset("the a an she he they her his their it its in inside outside back meanwhile later "
                                 "now then two both everyone".split())
    _CAMERA_MOVE = re.compile(
        r"(?:the )?camera\b|(?:then |slowly |gently |quickly |smoothly )?(?:pans?|tilts?|dolly|dollies|zooms?|"
        r"pushes in|pulls (?:back|out)|tracks?|trucks?|cranes?|orbits?|arcs?|racks? focus|reframes?|drifts?|"
        r"glides?|follows (?:her|him|them))\b|the (?:shot|frame|framing|lens)\b", re.IGNORECASE)
    _VIEWPOINT = re.compile(
        r"(?:an? )?(?:extreme |tight |medium |wide |full |long |low |high |overhead |top |side |back |"
        r"close|over[- ]the[- ]shoulder|ots\b|insert|pov|reverse|profile|two-shot|single|reaction|master|"
        r"dutch|handheld close)", re.IGNORECASE)

    @staticmethod
    def _timecode_shots(text, seconds, store=None):
        """Keep only real cuts as '[Shot N] At 00:SS.mmm, cut to ...': a marker that only describes camera
        motion or action is merged into the previous shot, a cut without a time gets one, and the
        survivors are renumbered."""
        cls = H3CompactMultimodalEditDirectorFix
        # an unmarked 'At 00:SS.mmm, cut to ...' is a real cut: give it a marker; an unmarked timed camera
        # move is just motion: drop the stray timestamp
        text = re.sub(r"(?<!\] )(?<!\])\bAt (\d\d:\d\d(?:\.\d+)?),?\s+(?=cut to\b)",
                      r"[Shot 99] At \1, ", text)
        text = re.sub(r"(?<!\] )(?<!\])\bAt \d\d:\d\d(?:\.\d+)?,?\s+(?=[Tt]he camera\b)", "", text)
        def official(m):  # MiniMax's own wording ('the camera cuts to', 'the shot transitions to') is a cut
            before = text[max(0, m.start() - 40):m.start()]
            marked = re.search(r"\[Shot \d+\](?:\s*At \d\d:\d\d(?:\.\d+)?,?)?\s*$", before)
            return ("" if marked else "[Shot 99] ") + ("cut back to" if m.group(1) else "cut to")
        text = re.sub(r"\b[Tt]he (?:camera|shot) (?:cuts|transitions|changes|switches) (back )?to\b", official, text)
        text = re.sub(r"\b[Cc]ut (back )to\b", official, text)
        marker = re.compile(r"\[Shot (\d+)\]")
        found = [m for m in marker.finditer(text) if int(m.group(1)) >= 2]
        if not found:
            return text
        parsed = []
        for m in found:
            tail = text[m.end():]
            head = re.match(r"\s*(?:At (\d\d:\d\d(?:\.\d+)?),?)?\s*", tail)
            stamp, rest = head.group(1), tail[head.end():]
            if re.match(r"cut (?:back )?to\b", rest, flags=re.IGNORECASE) or (
                    stamp and not re.match(r"(?:the )?camera\b", rest, flags=re.IGNORECASE)) or re.match(
                    r"(?:in|inside|outside|back (?:in|at)|meanwhile|later)\b", rest, flags=re.IGNORECASE):
                kind = "cut"
            elif cls._VIEWPOINT.match(rest):
                kind = "view"
            elif cls._CAMERA_MOVE.match(rest):
                kind = "merge"  # only camera motion: the same shot continues
            else:
                kind = "cut"  # a declared shot with new content stays a shot
            if _CLIP.get("single_take"):
                kind = "merge"  # Coverage Single take: no cuts at all
            if kind == "cut" and not re.match(r"cut (?:back )?to\b", rest, flags=re.IGNORECASE):
                kind = "cut_add"  # a real cut written without 'cut to': the marker gets it
            parsed.append((m, kind, stamp, head.end()))
        cls._fix_shot_times(parsed, seconds, text, store)
        out, last, number = [], 0, 1
        for m, kind, stamp, consumed in parsed:
            out.append(text[last:m.start()])
            last = m.end() + consumed
            if kind == "merge":
                if out and out[-1] and not out[-1][-1].isspace():
                    out.append(" ")
                lead = re.match(r"cut (?:back )?to\s+", text[last:], flags=re.IGNORECASE)
                if lead:  # Single take / camera-only marker: the instruction becomes a camera move, not a cut
                    out.append("the camera moves on, without a cut, to ")
                    last += lead.end()
                continue
            number += 1
            out.append(f"[Shot {number}] At {stamp}, " + ("" if kind == "cut" else "cut to "))
            if kind == "cut_add":  # 'cut to She waits' -> 'cut to she waits' for common opening words
                word = re.match(r"([A-Z][a-z]*)\b", text[last:])
                if word and word.group(1).lower() in cls._LOWER_AFTER_CUT:
                    out.append(word.group(1).lower())
                    last += len(word.group(1))
        out.append(text[last:])
        return "".join(out)

    MIN_SHOT = 2.0

    @staticmethod
    def _fix_shot_times(parsed, seconds, text="", store=None):
        """Give the kept cuts real times: at most max_shots shots, none shorter than MIN_SHOT (so the first cut comes
        after it and the last one at least that long before the end), strictly increasing. A missing or impossible
        time is placed by how much happens before it in the text (spoken words count double); when there are more
        cuts than fit, the cut closing the shortest shot becomes camera motion inside the same shot."""
        cls = H3CompactMultimodalEditDirectorFix
        clip = max(float(seconds), 1.0)
        keep = [i for i, item in enumerate(parsed) if item[1] != "merge"]
        if not keep:
            return
        shortest = cls.MIN_SHOT
        values = []
        for i in keep:
            try:
                minutes, secs = str(parsed[i][2]).split(":")
                value = int(minutes) * 60 + float(secs)
            except (TypeError, ValueError):
                value = None
            values.append(value if value is not None and 0.0 < value < clip else None)
        previous = 0.0
        for k, value in enumerate(values):  # times must grow: a time that goes back is treated as missing
            if value is not None and value <= previous:
                values[k] = None
            elif value is not None:
                previous = value

        def weight(a, b):
            segment = text[a:b] if text else ""
            spoken = 0
            if store:
                for key in re.findall(r"\x00(\d+)\x00", segment):
                    if int(key) < len(store):
                        spoken += len(store[int(key)])
            return max(1, len(re.sub(r"<d>\x00\d+\x00</d>", "", segment)) + 2 * spoken)
        positions = [parsed[i][0].start() for i in keep]
        begin = text.find("integrated_multimodal_description:") if text else -1
        finish = text.find("overall_soundscape:") if text else -1
        begin = begin if begin >= 0 else 0
        finish = finish if finish > begin else len(text or "")
        k = 0
        while k < len(values):  # fill each run of missing times between its known neighbours
            if values[k] is not None:
                k += 1
                continue
            run_end = k
            while run_end < len(values) and values[run_end] is None:
                run_end += 1
            left_t = values[k - 1] if k > 0 else 0.0
            left_p = positions[k - 1] if k > 0 else begin
            right_t = values[run_end] if run_end < len(values) else clip
            right_p = positions[run_end] if run_end < len(values) else max(finish, positions[-1] + 1)
            total = weight(left_p, right_p)
            for j in range(k, run_end):
                share = weight(left_p, positions[j]) / total if text else (j - k + 1) / (run_end - k + 1)
                values[j] = left_t + (right_t - left_t) * min(max(share, 0.0), 1.0)
            k = run_end
        alive = list(range(len(values)))  # indexes into keep / values still cutting
        budget = max(0, min(cls.max_shots(clip) - 1, int(clip // shortest) - 1))
        while len(alive) > budget:  # too many cuts: the cut that closes the shortest shot goes
            bounds = [0.0] + [values[a] for a in alive] + [clip]
            lengths = [bounds[j + 1] - bounds[j] for j in range(len(bounds) - 1)]
            shot = min(range(len(lengths)), key=lambda j: lengths[j])
            alive.pop(min(shot, len(alive) - 1))
        times = [values[a] for a in alive]
        for j in range(len(times)):  # every shot at least MIN_SHOT long, from the left...
            times[j] = max(times[j], (times[j - 1] if j else 0.0) + shortest)
        for j in range(len(times) - 1, -1, -1):  # ...and from the end of the clip
            times[j] = min(times[j], (times[j + 1] if j + 1 < len(times) else clip) - shortest)
        timed = dict(zip(alive, times))
        for k, i in enumerate(keep):
            m, kind, _, consumed = parsed[i]
            if k not in timed:
                parsed[i] = (m, "merge", None, consumed)
                continue
            ms = int(round(timed[k] * 1000))  # round the whole time first: 5.9996 s -> 00:06.000
            parsed[i] = (m, kind, f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}.{ms % 1000:03d}", consumed)

    _POSTURES = (
        ("seated", r"sits?(?: down| back| up)?|sitting(?: down| back)?|sat(?: down)?|seated|takes? a seat|"
                   r"settles? (?:onto|into|on|in)|sinks? (?:onto|into|down)"),
        ("standing", r"stands?(?: up)?|standing(?: up)?|stood(?: up)?|rises?(?: to (?:her|his|their) feet)?|"
                     r"gets? (?:up|to (?:her|his|their) feet)|getting up"),
        ("kneeling", r"kneels?(?: down)?|kneeling|knelt"),
        ("lying", r"lies(?: down| back)?|lying(?: down| back)?|lays? (?:down|back)|reclines?|reclining"),
    )
    _POSTURE_ANY = re.compile(r"\b(?:" + "|".join(p for _, p in _POSTURES) + r")\b", re.IGNORECASE)
    _POSTURE_PLACE = re.compile(
        r"\s+((?:on|onto|at|in|into|beside|next to|against|across) (?:the|a|an|her|his|their) [a-z][a-z\- ]{1,30}?)"
        r"(?=[,.;:!?]| and | while | as | with | to | facing |$)", re.IGNORECASE)
    _PLURAL_POSTURE = re.compile(r"^(?:sit|stand|kneel|lie|rise|get|settle|sink)\b", re.IGNORECASE)

    @classmethod
    def _carry_posture(cls, text):
        """A cut never resets a posture: when a subject who sat down (stood up, knelt, lay down) appears in a later
        shot that does not restate it, ', still seated on the sofa,' is added at their first mention. Works on
        dialogue-masked text; running it again changes nothing."""
        if not isinstance(text, str) or "[Shot 2]" not in text:
            return text
        head, sep, body = text.partition("integrated_multimodal_description:")
        if not sep:
            head, body = "", text
        body, sep2, tail = body.partition("overall_soundscape:")
        label = re.compile(r"<Subject \d+>(?!'s)")
        parts = re.split(r"(\[Shot \d+\])", body)
        state, number, last_label = {}, 0, None

        def sentences(chunk):
            return [m for m in re.finditer(r"[^.!?]+(?:[.!?]+|$)", chunk)]

        def owners_of(words, match):
            noted = re.search(r"(<Subject \d+>)(?:\s*\(S[\d, S]+\))?,\s*still\s+$", words[:match.start()])
            if noted:  # an earlier ', still seated' belongs to the label right before it
                return [noted.group(1)]
            before = [m.group(0) for m in label.finditer(words) if m.start() < match.start()]
            if before:
                return before if (len(before) > 1 and cls._PLURAL_POSTURE.match(match.group(0))) else before[:1]
            if re.match(r"\s*(?:They|Both|The two|The couple)\b", words):
                return None  # everyone
            return []

        def learn(chunk):
            nonlocal last_label
            seen = []
            for sentence in sentences(chunk):
                words = sentence.group(0)
                labels = [m for m in label.finditer(words)]
                found = list(cls._POSTURE_ANY.finditer(words))
                for match in found:
                    kind = next(name for name, pattern in cls._POSTURES
                                if re.fullmatch(pattern, match.group(0), flags=re.IGNORECASE))
                    owners = owners_of(words, match)
                    if owners is None:
                        owners = list(dict.fromkeys(seen + list(state)))
                    elif not owners and re.match(r"\s*(?:She|He)\b", words) and last_label:
                        owners = [last_label]
                    place = cls._POSTURE_PLACE.match(words, match.end())
                    where = re.sub(r"^(on|in)to\b", r"\1", place.group(1).strip(), flags=re.IGNORECASE) if place else ""
                    for owner in owners:
                        state[owner] = f"{kind} {where}".strip()
                for m in labels:
                    seen.append(m.group(0))
                    last_label = m.group(0)

        out = []
        for index, part in enumerate(parts):
            marker = re.fullmatch(r"\[Shot (\d+)\]", part)
            if marker:
                number = int(marker.group(1))
                out.append(part)
                continue
            if number >= 2 and state:
                stated = set()  # whoever this shot gives a posture to (its owner, as learn() reads it)
                for sentence in sentences(part):
                    words = sentence.group(0)
                    labels = [m.group(0) for m in label.finditer(words)]
                    for match in cls._POSTURE_ANY.finditer(words):
                        owners = owners_of(words, match)
                        if owners is None:
                            stated.update(state)
                        elif owners:
                            stated.update(owners)
                        elif re.match(r"\s*(?:She|He)\b", words) and labels:
                            stated.update(labels[:1])
                inserts, done = [], set()
                for m in label.finditer(part):
                    name = m.group(0)
                    if name in done:
                        continue
                    done.add(name)
                    if name in stated or name not in state:
                        continue
                    end = m.end()
                    speaker = re.match(r"\s*\(S[\d, S]+\)", part[end:])
                    if speaker:
                        end += speaker.end()
                    follows = part[end:end + 1]
                    note = f", still {state[name]}"
                    inserts.append((end, note + ("" if follows in (",", ":", ".", ";") else ",")))
                for end, note in reversed(inserts):
                    part = part[:end] + note + part[end:]
            learn(part)
            out.append(part)
        body = "".join(out)
        return head + sep + body + sep2 + tail

    @staticmethod
    def _json_candidates(body):
        """Yield repaired versions of a possibly truncated JSON document, most faithful first."""
        end = body.rfind("}")
        if end > 0:
            yield body[:end + 1]
        stack, in_str, esc, commas = [], False, False, []
        for i, ch in enumerate(body):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack:
                    stack.pop()
            elif ch == ",":
                commas.append((i, "".join(reversed(stack))))
        for pos, closers in reversed(commas[-6:]):
            yield body[:pos] + closers
        if not in_str:  # closing a string cut mid-text would pass truncated prose off as complete
            tail = re.sub(r"[,:\s]+$", "", body)
            yield tail + "".join(reversed(stack))

    _SCENE_KEYS = ("scenes", "prompts", "scene_prompt", "scenePrompts", "scene prompts", "scene_list", "clips",
                   "scene_prompts_list", "video_prompts", "shots")

    @classmethod
    def _scene_key_fix(cls, data):
        """A reply that is valid JSON but names its scene list differently (some models, e.g. via Venice) is
        mapped onto scene_prompts instead of failing; nothing changes when scene_prompts is present."""
        if not isinstance(data, dict) or "scene_prompts" in data:
            return data
        if len(data) == 1:  # {"result": {...scene_prompts...}}
            inner = next(iter(data.values()))
            if isinstance(inner, dict) and "scene_prompts" in inner:
                return inner
        for key in cls._SCENE_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                value = [value]
            if isinstance(value, list) and value:
                fixed = dict(data)
                fixed["scene_prompts"] = [
                    next((str(item[k]) for k in ("prompt", "scene_prompt", "text", "content") if isinstance(item, dict)
                          and isinstance(item.get(k), str) and len(item) == 1), item) for item in value]
                return fixed
        numbered = sorted((int(m.group(1)), key) for key in data
                          for m in [re.fullmatch(r"(?i)scene[ _-]?(\d+)", str(key))] if m)
        if numbered and all(isinstance(data[key], (str, dict)) for _, key in numbered):
            fixed = {k: v for k, v in data.items() if k not in {key for _, key in numbered}}
            fixed["scene_prompts"] = [data[key] for _, key in numbered]
            return fixed
        return data

    @classmethod
    def _root_object(cls, data):
        """The director reply must be one JSON object: unwrap [ {...} ], send anything else to repair."""
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            return cls._scene_key_fix(data[0])
        if not isinstance(data, dict):
            raise json.JSONDecodeError("director reply is not a JSON object", "", 0)
        return cls._scene_key_fix(data)

    @classmethod
    def _lenient_json(cls, text, first_error=None):
        """Parse a director reply that may be truncated, wrapped in a list or otherwise slightly off."""
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        body = text[start:] if start >= 0 else text
        error = first_error
        fixed = re.sub(r",(\s*[}\]])", r"\1", body)
        fixed = re.sub(r'(["\]}0-9el])([ \t]*\n\s*)("[^"\n]{1,80}"\s*:)', r"\1,\2\3", fixed)
        fixed = re.sub(r'(\})([ \t]*\n\s*)(\{)', r"\1,\2\3", fixed)
        sources = [body] if fixed == body else [body, fixed]
        attempts = (item for source in sources for item in cls._json_candidates(source))
        for attempt in attempts:
            try:
                data = json.loads(attempt, strict=False)
            except json.JSONDecodeError as exc:
                error = error or exc
                continue
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                data = data[0]
            if not isinstance(data, dict):
                continue
            data = cls._scene_key_fix(data)
            if cls._looks_truncated(data):
                continue  # a scene cut off mid-way is not a recovered reply: try a gentler repair
            return data
        raise error or json.JSONDecodeError("unparseable director response", text, 0)

    @classmethod
    def _looks_truncated(cls, data):
        """True when a REPAIRED reply is not a complete answer: no scenes, a scene missing fields the first
        scene has, no footage, or an empty field. (Text cut mid-string is never produced: the repair drops it instead.)"""
        scenes = data.get("scene_prompts")
        items = scenes if isinstance(scenes, list) else [data.get("edit_prompt")]
        items = [item for item in items if item is not None]
        if not items:
            return True
        creative = "target_design" in data
        if creative and not isinstance(data.get("target_design"), dict):
            return True
        first_keys = set(items[0]) if isinstance(items[0], dict) else set()
        for item in items:
            if isinstance(item, str):
                if not item.strip():
                    return True
                continue
            if not isinstance(item, dict):
                return True
            # every scene carries what the first one carries (a cut-off tail loses its last fields), and creative
            # scenes always need their footage; planning-only fields a model skips everywhere are not truncation
            required = first_keys | ({"integrated_multimodal_description"} if creative else set())
            if required - set(item):
                return True
            if any(isinstance(item.get(key), str) and not item[key].strip() for key in required):
                return True
        return False

    @staticmethod
    def _scrub_music(text):
        """Instrumentation-only music: turn 'that build(s) subtle tension through' into 'moving through'."""
        marker = "non_diegetic_music:"
        if marker not in text:
            return text
        head, music = text.rsplit(marker, 1)
        music = re.sub(
            r"\s*(?:,\s*)?(?:(?:that|which)\s+)?(?:builds?|building)\s+(?:a\s+|some\s+)?(?:subtle\s+|rising\s+|quiet\s+|mounting\s+)?"
            r"tension(?:\s+(?:through|with|via|by))?\s+", " moving through ", music, flags=re.IGNORECASE)
        return head + marker + music

    @staticmethod
    def _scope_edit_pictures(text):
        """Video task Edit: Picture 2 onward lend only the edited aspects, never the person who wears them."""
        refs = _CLIP.get("edit_refs") or []
        section = re.search(r"(?ms)^subject_definitions:\s*\n(.*?)(?=^\w+:\s*$|\Z)", text) if refs else None
        if not section:
            return text
        if _CLIP.get("task") == "add":
            body = section.group(1)
            lines = [f"<Picture {n}> is a new person added to the scene, not a replacement: everyone from the source "
                     f"video stays exactly who they are, with their own look and action." for n in refs
                     if f"<Picture {n}> is a new person added" not in body and re.search(rf"<Picture\s+{n}>", text)]
            body = body.rstrip() + ("\n" + "\n".join(lines) if lines else "") + "\n\n"
            return text[:section.start(1)] + body + text[section.end(1):]
        body = re.sub(r"(?i)(\bidentity\b[^.\n]{0,40}?)<Picture\s+([2-4])>", r"\1<Picture 1>", section.group(1))
        aspects = _CLIP.get("edit_aspects") or ["what the request takes from it (an outfit, a garment, a hairstyle, an "
                                               "object or a style)"]
        kept = ["face", "skin tone"] + [x for x, key in (("hair", "hair"), ("body", "body")) if key not in
                                        (_CLIP.get("edit_aspect_keys") or [])] + ["identity"]
        lines = [f"<Picture {n}> is referenced only for {', '.join(aspects)}; the person shown in it is not referenced: "
                 f"the {', '.join(kept)} stay those of the people of the source video." for n in refs
                 if f"<Picture {n}> is referenced only for" not in body and re.search(rf"<Picture\s+{n}>", text)]
        body = body.rstrip() + ("\n" + "\n".join(lines) if lines else "") + "\n\n"
        return text[:section.start(1)] + body + text[section.end(1):]

    _SHALLOW = re.compile(r"\b(?:an? )?(?:very |extremely )?shallow depth[- ]of[- ]field\b|\b(?:a )?shallow focus\b|"
                          r"\b(?:with |soft |creamy )?bokeh(?: background)?\b", re.IGNORECASE)

    @staticmethod
    def _same_clip(a, b):
        """Two sockets fed with the same frames (same shape, same first/middle/last frame)."""
        if a is None or b is None:
            return a is b and a is not None
        try:
            if tuple(a.shape) != tuple(b.shape):
                return False
            n = int(a.shape[0])
            import numpy as np
            frame = lambda v, i: np.asarray(v[i].detach().cpu().numpy(), dtype=np.float32)
            for i in sorted({0, n // 2, n - 1}):
                if float(np.abs(frame(a, i) - frame(b, i)).max()) > 1e-4:
                    return False
            return True
        except Exception:
            return False

    _VIDEO_PHRASE = re.compile(r"\s*,?\s*\b(?:of|from|in|as in|seen in|like in) <Video \d+>|<Video \d+>(?:'s)?\s*")

    @classmethod
    def _no_missing_videos(cls, text):
        """A <Video N> the conditioning does not have is never cited (dialogue untouched)."""
        have = set(_CLIP.get("videos") or [])

        def drop(m):
            if {int(n) for n in re.findall(r"<Video (\d+)>", m.group(0))} <= have:
                return m.group(0)
            if not m.group(0).lstrip(" ,").startswith("<"):  # "of/from/in <Video N>": the phrase goes
                return ""
            return "the scene" + (" " if m.group(0).endswith(" ") else "")
        masked, keep = cls._mask_d(text)
        out = cls._VIDEO_PHRASE.sub(drop, masked)
        out = re.sub(r"[ \t]+([,.;:])", r"\1", re.sub(r"[ \t]{2,}", " ", out))
        out = re.sub(r"(?m)(^|[.!?]\s+)the scene\b", lambda m: m.group(1) + "The scene", out)
        return cls._unmask_d(out, keep)

    @classmethod
    def _hold_capture(cls, key, record):
        """Hold record of this attempt: saved now, and replaced at the end by what was really delivered."""
        from .story_director import _set_held_director_plan
        identity = H3CompactMultimodalEditDirectorFix._real_provider(record.get("provider_identity"))
        if isinstance(identity, dict):  # never save the retry adapter as the provider
            record = dict(record, provider_identity=json.dumps(identity, sort_keys=True))
        record = dict(record, fix_mode={"compact": bool(_CLIP.get("compact")), "keep_audio": bool(_CLIP.get("keep_audio")),
                                        "basic": bool(_CLIP.get("basic")), "video_edit": bool(_CLIP.get("video_edit")),
                                        "edit_task": _CLIP.get("edit_task") or "",
                                        "ports": list(_CLIP.get("ports") or [])},
                      random_picks=list(_CLIP.get("random_picks") or []))
        _CLIP["held_last"] = (key, dict(record))
        _set_held_director_plan(key, record)

    @staticmethod
    def _real_provider(identity):
        """The retry adapter for strict-JSON failures is not a provider: Hold compares the real one."""
        try:
            data = json.loads(str(identity or ""))
        except ValueError:
            return str(identity or "")
        if isinstance(data, dict) and data.get("type") == "_RelaxedJSONLLM":
            real = _CLIP.get("real_llm")
            data["type"] = type(real).__name__ if real is not None and not isinstance(real, _RelaxedJSONLLM) else "*"
        return data

    @classmethod
    def _provider_changed(cls, saved, current):
        a, b = cls._real_provider(saved), cls._real_provider(current)
        if isinstance(a, dict) and isinstance(b, dict):
            if "*" in (a.get("type"), b.get("type")):
                a, b = dict(a, type="*"), dict(b, type="*")
            return a != b
        return a != b

    @staticmethod
    def _hold_seconds_changed(held, seconds):
        mode = held.get("fix_mode")
        if isinstance(mode, dict) and (bool(mode.get("compact")) != bool(_CLIP.get("compact"))
                                       or bool(mode.get("keep_audio")) != bool(_CLIP.get("keep_audio"))
                                       or bool(mode.get("basic")) != bool(_CLIP.get("basic"))
                                       or bool(mode.get("video_edit")) != bool(_CLIP.get("video_edit"))
                                       or (mode.get("edit_task") or "") != (_CLIP.get("edit_task") or "")):
            raise RuntimeError("The director mode changed since Hold Edit Prompt was saved (Normal / Compact / Edit "
                               "or Transfer / Transfer Keep Audio / Basic, or the Video Task). Disable Hold once to "
                               "regenerate; no LLM was called.")
        if isinstance(mode, dict) and "ports" in mode and list(mode.get("ports") or []) != list(_CLIP.get("ports") or []):
            raise RuntimeError("The connected video / image inputs changed since Hold Edit Prompt was saved ("
                               + (", ".join(mode.get("ports") or []) or "none") + " -> "
                               + (", ".join(_CLIP.get("ports") or []) or "none") + "), so its Picture / Video tags may "
                               "not match. Disable Hold once to regenerate; no LLM was called.")
        _CLIP["random_picks"] = list(held.get("random_picks") or [])  # Hold reports the picks of the saved plan
        saved = held.get("seconds_per_scene")
        try:
            changed = saved is not None and abs(float(saved) - float(seconds)) > 0.01
        except (TypeError, ValueError):
            changed = False
        if changed:
            raise RuntimeError(f"Seconds per Scene changed since Hold Edit Prompt was saved ({float(saved):g} s -> "
                               f"{float(seconds):g} s). Disable Hold once to regenerate; no LLM was called.")
        return False

    @staticmethod
    def _preview_of(prompt):
        try:
            data = json.loads(prompt) if isinstance(prompt, str) and prompt.lstrip().startswith("{") else None
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("scene_prompts"), list):
            scenes = [str(item) for item in data["scene_prompts"]]
            return "\n\n".join(f"SCENE {i}\n{item}" for i, item in enumerate(scenes, 1)), len(scenes)
        return str(prompt), 1

    _BODY_VOLUME = re.compile(
        r"\b(?:thick|thicc|full|wide|rounded|round|curvy|voluptuous|plump|ample|big|large|huge|heavy|chubby|fleshy|"
        r"generous|massive|juicy|meaty|bouncy|jiggling)\s+(?=(?:and\s+[a-z]+\s+)?(?:thighs?|bust|breasts?|boobs|hips|"
        r"buttocks|butt|ass|booty|curves|cleavage)\b(?!\s+(?:shot|sensor|frame|angle|lens|view|framing)))",
        re.IGNORECASE)
    _BODY_REQUEST = re.compile(
        r"gord|rellen|delgad|flac|curv|voluptu|tetona|culona|caderona|pechos? grande|senos? grande|nalgas? grande|"
        r"\bthick|\bfat\b|chubby|\bbbw\b|\bslim|skinny|petite|musc|plus[- ]size|big (?:ass|boobs|breasts|butt)|"
        r"cuerpo (?:m[aá]s )?(?:grande|gordo|delgado|flaco|voluptuoso|fitness|marcado)", re.IGNORECASE)

    _FUSION = re.compile(
        r"\b(?:their )?bodies (?:melting|melt|merging|merge|fusing|fuse|fused|blending|blend)(?: (?:together|into one|"
        r"into each other))?|\bmelting (?:together|into (?:each other|one another|one))|\bmerging into one|"
        r"\bbecoming one\b|\bas one body\b|\b(?:limbs|bodies) (?:tangled|intertwined|entwined)|\bintertwined bodies|"
        r"\bentangled bodies|\btangle of (?:limbs|bodies)", re.IGNORECASE)

    _GUIDE_WORDS = re.compile(
        r",?\s*\b(?:exactly |just )?(?:as|like) (?:(?:shown|seen|worn|depicted|pictured)(?: in| on| by)?|in) (?:the )?(?:guide|driven|driver|source)"
        r"(?: (?:image|video|frame|clip|performer|person|reference|blueprint))?\b|"
        r",?\s*\b(?:taken |copied )?from the (?:guide|driven|driver)(?: (?:image|video|frame|clip|performer|person|blueprint))?\b|"
        r",?\s*\b(?:matching|mirroring|following) the (?:guide|driven|driver)(?: (?:image|video|frame|clip|performer|person))?\b",
        re.IGNORECASE)
    _GUIDE_NOUNS = re.compile(r"\bthe (?:guide|driven|driver)(?:'s| (?:image|video|frame|clip|blueprint)'s)\s+|"
                              r"\bthe (?:guide|driven|driver) (?:performer|person|woman|man)\b", re.IGNORECASE)

    @classmethod
    def _no_guide_words(cls, text):
        """The video model never sees the guide: no 'as in the guide' in the prompt."""
        text = cls._GUIDE_WORDS.sub("", text)
        text = cls._GUIDE_NOUNS.sub(lambda m: "the " if "'s" in m.group(0) else "the person", text)
        return re.sub(r"[ \t]+([,.;:])", r"\1", text)

    @classmethod
    def _no_fusion(cls, text):
        """'bodies melting together' makes the video model fuse two people: write it as close contact."""
        return cls._FUSION.sub("bodies pressed close, each body whole and separate", text)

    @classmethod
    def _neutral_body(cls, text):
        """A referenced body is shown by its picture: volume adjectives only distort it (H3 adds them up)."""
        return cls._outside_dialogue(text, lambda t: cls._BODY_VOLUME.sub("", t))

    @staticmethod
    def _mask_d(text):
        """Hide the spoken words (<d>...</d> bodies) from every visual clean-up; _unmask_d puts them back."""
        store = []

        def keep(m):
            store.append(m.group(1))
            return f"<d>\x00{len(store) - 1}\x00</d>"
        return re.sub(r"<d>(.*?)</d>", keep, text, flags=re.S), store

    @staticmethod
    def _unmask_d(text, store):
        return re.sub(r"<d>\x00(\d+)\x00</d>",
                      lambda m: f"<d>{store[int(m.group(1))]}</d>" if int(m.group(1)) < len(store) else m.group(0), text)

    @staticmethod
    def _outside_dialogue(text, fn):
        """Apply a visual clean-up to the text around the <d>...</d> lines, never to the words spoken."""
        parts = re.split(r"(<d>.*?</d>)", text, flags=re.S)
        return "".join(part if part.startswith("<d>") else fn(part) for part in parts)

    @classmethod
    def _deep_focus(cls, text):
        """Sharp from front to back unless the request or a Visual look / Motion style asked otherwise: the
        shallow-focus wording is removed cleanly and 'deep focus' is said once."""
        state = {"said": "deep focus" in text.lower()}

        def clean(t):
            t = re.sub(r"\b(?:soft|creamy|blurry)?\s?bokeh\s+(?=[a-z])", "", t, flags=re.IGNORECASE)  # 'bokeh city lights'
            t = re.sub(r"\s+in (?:soft|shallow) focus\b", "", t, flags=re.IGNORECASE)
            t = re.sub(r"\s+(?:softly )?(?:blurred|out of focus) (?:in the )?background\b", " background", t,
                       flags=re.IGNORECASE)

            def swap(m):
                connector = re.match(r",?\s*(?:with |and )?", m.group(0)).group(0)
                if state["said"] and connector.strip():  # an add-on phrase: drop it
                    return ""
                body = "deep focus" if state["said"] else "deep focus, sharp from foreground to background"
                state["said"] = True
                before = m.string[:m.start()].rstrip()
                if not connector.strip() and (not before or before[-1] in ".:;\n"):
                    body = body[0].upper() + body[1:]
                return connector + body
            t = re.sub(r",?\s*(?:with |and )?(?:an? )?(?:very |extremely )?(?:shallow depth(?:[- ]of[- ]field)?|shallow focus|"
                       r"(?:soft |creamy )?bokeh(?: background)?)", swap, t, flags=re.IGNORECASE)
            return re.sub(r"\s+([,.;])", r"\1", re.sub(r"(\w) {2,}", r"\1 ", t))
        return cls._outside_dialogue(text, clean)

    _MOUTH_OK = re.compile(r"\b(?:lips?|mouth)\b[^.]{0,40}\b(?:closed|shut|pressed|sealed|still|together|silent)\b|"
                           r"\bclosed[- ]lip|\bsilent(?:ly)? nod|\bsips?\b",
                           re.IGNORECASE)
    _SPEAKER = re.compile(r"(<Subject \d+>)\s*\(S\d+\)[^<]{0,160}?<d>", re.IGNORECASE)

    @classmethod
    def _silent_listeners(cls, text):
        """Every visible non-speaker in a shot with a spoken line gets a closed-mouth action, written by the code so
        it never depends on the language model remembering the rule."""
        parts = re.split(r"(\[Shot \d+\])", text)
        for i, part in enumerate(parts):
            if "<d>" not in part or "voiceover" in part.lower() or re.search(r"\(S\d+\s*,", part):
                continue
            lines = list(cls._SPEAKER.finditer(part))
            if not lines:
                continue
            visible = [t for t in dict.fromkeys(re.findall(r"<Subject \d+>", part))
                       if not re.search(re.escape(t) + r"[^.]{0,100}\b(?:leaves|exits|walks out|steps out|goes out|"
                                        r"is gone|off-?screen|out of (?:frame|the room|shot|view)|disappears|"
                                        r"heard but not seen)", part, flags=re.IGNORECASE)]
            inserts = []
            for n, m in enumerate(lines):  # every line: each other visible person is silent for it
                end = part.find("</d>", m.end())
                if end < 0:
                    continue
                end += 4
                nxt = lines[n + 1].start() if n + 1 < len(lines) else len(part)
                around = part[max(0, m.start() - 200):nxt]
                quiet = []
                for t in visible:
                    if t == m.group(1):
                        continue
                    if any(cls._MOUTH_OK.search(part[x.end():x.end() + 90])
                           for x in re.finditer(re.escape(t), around)):
                        continue  # already given a closed mouth around this line
                    quiet.append(t)
                if quiet:  # right after the line, so the silence belongs to it
                    inserts.append((end, " " + " ".join(f"{t} listens silently, lips pressed together."
                                                        for t in quiet) + ("" if end >= len(part) or part[end].isspace() else " ")))
            for at, add in reversed(inserts):
                part = part[:at] + add + part[at:]
            parts[i] = part
        return "".join(parts)

    _UNDRESS = re.compile(r"\b(?:takes?|pulls?|strips?|yanks?|slips?|peels?|tugs?|drags?|lifts?|removes?|"
                          r"unhooks?|unzips?|unbuttons?)\b[^.]{0,30}\b(?:her|his|their)\s+(?:\w+\s+){0,2}(?:top|shirt|"
                          r"t-shirt|blouse|bra|dress|skirt|pants|trousers|jeans|shorts|panties|underwear|briefs|boxers|"
                          r"thong|bikini|robe|towel|tank top|crop top|leggings|lingerie|swimsuit)\b[^.]{0,30}\b(?:off|"
                          r"down|over (?:her|his|their) head|aside|away|open)\b|\b(?:undress|naked|nude|topless|bare (?:breasts?|chest|buttocks)|"
                          r"exposing (?:her |his |their |both )?(?:breasts?|nipples?|chest|buttocks|genitals|vulva|penis))",
                          re.IGNORECASE)

    @classmethod
    def _outfit_at_start(cls, text):
        """When clothes come off in the clip, Target appearance is only the starting state: H3 must not re-dress
        anyone after a cut."""
        if not cls._UNDRESS.search(text.split("integrated_multimodal_description:")[-1]
                                   .split("detailed_description:")[-1]):
            return text
        note = "as at the first frame; whatever comes off on screen stays off in every later shot"
        return re.sub(r"Target appearance(?: \(([^)]*)\))?:",
                      lambda m: f"Target appearance ({(m.group(1) + '; ') if m.group(1) else ''}{note}):"
                      if note not in m.group(0) else m.group(0), text)

    _STRAY_MOUTH = re.compile(r",?\s*(?:with )?(?:his|her|their) (?:mouth|lips) (?:moving|moves?|open(?:s|ing)? and "
                              r"clos(?:es|ing))(?: (?:as|while) (?:he|she|they) speaks?| with the words)?(?=[,.;])",
                              re.IGNORECASE)

    @staticmethod
    def _speaker_tags(text):
        """'(S1, warm low voice) says:' -> '(S1) says in a warm low voice:' ((S1,S2) stays a shared line)."""
        def fix(m):
            desc = m.group(2).strip().rstrip(",")
            article = "" if re.match(r"(?:a|an|the|his|her|their)\b", desc, re.IGNORECASE) else (
                "an " if desc[:1].lower() in "aeiou" else "a ")
            return f"({m.group(1)}) {m.group(3)} in {article}{desc}"
        return re.sub(r"\((S\d+),\s*(?!S\d)([^)]{3,160})\)\s*(says|whispers|asks|replies|sings|shouts|murmurs|moans)",
                      fix, text)

    @classmethod
    def _split_speakers(cls, text, seconds):
        """Every change of speaker inside one shot gets its own cut (a close-up on the new speaker) while the shot
        budget lasts, otherwise a reframe; one pass resolves them all, so running it again changes nothing."""
        if (_CLIP.get("single_take") or "elaborate" not in str(_CLIP.get("mode") or "") or "<d>" not in text
                or _CLIP.get("i2v")):
            return text
        budget = cls.max_shots(seconds) - max(len(re.findall(r"\[Shot \d+\]", text)), 1)
        parts = re.split(r"(\[Shot \d+\])", text)
        for i, part in enumerate(parts):
            found = list(cls._SPEAKER.finditer(part))
            if len({m.group(1) for m in found}) < 2:
                continue
            masked = re.sub(r"<d>.*?</d>", lambda m: "x" * len(m.group(0)), part, flags=re.S)
            inserts, prev = [], found[0].group(1)
            for m in found[1:]:
                who = m.group(1)
                if who == prev:
                    continue
                start = max(masked.rfind(".", 0, m.start()) + 1, masked.rfind("!", 0, m.start()) + 1,
                            masked.rfind("?", 0, m.start()) + 1, part.rfind("</d>", 0, m.start()) + 4)
                already = re.search(r"(?:reframes to favour|cut to a close-up on) " + re.escape(who),
                                    part[max(0, start - 240):m.start()])
                if start > 3 and not already:
                    if budget > 0:
                        lead = (f" [Shot 99] At 00:00.000, cut to a close-up on {who}, {prev} seen from behind or in "
                                "profile:")
                        budget -= 1
                    else:
                        lead = (f" The camera reframes to favour {who}'s face, {prev} now in profile or from behind "
                                "with lips closed.")
                    inserts.append((start, lead))
                prev = who
            for start, lead in reversed(inserts):
                part = part[:start] + lead + part[start:]
            parts[i] = part
        return "".join(parts)

    @classmethod
    def _finalize_prompts(cls, raw, seconds):
        """Idempotent clean-up applied to the node output (JSON scene list or a single prompt)."""
        base_fix = H3CompactMultimodalEditDirectorFix._timecode_shots
        scrub = H3CompactMultimodalEditDirectorFix._scrub_music

        drop_cap = H3CompactMultimodalEditDirectorFix._drop_identity_cap

        normalize = H3CompactMultimodalEditDirectorFix._normalize_h3

        def fix(item, secs):
            cls_ = H3CompactMultimodalEditDirectorFix
            store = []
            if isinstance(item, str):
                item, store = cls_._mask_d(item)  # the spoken words are never touched by a visual step
                try:
                    item = cls_._split_speakers(cls_._speaker_tags(item), secs)
                except Exception:
                    pass
                item = base_fix(item, secs, store)
                try:
                    item = cls_._carry_posture(item)
                except Exception:  # noqa: BLE001 - a clean-up never breaks the prompt
                    pass
                item = cls_._unmask_d(item, store)
                out = normalize(drop_cap(scrub(item)))
                out, store = cls_._mask_d(out)
            else:
                out = normalize(drop_cap(scrub(base_fix(item, secs))))
            if isinstance(out, str):
                out = H3CompactMultimodalEditDirectorFix._STRAY_MOUTH.sub("", out)
                out = H3CompactMultimodalEditDirectorFix._outfit_at_start(out)
            if _CLIP.get("task") in ("edit", "add") and _CLIP.get("mode") == "transfer":
                out = H3CompactMultimodalEditDirectorFix._scope_edit_pictures(out)
            official = _CLIP.get("mode") in ("normal", "compact") and not _CLIP.get("video_edit")
            if _CLIP.get("i2v") and _CLIP.get("mode") != "transfer" and not official:  # official first-frame layout
                out = H3CompactMultimodalEditDirectorFix._to_i2va(out)
            if isinstance(out, str) and _CLIP.get("body_neutral"):  # the picture carries the body
                out = H3CompactMultimodalEditDirectorFix._neutral_body(out)
            if isinstance(out, str):
                out = H3CompactMultimodalEditDirectorFix._no_fusion(out)
            if isinstance(out, str) and _CLIP.get("guide_creative"):
                out = H3CompactMultimodalEditDirectorFix._no_guide_words(out)
            if isinstance(out, str) and not _CLIP.get("keep_shallow"):  # sharp focus unless asked
                out = H3CompactMultimodalEditDirectorFix._deep_focus(out)
            if isinstance(out, str) and not _CLIP.get("keep_audio"):
                out = H3CompactMultimodalEditDirectorFix._silent_listeners(out)
            if isinstance(out, str) and official:  # the layout MiniMax documents for each input mode
                try:
                    out = H3CompactMultimodalEditDirectorFix._official_layout(out)
                except Exception:  # noqa: BLE001 - a layout step never breaks the prompt
                    pass
            if isinstance(out, str) and store:
                out = H3CompactMultimodalEditDirectorFix._unmask_d(out, store)
            if _CLIP.get("keep_audio") and isinstance(out, str):  # the source audio is the final audio
                out = re.sub(r"\s*(?:\(S\d+\)\s*)?(?:\b(?:says|whispers|asks|replies|murmurs|shouts|sings|moans)\s*:?\s*)?"
                             r"<d>.*?</d>", "", out, flags=re.IGNORECASE | re.DOTALL)
                out = re.sub(r"(?ms)^overall_soundscape:.*?(?=^non_diegetic_music:|\Z)", "overall_soundscape:\nN/A\n\n", out)
                out = re.sub(r"(?ms)^non_diegetic_music:.*\Z", "non_diegetic_music:\nN/A", out)
            if _CLIP.get("video_edit") and isinstance(out, str):
                out = H3CompactMultimodalEditDirectorFix._video_edit_format(out)
            if isinstance(out, str) and "<Video" in out:
                out = H3CompactMultimodalEditDirectorFix._no_missing_videos(out)
            return out
        if isinstance(raw, str) and raw.lstrip().startswith("{"):
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("scene_prompts"), list):
                data["scene_prompts"] = [fix(item, seconds) if isinstance(item, str) else item
                                         for item in data["scene_prompts"]]
                return json.dumps(data, ensure_ascii=False)
            return raw
        return fix(raw, seconds) if isinstance(raw, str) else raw

    _PLACEHOLDER_SPEECH = re.compile(
        r"\s*(?:<(?:Subject|Picture) \d+>\s*)*(?:\(S\d+\)\s*)?(?:says:?\s*)?<d>\[[^\]]+\]\s*[^<]{0,80}?"
        r"(?:letra de (?:la|esta|su) canci[oó]n|la letra que|lyrics (?:of|to|from) (?:the|this|her|his) song|song lyrics|"
        r"(?:the )?song (?:she|he|they) (?:is |are )?(?:sings?|singing)|canci[oó]n que (?:est[aá] )?cant|"
        r"texto de (?:la|su) (?:canci[oó]n|frase|di[aá]logo)|placeholder|dialogue here|di[aá]logo aqu[ií]|"
        r"words? (?:she|he|they) (?:sings|says))[^<]*</d>", re.IGNORECASE)

    _SOUND_START = re.compile(r"(?i)^(?:the |an? |its |her |his )?(?:overall |non[- ]diegetic |background |ambient )?"
                              r"(?:soundscape|sound design|audio|ambien(?:ce|t sound)|sounds?|music|score|soundtrack)\b|"
                              r"^(?:sound|music|audio)\s*:")
    _MUSIC_WORD = re.compile(r"(?i)\b(?:music|score|soundtrack|beat|melody)\b")

    @staticmethod
    def _relocate_sound(text):
        """Move sound/music sentences written inside integrated_multimodal_description into their sections
        when those sections are missing or N/A (with Keep Audio they are simply dropped)."""
        match = re.search(r"(?ms)^integrated_multimodal_description:\s*\n(.*?)(?=^(?:overall_soundscape|non_diegetic_music):|\Z)", text)
        if not match:
            return text
        body = match.group(1)
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        sound, music, kept = [], [], []
        def empty(name):
            section = re.search(rf"(?ms)^{name}:\s*(.*?)(?=^\w+:\s*$|\Z)", text)
            return section is None or section.group(1).strip().upper() in ("", "N/A")
        room = {"overall_soundscape": _CLIP.get("keep_audio") or empty("overall_soundscape"),
                "non_diegetic_music": _CLIP.get("keep_audio") or empty("non_diegetic_music")}
        for sentence in sentences:
            if H3CompactMultimodalEditDirectorFix._SOUND_START.match(sentence.strip()):
                is_music = bool(H3CompactMultimodalEditDirectorFix._MUSIC_WORD.search(sentence))
                if room["non_diegetic_music" if is_music else "overall_soundscape"]:
                    (music if is_music else sound).append(sentence.strip())
                    continue
            kept.append(sentence)  # nowhere to move it: the information stays where it is
        if not sound and not music:
            return text
        text = text[:match.start(1)] + " ".join(kept).strip() + "\n\n" + text[match.end(1):].lstrip("\n")
        if _CLIP.get("keep_audio"):
            return text
        for name, lines in (("overall_soundscape", sound), ("non_diegetic_music", music)):
            if not lines or not empty(name):
                continue
            value = re.sub(r"(?i)^(?:the |its )?(?:overall |non[- ]diegetic |background )?(?:soundscape|sound design|audio|"
                           r"music|score|soundtrack)\s+(?:features|is|includes|consists of|has)\s+", "", " ".join(lines))
            value = value[:1].upper() + value[1:]
            if re.search(rf"(?m)^{name}:", text):
                text = re.sub(rf"(?ms)^{name}:\s*(?:N/A)?\s*(?=^\w+:\s*$|\Z)", lambda m, v=value: f"{name}:\n{v}\n\n", text, count=1)
            else:
                text = text.rstrip() + f"\n\n{name}:\n{value}"
        if "non_diegetic_music:" in text and "overall_soundscape:" not in text:
            text = text.replace("non_diegetic_music:", "overall_soundscape:\nN/A\n\nnon_diegetic_music:", 1)
        return text.rstrip()

    @classmethod
    def _to_i2va(cls, text):
        """Official MiniMax H3 image-to-video layout for the prompt that starts at <Picture 1>: the alignment line, a
        blank line, then integrated_multimodal_description / overall_soundscape / non_diegetic_music only. The
        reference-mode header ([reference generation], where the image is NOT a frame) and subject_definitions are
        removed; the subjects are introduced in the first sentence, the <Picture 1> subject exactly as in the frame."""
        if not isinstance(text, str) or cls.I2V_ALIGNMENT not in text:
            return text
        body = text.replace(cls.I2V_ALIGNMENT, "", 1).strip()
        body = re.sub(r"For the target video, at 0\.00 seconds into the target video, <Picture 1> \(from[^)]*\) is fully "
                      r"referenced\.?[ \t]*", "", body)
        body = re.sub(r"^\[[^\]\n]+\]\s*\n", "", body).strip()
        intro = ""
        match = re.search(r"(?ms)^subject_definitions:[ \t]*\n?(.*?)(?=^(?:integrated_multimodal_description|"
                          r"overall_soundscape|non_diegetic_music):|\Z)", body)
        if match:
            sentences = []
            for line in [x.strip() for x in match.group(1).split("\n") if x.strip() and x.strip().upper() != "N/A"]:
                if "<Picture 1>" in line:
                    line = re.split(r"\s+(?:Target appearance|Wardrobe|Body proportions)\b", line)[0].rstrip(" .;,:")
                    line += (" (exactly as seen in <Picture 1> at 0.00 seconds: the same clothing, hair, place and "
                             "light)")
                sentences.append(line.rstrip(".") + ".")
            intro = " ".join(sentences)
            body = (body[:match.start()] + body[match.end():]).strip()
        if "integrated_multimodal_description:" in body:
            body = re.sub(r"(?m)^integrated_multimodal_description:[ \t]*\n?",
                          lambda m: "integrated_multimodal_description:\n" + (intro + " " if intro else ""), body, count=1)
        else:
            body = "integrated_multimodal_description:\n" + (intro + " " if intro else "") + body
        for section in ("overall_soundscape", "non_diegetic_music"):
            if f"{section}:" not in body:
                body = body.rstrip() + f"\n\n{section}:\nN/A"
        return cls.I2V_ALIGNMENT + "\n\n" + body.strip()

    _LAYOUT_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "integrated_multimodal_description",
                        "detailed_description", "overall_soundscape", "non_diegetic_music")

    @classmethod
    def _layout_sections(cls, text):
        heads = list(re.finditer(r"(?m)^(" + "|".join(cls._LAYOUT_SECTIONS) + r"):[ \t]*\n?", text))
        found = {}
        for index, head in enumerate(heads):
            end = heads[index + 1].start() if index + 1 < len(heads) else len(text)
            found[head.group(1)] = text[head.end():end].strip()
        return found

    @staticmethod
    def _short_name(description):
        """'a woman in her early 30s, with dark hair' -> 'the woman in her early 30s'; a name stays a name."""
        first = re.split(r",|;|\s+(?:with|wearing|who|whose|seen|in|on|at|of|from|by|near)\s+",
                         re.sub(r"\s+", " ", description).strip())[0].strip(" .")
        words = first.split()[:7]
        short = " ".join(words) or "the person"
        if re.match(r"(?i)(?:a|an)\b", short):
            short = "the" + short[short.index(" "):] if " " in short else "the person"
        elif not re.match(r"(?i)(?:the|his|her|their|its)\b", short) and not short[:1].isupper():
            short = "the " + short
        return short

    @classmethod
    def _official_layout(cls, text):
        """Write the prompt in the layout MiniMax documents for its input mode (skill h3-prompt-writing):
        - no reference pictures: T2VA, only integrated_multimodal_description / overall_soundscape /
          non_diegetic_music, the style at the start of [Shot 1] and the people described in words (no labels);
        - one picture that is the first frame: I2VA, the alignment line, then the same three fields;
        - reference pictures: Ref2VA, the six sections subject_definitions / summary (task in brackets) /
          retention_analysis / detailed_description / overall_soundscape / non_diegetic_music; a picture that is
          also the first frame adds keyframe completion and its own <Picture 1> line."""
        if not isinstance(text, str) or not re.search(r"(?m)^(?:integrated_multimodal_description|detailed_description):",
                                                       text):
            return text
        if re.search(r"(?m)^(?:summary|retention_analysis):", text):
            return text  # already in a six-section layout
        first_frame = cls.I2V_ALIGNMENT in text
        # MiniMax's own cut wording (the post-processor keeps cuts internally as 'cut to')
        text = re.sub(r"(\[Shot \d+\] At \d\d:\d\d\.\d{3}, )cut (back )?to\b",
                      lambda m: f"{m.group(1)}the camera cuts {m.group(2) or ''}to", text)
        body = re.sub(r"For the target video, at 0\.00 seconds into the target video, <Picture 1> \(from[^)]*\) is fully "
                      r"referenced\.?[ \t]*", "", text).strip()
        body = re.sub(r"^\s*\[[^\]\n]+\]\s*\n", "", body).strip()  # the old single header
        parts = cls._layout_sections(body)
        description = parts.get("integrated_multimodal_description") or parts.get("detailed_description") or ""
        sound = parts.get("overall_soundscape") or "N/A"
        music = parts.get("non_diegetic_music") or "N/A"
        definitions = []
        for line in (parts.get("subject_definitions") or "").split("\n"):
            match = re.match(r"\s*(<Subject \d+>)\s*(?:\(S[\d, S]+\)\s*)?(?:is|are)\s+(.+?)\s*$", line)
            if match:
                definitions.append((match.group(1), match.group(2).rstrip("."), line.strip()))
        pictures = sorted(set(re.findall(r"<Picture \d+>", body)))
        references = [p for p in pictures if not (first_frame and p == "<Picture 1>")]
        if references or (first_frame and len(pictures) > 1):
            return cls._ref2va_layout(description, sound, music, definitions, parts, first_frame)
        # T2VA / I2VA: the style opens [Shot 1]; every subject is introduced in words once, then named briefly
        preamble, marker, shots = description.partition("[Shot 1]")
        if not marker:
            preamble, shots = "", description
        names, intro = {}, []
        for label, described, _ in definitions:
            if first_frame and "<Picture 1>" in described:
                described = re.split(r"\s+(?:Target appearance|Wardrobe|Body proportions)\b", described)[0].rstrip(" .;,:")
                described += ", exactly as seen in <Picture 1> at 0.00 seconds"
            names[label] = cls._short_name(described)
            intro.append(described)
        seen = {}
        for label, short in list(names.items()):  # two people never share one name
            if short in seen:
                for other in (label, seen[short]):
                    full = next(d for l, d, _ in definitions if l == other)
                    names[other] = cls._short_name(re.sub(r"\s+(in|of|from|at)\s+", r" \1_", full, 1)).replace("_", " ")
            seen[short] = label
        if len(set(names.values())) < len(names):
            for order, label in enumerate(sorted(names), 1):
                names[label] = f"{names[label]} ({['first', 'second', 'third', 'fourth', 'fifth'][min(order, 5) - 1]})"

        def rename(chunk):
            def one(match):
                name = names.get(match.group(0))
                if not name:
                    return match.group(0)
                before = chunk[:match.start()].rstrip()
                start = not before or before[-1] in ".!?:]" or before.endswith("</d>")
                return name[:1].upper() + name[1:] if start else name
            return re.sub(r"<Subject \d+>", one, chunk)
        opening = preamble.strip()
        if intro:
            features = intro[0] if len(intro) == 1 else "; ".join(intro[:-1]) + "; and " + intro[-1]
            opening = (opening + " " if opening else "") + "The scene features " + features + "."
        shots = shots.strip()
        layout = "integrated_multimodal_description:\n[Shot 1] " + rename((opening + " " + shots).strip()) if names else \
            "integrated_multimodal_description:\n[Shot 1] " + (opening + " " + shots).strip()
        layout += "\n\noverall_soundscape:\n" + (rename(sound) if names else sound)
        layout += "\n\nnon_diegetic_music:\n" + (rename(music) if names else music)
        return (cls.I2V_ALIGNMENT + "\n\n" + layout) if first_frame else layout

    @classmethod
    def _ref2va_layout(cls, description, sound, music, definitions, parts, first_frame):
        task = "reference generation" + (" + keyframe completion" if first_frame else "")
        lines = [line for _, _, line in definitions] or [
            line.strip() for line in (parts.get("subject_definitions") or "").split("\n")
            if line.strip() and line.strip().upper() != "N/A"]
        if first_frame and not any(line.startswith("<Picture 1>") for line in lines):
            lines.append("<Picture 1> is the first frame of [Shot 1]; the video opens exactly on it.")
        markers = [(int(m.group(1)), m.start()) for m in re.finditer(r"\[Shot (\d+)\]", description)]

        def shots_of(label):
            found = []
            for index, (number, start) in enumerate(markers):
                end = markers[index + 1][1] if index + 1 < len(markers) else len(description)
                if label in description[start:end]:
                    found.append(f"[Shot {number}]")
            return found
        retention, referenced = [], []
        for label, described, _ in definitions:
            source = re.search(r"<Picture \d+>", described)
            if not source:
                continue
            referenced.append((label, source.group(0)))
            traits = re.split(r"\s+(?:Target appearance|Wardrobe|Body proportions)\b", described)[0]
            traits = traits.split(source.group(0), 1)[-1].strip(" ,.;:")
            traits = re.sub(r"^(?:with|wearing)\s+", "", traits) or f"the identity and appearance from {source.group(0)}"
            where = shots_of(label)
            changed = re.search(r"(?i)\b(?:Target appearance|Wardrobe:|now wears|instead of|replac)", described)
            status = ("partially_preserved - the identity is retained while the appearance follows the description"
                      if changed else f"fully_preserved - {traits} are retained")
            retention.append(f"{label}" + (f" (appears in {', '.join(where)})" if where else "") + f": {status}.")
        if first_frame:
            retention.append("<Picture 1>: fully_preserved - its composition, subjects and light are the opening frame "
                             "of [Shot 1].")
        opening = re.split(r"\[Shot 1\]", description, 1)
        action = opening[1] if len(opening) > 1 else description
        action = re.split(r"(?<=[.!?])\s+", re.sub(r"<d>.*?</d>|\x00\d+\x00", "", action).strip(), 1)[0]
        action = " ".join(action.split()[:40]).rstrip(" ,;:.!?")
        cast = ", ".join(f"{label} from {source}" for label, source in referenced)
        count = max(1, len(markers))
        summary = (f"[{task}] The target video unfolds over {count} shot{'s' if count > 1 else ''}"
                   + (f" with {cast}" if cast else "") + (f". {action[:1].upper() + action[1:]}" if action else "")
                   + (". <Picture 1> is the opening frame." if first_frame else "")).rstrip(".") + "."
        return ("subject_definitions:\n" + ("\n".join(lines) or "N/A")
                + "\n\nsummary:\n" + summary
                + "\n\nretention_analysis:\n" + ("\n".join(retention) or "N/A")
                + "\n\ndetailed_description:\n" + description.strip()
                + "\n\noverall_soundscape:\n" + sound
                + "\n\nnon_diegetic_music:\n" + music)

    @staticmethod
    def _order_reference_subjects(prompt):
        """Same subject ordering as the original engine, but speaker IDs (SN) are left alone: they follow the order
        of the vocal events, not the subject numbers."""
        match = re.search(
            r'(?ims)^subject_definitions:\s*(.*?)(?=^(?:integrated_multimodal_description|summary|retention_analysis|'
            r'detailed_description|overall_soundscape|non_diegetic_music):|\Z)', prompt)
        if not match:
            return prompt
        definitions = match.group(1)
        entries = list(re.finditer(r'(?m)^\s*<Subject\s+(\d+)>', definitions))
        mapping = {}
        for index, entry in enumerate(entries):
            end = entries[index + 1].start() if index + 1 < len(entries) else len(definitions)
            pictures = set(re.findall(r'<Picture\s+(\d+)>', definitions[entry.end():end], re.IGNORECASE))
            if len(pictures) == 1:
                mapping[int(entry[1])] = int(next(iter(pictures)))
        if not entries or not mapping or len(set(mapping.values())) != len(mapping):
            return prompt
        used = set(mapping.values())
        next_number = max(used | {int(entry[1]) for entry in entries}) + 1
        for entry in entries:
            old = int(entry[1])
            if old not in mapping:
                new = old if old not in used else next_number
                mapping[old] = new
                used.add(new)
                next_number = max(next_number, new + 1)

        def renumber(value):
            return re.sub(r'<Subject\s+(\d+)>', lambda item: f'<Subject {mapping.get(int(item[1]), int(item[1]))}>',
                          value, flags=re.IGNORECASE)
        blocks = []
        for index, entry in enumerate(entries):
            end = entries[index + 1].start() if index + 1 < len(entries) else len(definitions)
            blocks.append((mapping[int(entry[1])], renumber(definitions[entry.start():end]).strip()))
        ordered = '\n'.join(text for _, text in sorted(blocks, key=lambda item: item[0]))
        return renumber(prompt[:match.start(1)]) + ordered + '\n\n' + renumber(prompt[match.end(1):])

    @staticmethod
    def _normalize_h3(text):
        """Repair the H3 grammar of a delivered prompt: <Subject N> tags, no placeholder speech, no stray tag
        tails, and the four reference-generation sections present."""
        if not isinstance(text, str) or not text.strip():
            return text
        text = re.sub(r"(?<![<\w])Subject (\d+)(?![\w>])", r"<Subject \1>", text)
        request = _CLIP.get("request", "").lower()
        text = H3CompactMultimodalEditDirectorFix._PLACEHOLDER_SPEECH.sub(  # user-supplied lines are never removed
            lambda m: m.group(0) if (re.sub(r"^.*?<d>\[[^\]]+\]\s*|</d>$", "", m.group(0).strip()).strip().lower()[:40]
                                     in request and request) else "", text)
        text = re.sub(r"\s*\(?<Independent extra>\)?", "", text)
        text = re.sub(r"\s*<Subject \d+> performs the (?:full |entire |whole |complete )?(?:observed )?"
                      r"(?:action|sequence|performance|routine)\b[^.\n]*\.", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<=[A-Za-z0-9)])( Identity referenced from <Picture \d+>)", r".\1", text)
        text = re.sub(r"(?:\s*<(?:Subject|Picture|Video) \d+>)+\s*$", "", text)  # tags left dangling at the end
        text = re.sub(r"(?m)(?:\s*<(?:Subject|Picture|Video) \d+>){2,}\s*$", "", text)
        if "subject_definitions:" in text and "detailed_description:" not in text:
            if "integrated_multimodal_description:" not in text:
                shot = re.search(r"\[Shot 1\]", text)
                if shot:
                    text = (text[:shot.start()].rstrip() + "\n\nintegrated_multimodal_description:\n"
                            + text[shot.start():])
            if "overall_soundscape:" not in text:
                text = text.rstrip() + "\n\noverall_soundscape:\nN/A"
            if "non_diegetic_music:" not in text:
                text = text.rstrip() + "\n\nnon_diegetic_music:\nN/A"
        if _CLIP.get("mode") == "transfer":  # sound written inside the description goes to its own section
            text = H3CompactMultimodalEditDirectorFix._relocate_sound(text)
        return text

    _SHOT_WORDS = re.compile(
        r"close-up|two-shot|over-the-shoulder|wide|insert|single|medium|tight|low angle|high angle|profile|pov",
        re.IGNORECASE)

    @classmethod
    def _inherited_framing(cls, rendered, declared):
        """Framing the next scene must open on: the viewpoint of this scene's real last cut (the model's own
        closing_shot is kept only when it agrees with it)."""
        body = str(rendered).split("integrated_multimodal_description:")[-1].split("overall_soundscape:")[0]
        last_shot = re.split(r"\[Shot \d+\]", body)[-1]
        ends = re.findall(
            r"(?:pushes in|pulls (?:back|out)|moves in|tightens|settles|settling|ending|ends|until|landing|closing|"
            r"finishing|holds)[^.;]{0,40}?\b(?:on|in|into|to|at|with)\s+(?:a |an |the )?((?:extreme |tight |medium |wide |"
            r"full |close |low |high |steady )*(?:close-up|closeup|medium shot|wide shot|two-shot|full shot|close shot|"
            r"medium close-up|over-the-shoulder(?: shot)?)[^.;]{0,60})", last_shot, flags=re.IGNORECASE)
        if ends:  # a push-in or a settle changes the size: the next scene opens on the END of the last shot
            view = ends[-1].strip().rstrip(",")
            return view[0].upper() + view[1:] + ", continuing from the previous scene"
        cuts = re.findall(r"\[Shot \d+\] At [\d:.]+, cut (?:back )?to (.+?)(?=\[Shot \d+\]|\n|$)", rendered)
        if not cuts:
            return declared
        match = re.match(r"(?:the |a |an |his |her |their )?(.{5,110}?)(?=[.,;]| as | while | with | when |$)",
                         cuts[-1].strip(), flags=re.IGNORECASE)
        if not match:
            return declared
        view = match.group(1).strip()
        if not cls._SHOT_WORDS.search(view):  # the cut opens with an action: find its real end framing
            end = re.findall(r"(?:ending|ends|finishing|settling|holding|landing)\s+(?:in|on|with|into)\s+(?:a |an |the )?"
                             r"([^.;]{3,110}?\b(?:shot|close-up|closeup|framing|view|frame))\b", cuts[-1],
                             flags=re.IGNORECASE)
            framed = [m.group(0) for m in cls._SHOT_WORDS.finditer(cuts[-1])]
            if end:
                view = end[-1].strip()
            elif declared:
                return declared
            elif framed:
                view = framed[-1]
            else:
                return declared
        wanted = {w.lower() for w in cls._SHOT_WORDS.findall(view)}
        if declared and wanted and wanted <= {w.lower() for w in cls._SHOT_WORDS.findall(declared)}:
            return declared
        return view[0].upper() + view[1:] + ", continuing from the previous scene"

    @staticmethod
    def _drop_identity_cap(text):
        """The identity clause must not restate an accessory the target appearance redefines."""
        def fix(match):
            line = match.group(0)
            for item in ("cap", "hat", "beanie", "glasses", "sunglasses", "scarf"):
                if re.search(rf"Target appearance:[^\n]*\b{item}\b", line):
                    line = re.sub(rf",\s*wearing (?:an? )?[^.,\n]*?\b{item}\b[^.,\n]*?(?=[.,]?\s*Target appearance:)", "", line)
            return line
        return re.sub(r"(?m)^<Subject \d+> is [^\n]*$", fix, text)

    @staticmethod
    def _subject_tags(text):
        """Inside the footage, call each lead <Subject N> instead of 'the woman in <Picture M>'."""
        mapping = {pic: subj for subj, pic in re.findall(
            r"(?m)^(<Subject \d+>) is [^\n]*?<Picture (\d+)>", text)}
        marker = "integrated_multimodal_description:"
        if not mapping or marker not in text:
            return text
        head, body = text.split(marker, 1)
        for pic, subj in mapping.items():
            body = re.sub(
                r"\b(?:[Tt]he )(?:woman|man|person|girl|boy|character) in <Picture " + pic + r">", subj, body)
        return head + marker + body

    @staticmethod
    def _render_creative_scene(scene, design, opening_frame=False):
        if isinstance(scene, dict):
            previous = _CLIP.get("prev_closing")
            if previous and not opening_frame:  # chain: this scene opens on the last shot of the previous one
                previous = re.sub(r"[,;]?\s*held steady for (?:the )?(?:final|last)\s+\w+\s+seconds?",
                                  "", previous, flags=re.IGNORECASE)
                scene = dict(scene, opening_framing=previous)
            declared = str(scene.get("closing_shot", "")).strip() or None
        else:
            declared = None
        out = _Base._render_creative_scene(scene, design, opening_frame)
        masked, store = H3CompactMultimodalEditDirectorFix._mask_d(out)
        try:
            masked = H3CompactMultimodalEditDirectorFix._split_speakers(
                H3CompactMultimodalEditDirectorFix._speaker_tags(masked), _CLIP["seconds"])
        except Exception:
            pass
        masked = H3CompactMultimodalEditDirectorFix._timecode_shots(masked, _CLIP["seconds"], store)
        try:
            masked = H3CompactMultimodalEditDirectorFix._carry_posture(masked)
        except Exception:  # noqa: BLE001 - a clean-up never breaks the prompt
            pass
        out = H3CompactMultimodalEditDirectorFix._scrub_music(H3CompactMultimodalEditDirectorFix._unmask_d(masked, store))
        _CLIP["prev_closing"] = H3CompactMultimodalEditDirectorFix._inherited_framing(out, declared)
        out = re.sub(
            r",?\s*(?:identity |fully )?referenced from (<(?:Picture|Video) \d+>)\.?",
            r"; identity taken from \1.", out, flags=re.IGNORECASE)
        out = re.sub(r"(?m)^(<Subject \d+>) (?:is )?[Tt]he ", r"\1 is the ", out)
        out = re.sub(r"(?m)^(<Subject \d+>) (?:is )?(an?) ", lambda m: f"{m.group(1)} is {m.group(2).lower()} ", out,
                     flags=re.IGNORECASE)
        out = re.sub(r"</d>\.", "</d>", out)
        out = re.sub(r"(?<=[A-Za-z0-9>)])( Target appearance:| Body proportions:)", r".\1", out)
        out = H3CompactMultimodalEditDirectorFix._drop_identity_cap(
            H3CompactMultimodalEditDirectorFix._subject_tags(out))
        if _CLIP.get("wardrobe_new"):  # tell the video model the picture's clothes are not part of the target
            out = re.sub(r"(?m)^(<Subject \d+> [^\n]*?(<Picture \d+>)[^\n]*?) Target appearance:",
                         lambda m: m.group(0) if (_CLIP.get("i2v") and m.group(2) == "<Picture 1>") else
                         f"{m.group(1)} Target appearance (a new outfit; the clothes worn in {m.group(2)} are not used "
                         "unless the request names them):", out)
        if _CLIP.get("restyle_outfit"):  # Keep Reference + Outfit style: same clothes, visibly new cut
            style = _CLIP["restyle_outfit"]
            out = re.sub(r"(?m)^(<Subject \d+> [^\n]*?(<Picture \d+>)[^\n]*?) Target appearance:",
                         lambda m: f"{m.group(1)} Target appearance (the clothes of {m.group(2)} restyled to {style}: "
                         "same colours and fabrics, new cut; not worn exactly as in the picture):", out)
        if _CLIP.get("body_new"):
            out = re.sub(r"(?m)^(<Subject \d+> [^\n]*?(<Picture \d+>)[^\n]*?) Body proportions:",
                         lambda m: m.group(0) if (_CLIP.get("i2v") and m.group(2) == "<Picture 1>") else
                         f"{m.group(1)} Body proportions (adapted; the body shape in {m.group(2)} is not kept):", out)
        if _CLIP.get("environment_new"):
            pics = sorted(set(re.findall(r"<Picture \d+>", out)) - ({"<Picture 1>"} if _CLIP.get("i2v") else set()))
            if pics and "integrated_multimodal_description:\n" in out:
                out = out.replace("integrated_multimodal_description:\n",
                                  "integrated_multimodal_description:\nA newly designed setting; the backgrounds of "
                                  + ", ".join(pics) + " are not used. ", 1)
        return out

    _PATCH_RULES = (
        ("hold_provider", re.escape('if str(held.get("provider_identity") or "") != provider_identity:'),
         'if cls._provider_changed(held.get("provider_identity"), provider_identity):'),
        ("one_selection_c", r"Blend a second '\s*'selection coherently rather than alternating incompatible designs\. ", "' '"),
        ("one_selection_a", re.escape('"Blend the second selection as a compatible accent; "'), '""'),
        ("one_selection_b", r'If two selections conflict, the first is primary; "\s*"use the second only in a compatible '
                            r'way rather than canceling the first\. "', '"'),
        ("hold_capture", re.escape("_set_held_director_plan(cache_key, {"), "cls._hold_capture(cache_key, {"),
        ("hold_seconds", re.escape('if bool(held.get("i2v_mode", False)) != bool(i2v_mode):'),
         'if cls._hold_seconds_changed(held, scene_seconds) or bool(held.get("i2v_mode", False)) != bool(i2v_mode):'),
        ("raw_default", r"repair_retry_used = False",  # a repaired reply without debug left raw_response unset
         'raw_response = ""; repair_retry_used = False'),
        ("core_elab", r'resolved_system, cls\.DEEP_EDIT_RULES, cls\.ELABORATE_RULES,',
         'resolved_system, cls.CREATIVE_EDIT_CORE, cls.ELABORATE_RULES,'),
        ("core_plus", r'cls\.DEEP_EDIT_RULES,\s*cls\.CONTINUOUS_EDIT_RULES,\s*cls\.CONTINUOUS_ELABORATE_RULES,',
         'cls.CREATIVE_EDIT_CORE, cls.CONTINUOUS_ELABORATE_RULES,'),
        ("look_specs", r"specification = VISUAL_LOOKS\.get\(selection\)",
         "specification = (cls.FIX_LOOKS[selection] + cls.CAPTURE_ONLY) if selection in cls.FIX_LOOKS else VISUAL_LOOKS.get(selection)"),
        ("motion_specs", r"specification = MOTION_STYLES\.get\(selection\)",
         "specification = cls.FIX_MOTIONS.get(selection) or MOTION_STYLES.get(selection)"),
        ("source_speech", re.escape("contract and explicit locks; do not invent story or speech. "),
         "contract and explicit locks; do not invent story (speech only when the request or an active Dialogue control asks). "),
        ("mapping_enum", re.escape('"identity_source": {"type": "string", "enum": [f"<Picture {index}>" for index, _ in pictures]},'),
         '"identity_source": {"type": "string", "enum": [f"<Picture {index}>" for index, _ in pictures] + cls._extra_slot(role_names, pictures)},'),
        ("mapping_allowed", re.escape('allowed_sources = {f"<Picture {index}>" for index, _ in pictures}'),
         'allowed_sources = {f"<Picture {index}>" for index, _ in pictures} | set(cls._extra_slot(role_names, pictures))'),
        ("mapping", r'mapping = parsed\["driven_cast_mapping"\]',
         'mapping = parsed["driven_cast_mapping"] = cls._fix_driven_mapping(parsed.get("driven_cast_mapping"), role_names, pictures)'),
        ("transfer_cast_get", r"cls\._render_transfer_cast\(parsed\['transfer_cast'\],",
         "cls._render_transfer_cast(parsed.get('transfer_cast') or [],"),
        ("source_item", r"A source item may remain if coherent with the ",
         "A source item may remain (never with Wardrobe Create New) if coherent with the "),
        ("transfer_controls", r"cls\.TRANSFER_OWNERSHIP_RULES,",
         "cls.TRANSFER_OWNERSHIP_RULES + cls._control_rules_text(direction_text),"),
        ("controls", r'"\\n\\nSELECTED DIRECTION CONTRACT:\\n" \+ direction_text',
         '"\\n\\nSELECTED DIRECTION CONTRACT:\\n" + direction_text + cls._control_rules_text(direction_text)'),
        ("contract", r'"Open creative premises may develop compatible story within the continuous take; "',
         '"Open creative premises may develop compatible story and edited coverage; "'),
        ("env", r"'requested/story-authorized location transitions\. No extra section or API call\.'",
         "'requested/story-authorized location transitions. No extra section or API call. ' + cls.ENVIRONMENT_NOTE"),
        ("seconds_limit", r"not 1\.0 <= scene_seconds <= 30\.0:", "not 1.0 <= scene_seconds <= 60.0:"),
        ("seconds_msg", r"seconds_per_scene must be between 1 and 30 seconds\.", "seconds_per_scene must be between 1 and 60 seconds."),
        ("long_words", r'\+ \("90–140" if scene_seconds <= 5 else "120–200" if scene_seconds <= 10 else "160–240"\)',
         '+ ("90–140" if scene_seconds <= 5 else "120–200" if scene_seconds <= 10 else "160–240" if scene_seconds <= 20 '
         'else f"{round(scene_seconds * 8)}–{round(scene_seconds * 12)}")'),
        ("long_tokens", r"elaborate_tokens_per_scene = 500 if scene_seconds <= 5 else 700 if scene_seconds <= 10 else 900",
         "elaborate_tokens_per_scene = 500 if scene_seconds <= 5 else 700 if scene_seconds <= 10 else 900 if scene_seconds <= 15 else int(65 * scene_seconds)"),
        ("coverage", r"A pan/reframe is NOT a new \[Shot n\]; add another '\s*'shot marker only for a distinct timed beat; sequence blocks continue the preceding '",
         "' + cls.coverage_note(scene_seconds) + 'Sequence blocks continue the preceding '"),
        ("dialogue", r"'SHORT DIALOGUE, VISIBLE STORY: Prefer brief lines with a tactic, reply or subtext '",
         "'SHORT DIALOGUE, VISIBLE STORY: ' + cls.DIALOGUE_CRAFT_NOTE + ' Prefer brief lines with a tactic, reply or subtext '"),
        ("brief_prop", r"'subjects': \{'type': 'array', 'items': \{",
         "'creative_brief': __import__('copy').deepcopy(cls.CREATIVE_BRIEF_SCHEMA), 'subjects': {'type': 'array', 'items': {"),
        ("brief_req", r"'required': \['subjects', 'visual_style'\],",
         "'required': ['creative_brief', 'subjects', 'visual_style'],"),
        ("opening_desc", r"'opening_framing': \{'type': 'string'\},",
         "'opening_framing': {'type': 'string', 'description': cls.OPENING_FIELD_NOTE}, "
         "'coverage_plan': {'type': 'string', 'description': cls.COVERAGE_PLAN_NOTE}, "
         "'speech_plan': {'type': 'string', 'description': cls.SPEECH_PLAN_NOTE},"),
        ("closing_after", r"(Respect silence and source-audio locks\.'\},)",
         "Respect silence and source-audio locks.'}, "
         "'closing_shot': {'type': 'string', 'description': cls.CLOSING_FIELD_NOTE},"),
        ("closing_req", r"'required': \['opening_framing', 'integrated_multimodal_description', 'visible_result',",
         "'required': ['opening_framing', 'closing_shot', 'coverage_plan', 'speech_plan', 'integrated_multimodal_description', 'visible_result',"),
        ("music_desc", r"'non_diegetic_music': \{'type': 'string'\},",
         "'non_diegetic_music': {'type': 'string', 'description': cls.MUSIC_FIELD_NOTE},"),
        ("visible_desc", r"'description': 'The earned ending state, not another copy of already delivered dialogue or action\.'",
         "'description': 'The earned ending state, not another copy of already delivered dialogue or action. ' + cls.VISIBLE_RESULT_NOTE"),
        ("shot_desc", r"'description': 'Chronological footage and any permitted/requested speech\. ",
         "'description': cls.SHOT_FIELD_NOTE + ' Chronological footage and any permitted/requested speech. "),
        ("root_a", r"return json\.loads\(candidate\)(?=\n)",
         "return cls._root_object(json.loads(candidate))"),
        ("root_b", r"return json\.loads\(candidate, strict=False\)(?=\n)",
         "return cls._root_object(json.loads(candidate, strict=False))"),
        ("json_slice", r"return json\.loads\(candidate\[start:end \+ 1\], strict=False\)",
         "return cls._lenient_json(candidate, first_error)"),
        ("json_none", r"raise first_error",
         "return cls._lenient_json(candidate, first_error)"),
        ("repair_tokens", r'"max_tokens": payload\["max_tokens"\],',
         '"max_tokens": min(int(payload["max_tokens"] * 1.6), 8000),'),
        ("wardrobe", r"'Only identity traits come from Picture references; new garments are target design\.'",
         "'Only identity traits come from Picture references; new garments are target design. ' + cls.ACCESSORY_NOTE"),
    )
    @classmethod
    def _get_patched_execute(cls):
        """Parent execute with prompt literals replaced rule by rule (partial patches allowed)."""
        state = _PATCH_STATE
        if state["tried"]:
            return state["fn"]
        state["tried"] = True
        rules = H3CompactMultimodalEditDirectorFix._PATCH_RULES
        try:
            source = textwrap.dedent(inspect.getsource(_Base.execute.__func__))
            source = "\n".join(
                line for line in source.splitlines() if not line.startswith("@classmethod"))
            failed = []
            for key, pattern, replacement in rules:
                source, count = re.subn(pattern, lambda m, r=replacement: r, source)
                if count != 1:
                    failed.append(key)
            state["failed"] = tuple(failed)
            state["report"] = ("⚠ " if failed else "") + f"patch {len(rules) - len(failed)}/{len(rules)}" + (
                " missing:" + ",".join(failed) + " (fallback rules active)" if failed else "")
            namespace = {}
            exec(compile(source, "<h3_director_fix_execute>", "exec"),
                 vars(sys.modules[_Base.__module__]), namespace)
            state["fn"] = namespace["execute"]
            if failed:
                print(f"[H3 Director] {state['report']}")
        except Exception as error:  # fall back to the unpatched engine
            state["failed"] = tuple(key for key, _, _ in rules)
            state["report"] = "⚠ patch OFF (" + type(error).__name__ + ", original rules in use)"
            print(f"[H3 Director] prompt patch disabled: {error!r}")
        return state["fn"]

    # --------------------------------------------------------------- schema
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3CompactMultimodalEditDirector",
            display_name="H3 Compact Multimodal Edit Director",
            category="text/minimax_h3",
            search_aliases=[
                "h3 compact edit director fix", "h3 director fix", "h3 creative director",
            ],
            description=(
                "Compact optional-source director for background, identity, wardrobe, object, "
                "pose, action, style and relighting edits. Leaner rules, decision-making creative toolkit. Requires an external LLM provider."
            ),
            inputs=[
                io.String.Input(
                    "edit_request", multiline=True, dynamic_prompts=False,
                    default="Describe the exact edit and assign each connected source its role.",
                ),
                io.Combo.Input(
                    "edit_mode", options=["Normal", "Compact", "Edit", "Transfer", "Transfer Keep Audio", "Transfer Keep Audio Basic"], default="Normal",
                    tooltip=(
                        "Normal: the full creative director. It writes the story from your request, the references "
                        "and the Direction Controls, one prompt per scene (Scenes sets how many), with continuity "
                        "between scenes. Compact: the same director and rules with prompts about half as long "
                        "(fewer adjectives, light and sound in brief), cheaper and faster. Edit: H3 video editing of the "
                        "Guide Video (it becomes <Video 1>; for Viggle Remake swap/edit and similar): only what you "
                        "ask changes, movement, cuts and audio stay, six-section edit prompt. Transfer: puts the "
                        "people or objects of your references into the guide video's movement (Video Task: Swap, "
                        "Edit, Add, Remove), adding only what you or the controls ask. Transfer Keep Audio: the same, "
                        "keeping the guide video's own audio (no new dialogue, sound or music). Transfer Keep Audio Basic: the "
                        "same, with a very short prompt about who, wearing what, where and the light only (the video "
                        "already carries the movement)."
                    ),
                ),
                io.Image.Input("reference_image_1", optional=True),
                io.Image.Input("reference_image_2", optional=True),
                io.Image.Input("reference_image_3", optional=True),
                io.Image.Input("reference_image_4", optional=True),
                io.Image.Input("source_video_1", optional=True),
                io.Image.Input("source_video_2", optional=True),
                io.Image.Input("driven_image", display_name="Guide Image", optional=True,
                    tooltip="Guide: activity, setting and staging to adapt onto the numbered reference cast, combined with the user's request. Not an identity reference."),
                io.Image.Input("driven_video", display_name="Guide Video", optional=True,
                    tooltip="Guide: decoded video frames supplying visible activity, motion, interaction and progression for the reference cast. No audio analysis; user instructions take priority."),
                io.Combo.Input("video_samples", options=["3", "5", "10"], default="3"),
                io.String.Input(
                    "system_prompt", multiline=True, dynamic_prompts=False,
                    default=cls.DEFAULT_SYSTEM_PROMPT,
                ),
                io.Int.Input("max_tokens", default=1200, min=256, max=3072),
                io.Float.Input("temperature", default=0.2, min=0.0, max=1.0, step=0.05),
                io.Boolean.Input("reasoning", default=False),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input(
                    "image_max_dimension", default=1024, min=512, max=2048,
                    step=64, advanced=True,
                ),
                io.Int.Input(
                    "timeout_seconds", default=300, min=30, max=900, advanced=True,
                ),
                io.Boolean.Input(
                    "hold_prompt", display_name="Hold Edit Prompt", default=False,
                ),
                io.Boolean.Input(
                    "i2v_mode", display_name="I2V Mode — Picture 1 is First Frame",
                    default=False,
                    tooltip=(
                        "When enabled, Picture 1 is treated as the exact opening frame and every "
                        "director mode writes motion beginning from its visible state."
                    ),
                ),
                io.Int.Input(
                    "continuous_scene_count", display_name="Scenes — Sequence",
                    default=3, min=1, max=12, step=1,
                    tooltip=(
                        "Number of generated scenes for sequence modes. More "
                        "scenes increase the required LLM output budget."
                    ),
                ),
                io.Float.Input(
                    "seconds_per_scene", display_name="Seconds per Scene",
                    default=5.0, min=1.0, max=60.0, step=0.5, force_input=True,
                    tooltip=(
                        "Connected FLOAT containing the 1–60 second screen-time budget used by "
                        "Elaborate and Elaborate Plus."
                    ),
                ),
                io.String.Input(
                    "direction_context", optional=True, force_input=True,
                    tooltip="Optional Direction Controls reinforcement. Empty or disconnected preserves the existing behavior.",
                ),
                io.Custom("LLMMODEL").Input(
                    "llm", optional=True,
                    tooltip="External provider, required unless bypass is enabled. No internal API or fallback.",
                ),
                io.Boolean.Input(
                    "bypass", default=False,
                    tooltip="Return the user prompt verbatim. Skip LLM, references, direction controls and Hold.",
                ),
                io.Boolean.Input(
                    "debug_request", default=False,
                    tooltip="Expose sanitized request text and parameters, without image data or credentials. No extra LLM call.",
                ),
            ],
            outputs=[
                io.String.Output("edit_prompt"),
                io.String.Output("validation"),
                io.String.Output("usage_stats"),
                io.Int.Output("continuous_scene_count"),
                io.String.Output("debug_request"),
                io.String.Output("raw_response"),
            ],
            hidden=[io.Hidden.unique_id],
        )


    @classmethod
    def execute(cls, *args, **kwargs):
        _CLIP.reset()  # a fresh state for this execution only
        _CLIP["real_llm"] = kwargs.get("llm")
        _CLIP["video_edit"] = str(kwargs.get("edit_mode", "")).strip().lower() == "edit"
        _CLIP["route_note"] = ""
        if (not _CLIP["video_edit"] and kwargs.get("driven_video") is not None
                and cls._same_clip(kwargs.get("driven_video"), kwargs.get("source_video_1"))):
            kwargs["source_video_1"] = None  # one clip, one analysis: the Guide route of this mode
            _CLIP["route_note"] = " · the same clip was in Guide Video and Video 1: analysed once, as Guide"
        _CLIP["ports"] = sorted(name for name in ("driven_video", "driven_image", "source_video_1", "source_video_2",
                                                  "reference_image_1", "reference_image_2", "reference_image_3",
                                                  "reference_image_4") if kwargs.get(name) is not None)
        if _CLIP["video_edit"]:  # the base video-edit sequence engine; the guide video is <Video 1>
            kwargs["edit_mode"] = "Continuous Edit"
            _CLIP["edit_pictures"] = [i for i in (1, 2, 3, 4) if kwargs.get(f"reference_image_{i}") is not None]
            if kwargs.get("source_video_1") is None and kwargs.get("driven_video") is not None:
                kwargs["source_video_1"] = kwargs.get("driven_video")
            kwargs["driven_video"] = None
            kwargs["driven_image"] = None
        _CLIP["videos"] = [i for i in (1, 2) if kwargs.get(f"source_video_{i}") is not None]
        _mode = str(kwargs.get("edit_mode", "")).strip().lower()
        if _mode in ("normal", "compact"):  # both run the full creative sequence engine
            _CLIP["compact"] = _mode == "compact"
            kwargs["edit_mode"] = "Elaborate Plus"
        _CLIP["basic"] = str(kwargs.get("edit_mode", "")).strip().lower() == "transfer keep audio basic"
        _CLIP["keep_audio"] = str(kwargs.get("edit_mode", "")).strip().lower() in ("transfer keep audio",
                                                                                  "transfer keep audio basic")
        if _CLIP["keep_audio"]:  # same Transfer engine; the video's own audio is the final audio
            kwargs["edit_mode"] = "Transfer"
        if _CLIP["video_edit"]:  # the source soundtrack is the final audio
            _CLIP["keep_audio"] = True
        try:
            clip = float(kwargs.get("seconds_per_scene", 10.0))
        except (TypeError, ValueError):
            clip = 10.0
        _CLIP["seconds"] = clip
        sequence_modes = ("elaborate plus", "continuous edit", "transfer", "continuous elaborate", "elaborate continuo")
        try:
            _CLIP["scenes"] = (max(1, min(12, int(kwargs.get("continuous_scene_count", 3))))
                               if str(kwargs.get("edit_mode", "")).strip().lower() in sequence_modes else 1)
        except (TypeError, ValueError):
            _CLIP["scenes"] = 1
        _CLIP["prev_closing"] = None
        creative_modes = {
            "elaborate", "elaborate plus", "continuous elaborate", "elaborate continuo"}
        if (
            str(kwargs.get("edit_mode", "")).strip().lower() in creative_modes
            and str(kwargs.get("system_prompt", "")).strip() == cls.DEFAULT_SYSTEM_PROMPT.strip()
        ):
            kwargs["system_prompt"] = cls.CREATIVE_BASE_PROMPT
            try:  # room for the short creative_brief the schema now asks for first
                per_scene = 650 if clip <= 5 else 1000 if clip <= 10 else 1250 if clip <= 15 else int(85 * clip)
                if _CLIP.get("compact"):
                    per_scene = int(per_scene * 0.6)
                scenes = (max(1, min(12, int(kwargs.get("continuous_scene_count", 3))))
                          if str(kwargs.get("edit_mode", "")).strip().lower() != "elaborate" else 1)
                kwargs["max_tokens"] = max(int(kwargs.get("max_tokens", 1200)), scenes * per_scene + 1000)
            except (TypeError, ValueError):
                pass
            cls._get_patched_execute()  # builds the patch report used below
            missing = set(_PATCH_STATE["failed"])
            notes = []
            if "coverage" in missing:
                try:
                    seconds = float(kwargs.get("seconds_per_scene", 5.0))
                except (TypeError, ValueError):
                    seconds = 5.0
                notes.append(cls.coverage_note(seconds))
            if "dialogue" in missing:
                notes.append(cls.DIALOGUE_CRAFT_NOTE)
            if "wardrobe" in missing:
                notes.append(cls.ACCESSORY_NOTE)
            if "env" in missing:
                notes.append(cls.ENVIRONMENT_NOTE)
            if notes:  # fallback: same craft rules, carried in the base prompt
                kwargs["system_prompt"] += (
                    "\n\nCRAFT RULES (these override any edit-contract text that forbids cuts, "
                    "new story or new design): " + " ".join(notes)
                )
        if str(kwargs.get("edit_mode", "")).strip().lower() in ("compact", "edit", "continuous edit"):
            kwargs["system_prompt"] = str(kwargs.get("system_prompt") or "") + "\n\n" + cls.OUTPUT_LANGUAGE_RULE
        if _CLIP.get("compact"):
            kwargs["system_prompt"] = str(kwargs.get("system_prompt") or "") + "\n\n" + cls.COMPACT_NOTE
        kwargs["system_prompt"] = str(kwargs.get("system_prompt") or "") + "\n\n" + cls.REQUEST_READING_NOTE
        if _CLIP.get("video_edit"):
            _vt = re.search(r"(?im)^Video task:\s*(Edit|Add|Remove)\b", str(kwargs.get("direction_context") or ""))
            _CLIP["edit_task"] = _vt.group(1).lower() if _vt else "swap"
            try:
                scenes = max(1, min(12, int(kwargs.get("continuous_scene_count", 3))))
                kwargs["max_tokens"] = max(int(kwargs.get("max_tokens", 1200)), scenes * 700 + 800)
            except (TypeError, ValueError):
                pass
        try:  # Genre / Situation on Auto in the creative modes: a random pick the request can override
            picks = cls._random_picks(kwargs)
            if picks:
                kwargs["system_prompt"] += "\n\n" + picks
            kwargs["direction_context"] = "\n".join(  # the picks are written above; the marker line is not a rule
                line for line in str(kwargs.get("direction_context") or "").splitlines()
                if not line.startswith("Random pick:"))
        except Exception:
            pass
        _CLIP["driven_note"] = ""
        _CLIP["extra_rule"] = ""
        _CLIP["request"] = str(kwargs.get("edit_request") or "")
        _CLIP["keep_shallow"] = bool(_CLIP.get("keep_shallow") or re.search(r"shallow|bokeh|desenfo|blur|depth of field|profundidad de campo",
                                               _CLIP["request"], re.IGNORECASE) or re.search(
            r"(?im)^(?:Visual look|Motion style):\s*\S", str(kwargs.get("direction_context") or "")))
        _CLIP["i2v"] = bool(kwargs.get("i2v_mode")) and kwargs.get("reference_image_1") is not None
        _CLIP["reference_nude"] = False
        _CLIP["blueprint_cache"] = None
        _CLIP["mode"] = str(kwargs.get("edit_mode", "")).strip().lower()
        _ctx = str(kwargs.get("direction_context") or "")
        if _CLIP["mode"] != "transfer" and re.search(r"(?im)^Wardrobe:\s*Guide Garments", _ctx):
            _ctx = re.sub(r"(?im)^Wardrobe:\s*Guide Garments.*$", "Wardrobe: Keep Reference", _ctx)
            kwargs["direction_context"] = _ctx  # no guide performance outside Transfer: the reference outfit
        _CLIP["wardrobe_new"] = bool(re.search(r"(?im)^Wardrobe:\s*Create New", _ctx))
        _CLIP["wardrobe_keep"] = bool(re.search(r"(?im)^Wardrobe:\s*Keep Reference", _ctx))
        _CLIP["body_neutral"] = (not re.search(r"(?im)^Body proportions:\s*(?:Creative|Keep Guide)", _ctx)
                                 and not cls._BODY_REQUEST.search(str(kwargs.get("edit_request") or ""))
                                 and any(kwargs.get(f"reference_image_{i}") is not None for i in range(1, 5)))
        _style = (re.search(r"(?im)^Outfit style:\s*(.+)$", _ctx) or [None, ""])[1].strip()
        _CLIP["restyle_outfit"] = (_style or _CLIP.get("restyle_outfit") or "") if _CLIP["wardrobe_keep"] else ""
        _CLIP["wardrobe_follow"] = bool(re.search(r"(?im)^Wardrobe:\s*Guide Garments", _ctx))
        _task = re.search(r"(?im)^Video task:\s*(Edit|Add|Remove)\b", _ctx)
        if _CLIP.get("video_edit"):  # Edit mode reads the Video Task itself (Swap when the line is absent)
            _CLIP["edit_task"] = _task.group(1).lower() if _task else "swap"
        _CLIP["task"] = _task.group(1).lower() if _task and _CLIP["mode"] == "transfer" else ""
        if _task and not _CLIP["task"]:  # outside Transfer the Video task line is dropped, never read by the director
            _ctx = re.sub(r"(?im)^Video task:.*(?:\n|$)", "", _ctx).strip()
            kwargs["direction_context"] = _ctx
        if _CLIP["task"] == "edit" and not any(kwargs.get(f"reference_image_{i}") is not None for i in (2, 3, 4)):
            _ctx = re.sub(r"(?im)^(Wardrobe|Hair & Makeup|Body proportions|Environment):\s*(?:Keep Reference|Guide Garments).*$",
                          r"\1: Keep Guide", _ctx)  # no Picture 2: nothing to take, the video stays as it is
            kwargs["direction_context"] = _ctx
            _CLIP["wardrobe_keep"] = _CLIP["wardrobe_follow"] = False
        _CLIP["edit_refs"] = ([i for i in (2, 3, 4) if kwargs.get(f"reference_image_{i}") is not None]
                              if _CLIP["task"] in ("edit", "add") else [])
        _keys = {"wardrobe": ("Wardrobe", "the outfit"), "hair": ("Hair & Makeup", "the hairstyle and makeup"),
                 "body": ("Body proportions", "the build"), "place": ("Environment", "the place")}
        _CLIP["edit_aspect_keys"] = [k for k, (label, _) in _keys.items()
                                     if re.search(rf"(?im)^{re.escape(label)}:\s*Keep Reference", _ctx)]
        _CLIP["edit_aspects"] = [_keys[k][1] for k in _CLIP["edit_aspect_keys"]]
        _CLIP["single_take"] = bool(re.search(r"(?im)^Coverage:\s*Single take", _ctx))
        _CLIP["dynamic"] = bool(re.search(r"(?im)^Coverage:\s*Dynamic", _ctx))
        _rate = re.search(r"(?im)^Coverage:\s*(Calm|Moderate|Dynamic)", _ctx)
        _CLIP["cut_every"] = {"calm": 12.5, "moderate": 7.0, "dynamic": 4.0}[_rate.group(1).lower()] if _rate else None
        _pace = (re.search(r"(?im)^Story pace:\s*(Fast|Very Fast)\s*$", _ctx) or [None, ""])[1].lower()
        if _pace and not _rate and not _CLIP["single_take"]:  # a fast story needs room for its cuts
            _CLIP["cut_every"] = 3.5 if _pace == "very fast" else 5.0
        try:
            _CLIP["seconds"] = float(kwargs.get("seconds_per_scene") or 10.0)
        except (TypeError, ValueError):
            _CLIP["seconds"] = 10.0
        _CLIP["checklist"] = cls.control_checklist(kwargs.get("direction_context"))  # after mode and task are known
        _era = bool(re.search(r"(?im)^Era:\s*\S", _ctx))
        _CLIP["mask"] = {
            "setting": bool(re.search(r"(?im)^Environment:\s*(Keep Reference|Create New)", _ctx)),
            "wardrobe": bool(re.search(r"(?im)^Wardrobe:\s*(Keep Reference|Create New)", _ctx)),
            "keep_guide": {label: bool(re.search(rf"(?im)^{label}:\s*Keep Guide", _ctx))
                           for label in ("Wardrobe", "Environment", "Body proportions", "Hair & Makeup")},
            "restyle": ([line.strip() for line in _ctx.splitlines() if ":" in line and not re.match(
                r"(?i)\s*(Coverage|Random pick|Action density|Action chain|Story pace|Dialogue density|Progression|Adults only|Palette: follows|Video task|LoRA)", line)
                and not re.search(r"(?i):\s*Keep Guide", line)]
                if _CLIP["mode"] == "transfer" else None),
            "hair": ("control" if re.search(r"(?im)^Hair & Makeup:\s*(Keep Reference|Create New)", _ctx)
                     else "legacy" if (_CLIP["wardrobe_keep"] and _CLIP["mode"] != "transfer"
                                       and not re.search(r"(?im)^Hair & Makeup:", _ctx)) else False),
            "body": bool(re.search(r"(?im)^Body proportions:\s*(Keep Reference|Creative Adaptation)", _ctx)),
            "look": bool(re.search(r"(?im)^Visual look:\s*\S", _ctx)),
            "mood": bool(re.search(r"(?im)^Genre:\s*\S", _ctx)),
        }
        _era_old = _era and _CLIP["mode"] != "transfer"  # Elaborate/other modes keep their v48 behaviour
        if _era_old:
            _CLIP["mask"].update(setting=True, wardrobe=True)
        _CLIP["environment_new"] = bool(re.search(r"(?im)^Environment:\s*Create New", _ctx))
        _CLIP["body_new"] = bool(re.search(r"(?im)^Body proportions:\s*Creative Adaptation", _ctx))
        has_driven = any(kwargs.get(name) is not None for name in ("driven_image", "driven_video"))
        _CLIP["guide_creative"] = has_driven and _CLIP["mode"] != "transfer"
        strength, axes = cls.source_axes(kwargs.get("direction_context"))
        kwargs["driven_strength"] = strength
        has_source = has_driven or any(kwargs.get(f"source_video_{i}") is not None for i in (1, 2))
        has_any_ref = any(kwargs.get(f"reference_image_{i}") is not None for i in range(1, 5))
        _CLIP["axes_note"] = axes if (has_source and has_any_ref) else ""
        if not has_any_ref:  # Keep Reference has no picture to keep from: only new designs mask the source
            _CLIP["mask"].update(setting=_CLIP["environment_new"] or _era_old, wardrobe=_CLIP["wardrobe_new"] or _era_old,
                                 hair=("control" if re.search(r"(?im)^Hair & Makeup:\s*Create New", _ctx) else False),
                                 body=_CLIP["body_new"])
        request = str(kwargs.get("edit_request") or "").strip()
        if (not request or request == cls.PLACEHOLDER_REQUEST) and not kwargs.get("bypass"):  # never an error
            kwargs["edit_request"] = cls.empty_request_brief(
                kwargs.get("edit_mode"), has_driven,
                any(kwargs.get(f"reference_image_{i}") is not None for i in range(1, 5)),
                kwargs.get("direction_context"), has_source)
        items = cls.control_items(kwargs.get("direction_context"))
        if items and not kwargs.get("bypass"):  # the user's own selections, stated in the request itself
            kwargs["edit_request"] = (str(kwargs.get("edit_request") or "").rstrip() + "\n\nSELECTED CONTROLS (apply each "
                                      "visibly; where the user's own words above say otherwise, the words win): "
                                      + "; ".join(items) + ".")
        if has_driven and str(kwargs.get("edit_mode", "")).strip().lower() in ("elaborate", "elaborate plus"):
            has_refs = any(kwargs.get(f"reference_image_{i}") is not None for i in range(1, 5))
            kwargs["system_prompt"] = (str(kwargs.get("system_prompt") or "") + "\n\n"
                                       + cls.driven_note(kwargs.get("driven_strength", 3), has_refs))
        if str(kwargs.get("edit_mode", "")).strip().lower() == "transfer":
            try:  # transfer_cast + object scenes need room, like the creative modes
                scenes = max(1, min(12, int(kwargs.get("continuous_scene_count", 3))))
                kwargs["max_tokens"] = max(int(kwargs.get("max_tokens", 1200)), scenes * (400 if _CLIP.get("basic") else 650 if _CLIP.get("keep_audio") else 800) + 800)
            except (TypeError, ValueError):
                pass
        patched = cls._get_patched_execute()
        if patched is None or "controls" in _PATCH_STATE["failed"]:
            extra = cls._control_rules_text(kwargs.get("direction_context"))
            if extra:  # fallback: carry the control rules in the system prompt
                kwargs["system_prompt"] = str(kwargs.get("system_prompt") or "") + "\n" + extra
        spent = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        dispatch = _install_provider_dispatch()

        def run_once(options):
            result_ = run_attempt(options)
            try:
                _CLIP.setdefault("held_by_result", {})[id(result_)] = _CLIP.get("held_last")
            except Exception:
                pass
            return result_

        def run_attempt(options):
            _CLIP["prev_closing"] = None  # each attempt starts its chain from scratch
            real_request = dispatch._original

            def counted(llm_model, payload):  # every paid call of this execution is added to usage_stats
                try:  # Transfer director call (not the Driven analysis): audio rules
                    messages = payload.get("messages") or []
                    system = messages[0].get("content") if messages else None
                    if isinstance(system, str) and not system.startswith("IMAGE/VIDEO TO SCENE BLUEPRINT"):
                        system = cls._clean_engine_text(system)
                        payload = dict(payload, messages=[dict(messages[0], content=system), *messages[1:]])
                    if (_CLIP.get("mode") == "transfer" and isinstance(system, str)
                            and not system.startswith("IMAGE/VIDEO TO SCENE BLUEPRINT")):
                        system = cls._transfer_audio_system(system)
                        payload = dict(payload, messages=[dict(messages[0], content=system), *messages[1:]])
                    if _CLIP.get("video_edit") and isinstance(system, str):
                        system = system + "\n\n" + cls.video_edit_contract()
                        payload = dict(payload, messages=[dict(messages[0], content=system), *messages[1:]])
                    if (_CLIP.get("guide_creative") and isinstance(system, str)
                            and not system.startswith("IMAGE/VIDEO TO SCENE BLUEPRINT")):
                        look = cls.guide_look_note()
                        system = system + "\n\n" + cls.GUIDE_SILENT_NOTE + (("\n\n" + look) if look else "")
                        payload = dict(payload, messages=[dict(messages[0], content=system), *messages[1:]])
                    if (_CLIP.get("i2v") and _CLIP.get("mode") != "transfer" and isinstance(system, str)
                            and not system.startswith("IMAGE/VIDEO TO SCENE BLUEPRINT")):
                        system = system + "\n\n" + cls.I2V_OPENING_NOTE
                        payload = dict(payload, messages=[dict(messages[0], content=system), *messages[1:]])
                    if (_CLIP.get("checklist") and isinstance(system, str)
                            and not system.startswith("IMAGE/VIDEO TO SCENE BLUEPRINT")):
                        tail = _CLIP["checklist"] + (("\n" + _CLIP["extra_rule"])
                                                     if str(_CLIP.get("extra_rule", "")).startswith("CORRECTION") else "")
                        payload = dict(payload, messages=[dict(messages[0], content=system + "\n\n" + tail),
                                                          *messages[1:]])
                except Exception:
                    pass
                if isinstance(payload, dict) and "seed" in payload:  # some providers (Venice) accept only int32 seeds
                    try:
                        seed_value = int(payload["seed"])
                    except (TypeError, ValueError):
                        seed_value = 0
                    payload = dict(payload, seed=seed_value & 0x7FFFFFFF)
                def send(body):  # every network attempt counts when it is sent, answered or not
                    if spent["calls"] >= CALL_CAP:
                        raise RuntimeError(f"model call limit of {CALL_CAP} reached for this execution")
                    spent["calls"] += 1
                    return real_request(llm_model, body)
                try:
                    reply = send(payload)
                except RuntimeError as error:  # a provider that rejects the seed: same call once without it
                    if "seed" not in str(error).lower() or "seed" not in payload:
                        raise
                    print("[H3 Director] provider rejected the seed; repeating the call without it")
                    reply = send({k: v for k, v in payload.items() if k != "seed"})
                usage = reply.get("usage") if isinstance(reply, dict) else None
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    try:
                        spent[key] += int((usage or {}).get(key) or 0)
                    except (TypeError, ValueError):
                        pass
                return reply
            token = _HOOK_VAR.set(counted)
            try:
                return (
                    patched(cls, *args, **options) if patched is not None
                    else _Base.execute.__func__(cls, *args, **options)
                )
            finally:
                _HOOK_VAR.reset(token)

        retryable_types = (RuntimeError, ValueError, KeyError, TypeError, IndexError, AttributeError, NameError)
        fatal_markers = ("provider returned an error", "Connect ", "Hold", "no LLM was called", "Driven Strength must",
                         "No internal API", "api key", "API key", "requires an external", "HTTP 401", "HTTP 402",
                         "HTTP 403", "HTTP 404")
        schema_markers = ("response_format", "json_schema", "JSON response", "JSON schema", "InvalidParameter",
                          "invalid JSON")

        def relax(options, error):  # the provider choked on strict JSON: retry with the schema as text
            if any(m in str(error) for m in schema_markers) and not isinstance(options.get("llm"), _RelaxedJSONLLM):
                options = dict(options, llm=_RelaxedJSONLLM(options.get("llm")))
                print("[H3 Director] provider aborted strict JSON output; retrying with the schema as text")
            return options

        def retryable(error):  # a bad reply is retried; setup, provider and Hold messages are shown as they are
            return isinstance(error, retryable_types) and not any(m in str(error) for m in fatal_markers)

        def fallback(error):  # never an error: a basic prompt from what is known; Hold forgets the older story
            print(f"[H3 Director] model could not produce a valid reply ({type(error).__name__}); "
                  "basic fallback prompt delivered")
            try:
                from .story_director import _set_held_director_plan
                uid = kwargs.get("unique_id") or getattr(getattr(cls, "hidden", None), "unique_id", None)
                _set_held_director_plan(f"{cls.__name__}:{str(uid or 'default')}",
                                        {"cache_kind": "fallback", "note": "the last run ended in the fallback prompt"})
            except Exception:
                pass
            prompt, count = cls._fallback_prompt(kwargs)
            usage = (f"ALL {spent['calls']} calls: input {spent['prompt_tokens']} · output "
                     f"{spent['completion_tokens']} · total {spent['total_tokens']}")
            validation = (f"{_CLIP.get('mode', '')} fallback prompt · ⚠ the model could not follow the director "
                          f"format after {spent['calls']} model calls (a larger/API model is recommended) · "
                          f"{cls.VERSION_TAG} · {_PATCH_STATE['report']}" + _CLIP.get("driven_note", ""))
            return io.NodeOutput(prompt, validation, usage, count, "", str(error))

        CALL_CAP = 6  # every paid call of one execution (analysis, repairs, retries) shares this limit
        try:
            result = run_once(kwargs)
        except retryable_types as error:
            text = f"{type(error).__name__}: {error}"
            if not retryable(error):
                raise
            if spent["calls"] + 2 > CALL_CAP:  # an attempt can cost two calls (reply + JSON repair)
                return fallback(error)
            # A cut-off or malformed reply: one last attempt with a larger output budget.
            retry = relax(dict(kwargs), error)
            _budget = int(retry.get("max_tokens", 1200) or 1200)
            retry["max_tokens"] = max(_budget, min(int(_budget * 1.6), 16384))
            print(f"[H3 Director] reply unusable ({text}); retrying with max_tokens={retry['max_tokens']}")
            try:
                result = run_once(retry)
            except retryable_types as error2:
                text2 = f"{type(error2).__name__}: {error2}"
                if not retryable(error2):
                    raise
                if spent["calls"] + 2 > CALL_CAP:
                    return fallback(error2)
                final = relax(dict(retry), error2)  # last attempt: stricter output discipline, same budget
                final["system_prompt"] = str(final.get("system_prompt", "")) + (
                    "\n\nOUTPUT DISCIPLINE: reply with exactly one valid JSON object and nothing else; escape every "
                    "double quote inside strings; keep each field short enough that the whole object is complete.")
                print(f"[H3 Director] reply unusable again ({text2}); final attempt with strict output rules")
                try:
                    result = run_once(final)
                except retryable_types as error3:
                    if not retryable(error3):
                        raise
                    return fallback(error3)
        language_note = ""
        _NUDE = re.compile(r"\b(?:nude|naked|topless|bare breasts?|nipples?|areolae?|vulva|labia|penis|scrotum|"
                           r"testicles|genitals?|unclothed|undressed)\b", re.IGNORECASE)
        ctx_text = str(kwargs.get("direction_context") or "")
        asked_nude = re.search(r"desnud|nude|naked|sin ropa|undress|topless|NSFW|sex", str(_CLIP.get("request") or ""),
                               re.IGNORECASE)

        def problems(text):
            """Every post-check at once: unreached controls, nudity against Keep Reference, prose not in English."""
            found = {}
            try:
                missing = [item for item in cls.missing_controls(kwargs.get("direction_context"), text)
                           if item.split(" ")[0] not in limits]
                if missing:
                    found["controls"] = missing
            except Exception:
                pass
            try:
                if (_CLIP.get("wardrobe_keep") and _CLIP.get("mode") == "transfer" and not asked_nude
                        and not _CLIP.get("wardrobe_follow") and not _CLIP.get("task")
                        and not _CLIP.get("reference_nude")
                        and cls.nudity_key(ctx_text) not in ("topless", "nude", "explicit")
                        and _NUDE.search(str(text))):
                    found["wardrobe"] = True
            except Exception:
                pass
            try:
                if cls._prose_not_english(text):
                    found["language"] = True
            except Exception:
                pass
            try:
                over = cls.nudity_excess(ctx_text, text)
                if over and "NSFW" not in limits_up:
                    found["nudity"] = over
            except Exception:
                pass
            try:
                exposed = cls.undescribed_exposure(text)
                if exposed:
                    found["anatomy"] = exposed
            except Exception:
                pass
            return found

        level_now = (re.search(r"(?im)^Intensity:\s*(\w+)", ctx_text) or [None, ""])[1].lower()
        for name in cls.lora_names(ctx_text):
            if level_now in ("flirt", "sensual", "heated") and cls._LORA_EXPLICIT.search(name):
                language_note += f" · ⚠ LoRA {name} limited by NSFW {level_now.capitalize()}"
        limits = cls.request_limits(_CLIP.get("request"))
        for name in sorted(limits):
            if re.search(rf"(?im)^{'(?:Intensity|Nudity)' if name == 'NSFW' else 'Violence'}:\s*\S", ctx_text):
                language_note += f" · {name} level limited by the request"

        limits_up = {"NSFW"} if cls._NUDITY_REQUEST.search(str(_CLIP.get("request") or "")) else set()

        def score(found):  # hard errors (nudity against Keep Reference, prose not in English) weigh first
            return ((1 if found.get("wardrobe") else 0) + (1 if found.get("language") else 0)
                    + (1 if found.get("nudity") else 0),
                    len(found.get("controls", [])) + len(found.get("anatomy", [])))
        try:  # one combined corrective attempt for everything the first answer missed, never an error
            found = problems(result.args[0]) if not kwargs.get("bypass") and not kwargs.get("hold_prompt") else {}
            if found:
                rules = []
                if found.get("controls"):
                    rules.append("it ignored these selected Direction Controls: " + "; ".join(found["controls"])
                                 + "; the footage must visibly reach them, keeping the characters, the request and every "
                                 "other control")
                if found.get("wardrobe"):
                    rules.append("it described nudity, but Wardrobe is Keep Reference and nothing asked for nudity: every "
                                 "character wears exactly what their reference picture shows in every shot (if that "
                                 "picture shows clothes, remove the nudity; if the picture itself is nude, keep it)")
                if found.get("language"):
                    rules.append("it was not in English: write the prompt in English, keeping the dialogue lines in their "
                                 "language")
                if found.get("nudity"):
                    rules.append(f"it shows more skin than Nudity {found['nudity']} allows: keep everyone covered as "
                                 "that rule says in every shot, performing the same actions through or around the "
                                 "clothing")
                for who in found.get("anatomy", []):
                    rules.append(f"the {who}'s clothing is pulled down or open but the {who}'s genitals are never "
                                 "described: name them exactly (state, position, contact) in every shot that shows that "
                                 f"lower body, or keep the {who}'s clothing on if the action does not need it")
                fix = dict(kwargs)
                if found.get("language"):
                    fix["system_prompt"] = str(fix.get("system_prompt") or "") + "\n\n" + cls.OUTPUT_LANGUAGE_RULE
                _CLIP["extra_rule"] = "CORRECTION: the previous answer had problems: " + "; ".join(rules) + "."
                print("[H3 Director] corrective attempt for: " + ", ".join(found))
                try:
                    fixed = run_once(fix)
                finally:
                    _CLIP["extra_rule"] = ""
                after = problems(fixed.args[0])
                if score(after) < score(found):
                    result = fixed
                    if found.get("controls") and len(after.get("controls", [])) < len(found["controls"]):
                        language_note += " · controls corrected"
                    if found.get("wardrobe") and not after.get("wardrobe"):
                        language_note += " · wardrobe corrected"
                    if found.get("language") and not after.get("language"):
                        language_note += " · prompt language corrected"
                    if found.get("nudity") and not after.get("nudity"):
                        language_note += " · nudity corrected"
                    if found.get("anatomy") and not after.get("anatomy"):
                        language_note += " · anatomy corrected"
                    final = after
                else:
                    final = found
                if final.get("controls"):
                    language_note += " · ⚠ not reached: " + ", ".join(s.split(" (")[0] for s in final["controls"])
                if final.get("wardrobe"):
                    language_note += " · ⚠ nudity despite Wardrobe Keep Reference"
                if final.get("language"):
                    language_note += " · ⚠ prompt prose not in English"
                if final.get("nudity"):
                    language_note += f" · ⚠ more skin than Nudity {final['nudity']}"
                if final.get("anatomy"):
                    language_note += " · ⚠ exposed but undescribed: " + ", ".join(final["anatomy"])
        except Exception:
            _CLIP["extra_rule"] = ""
        try:  # final safety pass on the shot markers of the delivered prompts (Bypass returns the text untouched)
            if kwargs.get("bypass"):
                raise StopIteration
            values = list(result.args)
            values[0] = cls._finalize_prompts(values[0], clip)
            preview, count = cls._preview_of(values[0])
            if _CLIP.get("route_note") and isinstance(values[1], str):
                values[1] += _CLIP["route_note"]
            if _CLIP.get("video_edit") and isinstance(values[1], str):
                _et = _CLIP.get("edit_task") or "swap"
                values[1] = re.sub(r"^Continuous Edit continuous_edit_sequence",
                                   f"Edit (video editing of <Video 1>) · Video Task {_et.capitalize()} · Viggle LoRA: "
                                   f"{cls._EDIT_LORA.get(_et, 'swap')}", values[1])
            if _CLIP.get("random_picks"):
                values[1] = f"{values[1]} · random picks: {' / '.join(_CLIP['random_picks'])}"
            if _CLIP.get("random_ignored"):
                values[1] = f"{values[1]} · Random is not used in Transfer (the guide video keeps its design)"
            values[3] = count  # the real number of prompts delivered
            result.args = tuple(values)
            try:  # the preview shows exactly what H3 receives
                from comfy_api.latest import ui as _ui
                result.ui = _ui.PreviewText(preview)
            except Exception:
                pass
            held = (_CLIP.get("held_by_result") or {}).get(id(result))
            if held and not kwargs.get("hold_prompt"):  # Hold keeps the delivered prompt, never a rejected attempt
                from .story_director import _set_held_director_plan
                record = dict(held[1], edit_prompt=values[0], preview_prompt=preview)
                record["validation"] = str(values[1])
                _set_held_director_plan(held[0], record)
        except Exception:
            pass
        try:  # the relaxed-JSON wrapper is not the provider: report the real one
            values = list(result.args)
            values[2] = str(values[2]).replace("provider: _RelaxedJSONLLM",
                                               f"provider: {type(kwargs.get('llm')).__name__}")
            result.args = tuple(values)
        except Exception:
            pass
        try:  # usage_stats covers analysis, repairs, retries and corrective attempts
            if spent["calls"] > 1:
                values = list(result.args)
                values[2] = (f"{values[2]} · ALL {spent['calls']} calls: input {spent['prompt_tokens']} · output "
                             f"{spent['completion_tokens']} · total {spent['total_tokens']}")
                result.args = tuple(values)
        except Exception:
            pass
        try:  # visible version marker in the validation output
            values = list(result.args)
            if all(f"{section}:" in str(values[0]) for section in ("integrated_multimodal_description",
                                                                     "overall_soundscape", "non_diegetic_music")):
                values[1] = re.sub(r"(?: · )?(?:warnings: )?scene \d+ missing H3 sections:[^·]*", "", str(values[1]))
            # I2V: only scene 1 starts at Picture 1; later scenes continue the previous one (no alignment by design)
            values[1] = re.sub(r";? ?scene (?:[2-9]|1[0-2]) missing I2V alignment", "", str(values[1]))
            values[1] = re.sub(r"warnings:\s*;\s*", "warnings: ", values[1])
            values[1] = re.sub(r"(?:\s*·)?\s*warnings:\s*(?=·|$)", "", values[1]).rstrip(" ·")
            values[1] = re.sub(r"\bdriven (image|video)\b", r"guide \1", re.sub(r" · Driven Strength \d+", "", str(values[1])))
            values[1] = (f"{values[1]} · {cls.VERSION_TAG} · {_PATCH_STATE["report"]}"
                         + cls._language_warning(values[0], kwargs.get("direction_context")) + language_note
                         + _CLIP.get("driven_note", ""))
            result.args = tuple(values)
        except Exception:
            pass
        return result
