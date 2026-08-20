from __future__ import annotations

import base64
import http.client
import io as binary_io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

import av
import numpy as np
from comfy_api.latest import io, ui
from comfy_execution.graph_utils import ExecutionBlocker
from PIL import Image, ImageDraw, ImageFont


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
DEFAULT_MODEL = "x-ai/grok-4.20"
DIRECTOR_PROFILES = ["OpenRouter", "Gemma"]
DIALOGUE_OPTIONS = [
    "No dialogue",
    "English",
    "Mandarin Chinese",
    "Hindi",
    "Spanish",
    "Standard Arabic",
    "French",
    "Bengali",
    "Portuguese",
    "Russian",
    "Indonesian",
    "Japanese",
]
AUDIO_CONTENT_MODES = [
    "Auto",
    "Dialogue Only",
    "Dialogue and Music",
    "Singing Music Only",
    "Instrumental Music Only",
]

AUDIO_CONTENT_LEGACY_ALIASES = {
    "Solo diálogo": "Dialogue Only",
    "Diálogo y música": "Dialogue and Music",
    "Solo música con canto": "Singing Music Only",
    "Solo música instrumental": "Instrumental Music Only",
}
DIRECTOR_MODES = [
    "Continuous Story",
    "Cinematic Cuts",
    "Reference Edit",
    "Edit",
]
CINEMATIC_CUT_SHOT_LIBRARY = """
INTERNAL CINEMATOGRAPHY LIBRARY — SELECT, DO NOT DUMP:
- Shot scale: extreme wide establishing shot, wide shot, full shot, medium-wide shot,
  cowboy shot, medium shot, medium close-up, close-up, extreme close-up, insert shot,
  detail shot, macro shot.
- Camera height and angle: eye level, shoulder level, hip level, knee level, ground level,
  low angle, worm's-eye view, high angle, overhead or bird's-eye view, top-down, Dutch/canted
  angle, profile view, three-quarter front, three-quarter rear, rear view.
- Subject relationship and point of view: single, two-shot, group shot, over-the-shoulder,
  reverse over-the-shoulder, clean single, reaction shot, point-of-view, subjective POV,
  observer POV, cutaway, eyeline match, foreground-obstructed frame, mirror/reflection shot,
  silhouette, frame-within-a-frame, negative-space composition, deep staging.
- Lens and focus behavior: ultra-wide perspective, wide-angle environmental perspective,
  natural normal-lens perspective, portrait compression, long-lens compression, shallow depth
  of field, deep focus, split diopter, rack focus, selective focus, soft foreground bokeh.
- Camera behavior: locked-off, subtle handheld, aggressive handheld, pan, tilt, pedestal,
  lateral tracking, lead tracking, follow tracking, push-in, pull-back, dolly, slider,
  arc/orbit, crane/jib, drone reveal, whip pan, snap reframing, crash zoom, dolly zoom.
""".strip()
GENRES = [
    "Auto",
    "Cinematic Drama",
    "Action",
    "Thriller",
    "Horror",
    "Comedy",
    "Romance",
    "Science Fiction",
    "Fantasy",
    "Documentary",
    "Music Video",
    "Anime",
    "Animated Movie",
    "Adventure",
    "Crime",
    "Detective Mystery",
    "Film Noir",
    "Sitcom",
    "Slasher Horror",
    "Superhero",
    "Western",
    "Martial Arts",
    "Heist",
    "Spy / Espionage",
    "Disaster",
    "Psychological Drama",
    "Dark Comedy",
    "Musical",
    "Commercial / Advertising",
    "Product Showcase",
    "Fashion Campaign",
    "Beauty / Cosmetics",
    "Food / Beverage Commercial",
    "Luxury Brand Film",
    "TikTok / Reels Short",
    "YouTube Video",
    "Vlog",
    "Influencer Content",
    "UGC Advertisement",
    "Corporate Brand Film",
    "Sports Promo",
    "Travel Film",
    "Educational / Explainer",
    "Video Podcast",
    "Sensual Romance (Adults 18+)",
    "Erotic Drama (Adults 18+)",
    "Erotic Comedy (Adults 18+)",
    "Erotic Thriller (Adults 18+)",
    "Intimate Art Film (Adults 18+)",
    "Adult Glamour / Boudoir (Adults 18+)",
    "Glamour Creator Clip (Adults 18+)",
    "OnlyFans-Style Creator Video (Adults 18+)",
    "Pornographic Film (Adults 18+)",
    "Explicit Adult Film (Adults 18+)",
    "Explicit Music Video (Adults 18+)",
    "Amateur-Style Adult Film (Adults 18+)",
    "POV Adult Film (Adults 18+)",
    "Couples Adult Film (Adults 18+)",
    "Fetish-Themed Adult Film (Adults 18+)",
    "BDSM-Themed Adult Film (Consenting Adults 18+)",
    "Adult Fantasy Roleplay (Consenting Adults 18+)",
]
MOTION_STYLES = {
    "Auto": (
        "Infer the most appropriate subject-motion intensity, temporal rhythm, "
        "camera behavior, stability, and movement language from the user prompt, "
        "references, source video, genre, and Director Mode. Apply the inferred "
        "choice directly; never output Auto as a motion description."
    ),
    "Normal Motion": (
        "Use natural real-time movement, believable body mechanics, ordinary "
        "camera acceleration, and physically coherent secondary motion."
    ),
    "Fast Motion": (
        "Use energetic rapid action and responsive camera work in real time. "
        "Keep actions readable and physically complete; do not merely fast-forward footage."
    ),
    "Slow Motion": (
        "Stage important action as deliberate cinematic slow motion with detailed "
        "weight transfer, facial response, fabric, hair, particles, and secondary motion."
    ),
    "Time Lapse": (
        "Compress a meaningful passage of time into visible environmental and action "
        "progression. Use a stable or deliberately evolving camera and avoid normal "
        "real-time dialogue unless the story specifically requires it."
    ),
    "Stop Motion": (
        "Use a handcrafted stop-motion cadence with intentional pose increments and "
        "slightly staccato object movement while preserving identity and spatial continuity."
    ),
    "Hyperlapse": (
        "Combine accelerated time with strong camera travel through the environment, "
        "using stable visual anchors and a clearly readable destination."
    ),
    "Speed Ramp": (
        "Motivate transitions between real time, brief acceleration, and slow motion "
        "around important actions; keep timing changes smooth and narratively purposeful."
    ),
    "Minimal Motion": (
        "Favor restrained movement: breathing, eye lines, small gestures, subtle fabric "
        "and environmental motion, and a mostly controlled camera."
    ),
    "Fluid Motion": (
        "Favor graceful continuous body mechanics and smooth camera paths with clean arcs, "
        "stable screen direction, and flowing secondary motion."
    ),
    "Intense Dynamic Motion": (
        "Use forceful full-body action, pronounced depth changes, strong parallax, and "
        "dynamic but readable camera movement without losing subject identity."
    ),
    "Super Fast Motion": (
        "Stage extremely fast, explosive action with decisive poses, strong anticipation "
        "and follow-through, rapid environmental response, and readable motion beats. "
        "Keep the subject coherent instead of simulating simple playback acceleration."
    ),
    "Super Slow Motion": (
        "Stage the scene as extreme high-speed-camera slow motion, revealing minute facial, "
        "fabric, hair, liquid, debris, and impact details through long, graceful movement arcs."
    ),
    "Handheld Camera": (
        "Use an intentional operator-held camera with subtle breathing, responsive reframing, "
        "small natural shakes, and documentary immediacy. Keep the subject readable and avoid "
        "random jitter or unstable identity."
    ),
    "Gimbal Camera": (
        "Use stabilized gimbal movement with smooth walking or running follow shots, controlled "
        "turns, clean subject tracking, and natural parallax through the environment."
    ),
    "Steadicam Follow": (
        "Follow the subject continuously with a floating Steadicam feel, fluid height changes, "
        "precise blocking, and seamless movement through foreground and background layers."
    ),
    "Locked-Off Camera": (
        "Keep the camera firmly locked on a tripod. Create motion through subject performance, "
        "blocking, depth, lighting, and environmental activity rather than camera movement."
    ),
    "Dolly / Tracking Shot": (
        "Use a deliberate dolly or lateral tracking move with constant speed, strong parallax, "
        "stable composition, and a clearly motivated start and end frame."
    ),
    "Crane / Drone Camera": (
        "Use a broad elevated camera move that reveals geography and scale, with smooth altitude "
        "changes, controlled banking, and a clear visual relationship to the main subject."
    ),
    "Orbit Camera": (
        "Circle around the main subject on a smooth controlled arc while preserving identity, "
        "screen geography, eyelines, lighting direction, and a stable center of attention."
    ),
    "Whip Pan Energy": (
        "Use motivated rapid pans between clear visual targets, with brief directional blur and "
        "clean landings. Do not let fast camera motion obscure the essential action."
    ),
    "Crash Zoom Energy": (
        "Use purposeful rapid push-ins or pull-outs to punctuate key reactions and actions, "
        "settling into a readable composition after every energetic camera move."
    ),
    "Chaotic Erratic Motion": (
        "Create intentionally erratic, unpredictable motion with abrupt direction changes, "
        "uneven rhythms, sudden reframing, and volatile environmental reactions. Preserve "
        "subject identity, anatomy, and the legibility of the central action beneath the chaos."
    ),
    "Frenetic Kinetic Motion": (
        "Use relentless kinetic energy, rapid blocking, aggressive camera pursuit, layered "
        "foreground crossings, and frequent changes in depth and direction. Keep every major "
        "action beat visually traceable despite the intensity."
    ),
    "Unhinged Handheld Camera": (
        "Use wild operator-driven handheld movement, rough pursuit, sudden tilts, imperfect "
        "recovery, snap reframing, and visceral close proximity. Make it feel intentional and "
        "raw without producing random digital jitter or identity distortion."
    ),
    "Surreal Unpredictable Motion": (
        "Use dream-logic motion with unexpected changes in direction, scale, gravity, spatial "
        "relationships, or temporal rhythm. Maintain a coherent subject and a readable visual "
        "idea even when physical behavior becomes strange."
    ),
    "Pulsing Rhythmic Motion": (
        "Synchronize body movement, camera accents, environmental reactions, and editing-like "
        "visual beats to a strong repeating pulse, alternating controlled pauses with energetic bursts."
    ),
    "Meditative Calm Motion": (
        "Use exceptionally calm pacing, patient observation, gentle breathing, minimal gestures, "
        "soft environmental movement, and long stable compositions with no unnecessary camera activity."
    ),
    "Gentle Organic Motion": (
        "Favor small natural movements, relaxed posture shifts, subtle expressions, soft fabric "
        "and hair response, and restrained camera drift that feels human and unforced."
    ),
    "Dreamlike Floating Motion": (
        "Use slow weightless movement, smooth suspended camera drift, soft arcs, gradual parallax, "
        "and delicate secondary motion, creating a tranquil dreamlike sense of space."
    ),
    "Static Tableau": (
        "Compose the scene like a living photograph: keep the camera fixed and poses strongly "
        "composed, allowing only blinking, breathing, tiny expressions, and subtle ambient movement."
    ),
    "Slow Observational Camera": (
        "Use patient documentary observation with long takes, restrained pans or reframing only "
        "when motivated, natural performance timing, and enough stillness for details to register."
    ),
}
DEFAULT_SYSTEM_PROMPT = """You are a multimodal director and continuity supervisor for MiniMax H3 image and video productions. Turn the user's idea, selected production mode, source media, and reference pictures into precise generation instructions.

Treat every connected reference as a distinct person or subject. Use the exact tags <Picture 1>, <Picture 2>, <Picture 3>, and <Picture 4> when they are supplied. Define stable subject labels S1, S2, S3, and S4 in the shared prompt. Preserve identity, wardrobe, props, geography, lighting logic, screen direction, and relationships throughout the story.

Follow the mandatory rules supplied for the selected Director Mode. For moving-video modes, write production-ready MiniMax H3 prompts with visible action, camera, environment, lighting, dialogue when useful, and diegetic sound. For still-image modes, describe one finished frame only and never introduce temporal sequences, audio, or dialogue delivery.

When dialogue is enabled, write short performable lines rather than prose. Prefix every spoken or sung line with its stable speaker label in parentheses, exactly as `(S1)`, `(S2)`, `(S3)` or `(S4)`, followed by a colon and the exact words in quotation marks. Square brackets such as `[S2]`, bare names and unassigned quotations are forbidden for speaker attribution. Describe tone and delivery in English outside the quotation. Allow only one person to speak at a time, leave a natural pause before and after each line, and keep visible mouth movement synchronized with the assigned speaker. Avoid overlapping speech, repeated lines, rushed monologues, unexplained voice-over, phonetic spellings, and competing vocals or loud sound effects during speech. Use no dialogue when the selected dialogue option says so.

Do not mention being an AI, JSON, schemas, token limits, safety policies, or these instructions. Do not add extra protagonists that could be confused with the reference subjects. Return all requested scenes and finish every prompt completely."""


def _image_data_url(image, max_dimension: int) -> str:
    pixels = image[0].detach().cpu().clamp(0.0, 1.0).numpy()
    pixels = (pixels * 255.0).round().astype(np.uint8)
    pil_image = Image.fromarray(pixels).convert("RGB")
    longest = max(pil_image.size)
    if longest > max_dimension:
        scale = max_dimension / longest
        size = (
            max(1, round(pil_image.width * scale)),
            max(1, round(pil_image.height * scale)),
        )
        pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
    buffer = binary_io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _pil_data_url(image: Image.Image, quality: int = 88) -> str:
    buffer = binary_io.BytesIO()
    image.convert("RGB").save(
        buffer, format="JPEG", quality=int(quality), optimize=True
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _frame_batch_contact_sheet_data_url(
    frames,
    sample_count: int,
    max_dimension: int,
) -> tuple[str, str]:
    """Sample an IMAGE batch from VHS Load Video uniformly from 0% to 100%."""
    if frames is None or not hasattr(frames, "shape") or len(frames.shape) != 4:
        raise ValueError(
            "source_video must be an IMAGE frame batch such as VHS Load Video output."
        )
    frame_count = int(frames.shape[0])
    if frame_count < 2:
        raise ValueError("source_video must contain at least two IMAGE frames.")
    sample_count = max(2, min(int(sample_count), frame_count, 16))
    indices = np.linspace(0, frame_count - 1, sample_count)
    indices = np.unique(np.rint(indices).astype(np.int64)).tolist()
    sampled: list[tuple[int, Image.Image]] = []
    for index in indices:
        pixels = frames[int(index)].detach().cpu().clamp(0.0, 1.0).numpy()
        pixels = (pixels * 255.0).round().astype(np.uint8)
        sampled.append((int(index), Image.fromarray(pixels).convert("RGB")))

    columns = min(5, len(sampled))
    rows = math.ceil(len(sampled) / columns)
    gap = 10
    label_height = 30
    sheet_limit = max(768, min(3072, int(max_dimension) * 2))
    cell_width = max(180, (sheet_limit - gap * (columns + 1)) // columns)
    aspect = sampled[0][1].height / max(1, sampled[0][1].width)
    cell_height = max(120, round(cell_width * aspect))
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + gap * (columns + 1),
            rows * (cell_height + label_height) + gap * (rows + 1),
        ),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    for sample_index, (frame_index, frame) in enumerate(sampled):
        row, column = divmod(sample_index, columns)
        x = gap + column * (cell_width + gap)
        y = gap + row * (cell_height + label_height + gap)
        scale = min(cell_width / frame.width, cell_height / frame.height)
        resized = frame.resize(
            (
                max(1, round(frame.width * scale)),
                max(1, round(frame.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        sheet.paste(
            resized,
            (
                x + (cell_width - resized.width) // 2,
                y + label_height + (cell_height - resized.height) // 2,
            ),
        )
        percentage = 100.0 * frame_index / max(1, frame_count - 1)
        draw.text(
            (x + 5, y + 5),
            f"{sample_index + 1:02d} · {percentage:05.1f}% · frame {frame_index}",
            fill=(255, 255, 255),
            font=font,
        )

    note = (
        f"{len(sampled)} uniformly distributed timeline samples from an IMAGE "
        f"batch containing {frame_count} source-video frames."
    )
    return _pil_data_url(sheet), note


def _frame_batch_sample_data_urls(
    frames,
    sample_count: int,
    max_dimension: int,
) -> tuple[list[tuple[int, float, str]], str]:
    """Return full-detail chronological samples from a VHS IMAGE batch."""
    if frames is None or not hasattr(frames, "shape") or len(frames.shape) != 4:
        raise ValueError(
            "source_video must be an IMAGE frame batch such as VHS Load Video output."
        )
    frame_count = int(frames.shape[0])
    if frame_count < 2:
        raise ValueError("source_video must contain at least two IMAGE frames.")
    sample_count = max(2, min(int(sample_count), frame_count, 16))
    indices = np.unique(
        np.rint(np.linspace(0, frame_count - 1, sample_count)).astype(np.int64)
    ).tolist()
    samples: list[tuple[int, float, str]] = []
    for index in indices:
        pixels = frames[int(index)].detach().cpu().clamp(0.0, 1.0).numpy()
        pixels = (pixels * 255.0).round().astype(np.uint8)
        image = Image.fromarray(pixels).convert("RGB")
        limit = max(256, min(2048, int(max_dimension)))
        scale = min(1.0, limit / max(image.width, image.height))
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)),
                 max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        percentage = 100.0 * int(index) / max(1, frame_count - 1)
        samples.append((int(index), percentage, _pil_data_url(image, quality=90)))
    note = (
        f"{len(samples)} separate full-detail chronological samples from an "
        f"IMAGE batch containing {frame_count} source-video frames."
    )
    return samples, note


def _video_contact_sheet_data_url(
    video,
    sample_count: int,
    max_dimension: int,
) -> tuple[str, str]:
    """Sample a VIDEO uniformly without materializing every frame."""
    sample_count = max(2, min(16, int(sample_count)))
    source = video.get_stream_source()
    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("The connected VIDEO contains no video stream.")
        stream = container.streams.video[0]
        duration = max(0.001, float(video.get_duration()))
        frame_rate = float(video.get_frame_rate())
        active_start, _ = video.get_active_trim_window()
        last_time = max(0.0, duration - (1.0 / max(frame_rate, 1.0)))
        targets = np.linspace(0.0, last_time, sample_count)
        sampled: list[tuple[float, Image.Image]] = []

        for target in targets:
            absolute_target = float(active_start) + float(target)
            time_base = float(stream.time_base or 1 / max(frame_rate, 1.0))
            seek_timestamp = max(0, int(absolute_target / time_base))
            container.seek(
                seek_timestamp,
                stream=stream,
                backward=True,
                any_frame=False,
            )
            selected = None
            selected_time = absolute_target
            for frame in container.decode(stream):
                if frame.pts is not None:
                    selected_time = float(frame.pts * frame.time_base)
                selected = frame
                if selected_time + (0.5 / max(frame_rate, 1.0)) >= absolute_target:
                    break
            if selected is None:
                continue
            sampled.append((float(target), selected.to_image().convert("RGB")))

    if len(sampled) < 2:
        raise ValueError("Could not extract enough frames from the connected VIDEO.")

    columns = min(5, len(sampled))
    rows = math.ceil(len(sampled) / columns)
    gap = 10
    label_height = 30
    sheet_limit = max(768, min(3072, int(max_dimension) * 2))
    cell_width = max(180, (sheet_limit - gap * (columns + 1)) // columns)
    aspect = sampled[0][1].height / max(1, sampled[0][1].width)
    cell_height = max(120, round(cell_width * aspect))
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + gap * (columns + 1),
            rows * (cell_height + label_height) + gap * (rows + 1),
        ),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    for index, (timestamp, frame) in enumerate(sampled):
        row, column = divmod(index, columns)
        x = gap + column * (cell_width + gap)
        y = gap + row * (cell_height + label_height + gap)
        scale = min(cell_width / frame.width, cell_height / frame.height)
        resized = frame.resize(
            (
                max(1, round(frame.width * scale)),
                max(1, round(frame.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        sheet.paste(
            resized,
            (
                x + (cell_width - resized.width) // 2,
                y + label_height + (cell_height - resized.height) // 2,
            ),
        )
        percentage = 100.0 * timestamp / max(duration, 0.001)
        draw.text(
            (x + 5, y + 5),
            f"{index + 1:02d} · {percentage:05.1f}% · {timestamp:05.2f}s",
            fill=(255, 255, 255),
            font=font,
        )

    analysis_note = (
        f"{len(sampled)} uniformly distributed timeline samples from a "
        f"{duration:.2f}s source video at approximately {frame_rate:.3g} fps."
    )
    return _pil_data_url(sheet), analysis_note


def _story_schema(
    scene_count: int,
    include_storyboard: bool = False,
    video_reference_mode: str = "ref2va",
    director_mode: str = "Continuous Story",
    source_video_connected: bool = False,
    director_profile: str = "OpenRouter",
    scene_duration_seconds: float = 5.0,
) -> dict:
    is_edit_mode = director_mode in {"Edit", "Reference Edit"}
    is_still = is_edit_mode and not source_video_connected
    is_video_edit = is_edit_mode and source_video_connected
    shot = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Short unique snake_case scene identifier.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Complete static image generation/edit prompt for one finished frame. "
                    "No temporal sequence, camera movement, audio, or dialogue delivery."
                    if is_still else
                    "Complete production prompt for this scene, including continuity, "
                    "visible action, camera, lighting, dialogue when useful, and sound."
                ),
            },
        },
        "required": ["id", "prompt"],
        "additionalProperties": False,
    }
    if director_profile == "Gemma" and not is_still:
        minimum_beats = (
            3 if float(scene_duration_seconds) <= 5.0 else
            4 if float(scene_duration_seconds) <= 10.0 else 5
        )
        shot["properties"].pop("prompt")
        shot["required"].remove("prompt")
        shot["properties"].update({
            "opening_state": {
                "type": "string",
                "minLength": 80,
                "description": (
                    "Exact visible opening state: present subjects, wardrobe or nudity, "
                    "pose, expression, active props, environment and spatial relationships."
                ),
            },
            "action_beats": {
                "type": "array",
                "minItems": minimum_beats,
                "maxItems": 6,
                "items": {"type": "string", "minLength": 45},
                "description": (
                    "Chronological, physically achievable visible beats. Cover every explicit "
                    "requested action or transformation instead of substituting atmosphere."
                ),
            },
            "physical_performance": {
                "type": "string",
                "minLength": 80,
                "description": (
                    "Body mechanics, hand and object paths, weight shifts, contact, gaze, "
                    "expressions and reactions that make the action readable and coherent."
                ),
            },
            "camera_plan": {
                "type": "string",
                "minLength": 60,
                "description": (
                    "Opening framing, motivated camera or focus response during the action, "
                    "and final framing. Avoid generic cinematic language."
                ),
            },
            "environment_and_lighting": {
                "type": "string",
                "minLength": 60,
                "description": (
                    "Concrete location, visible materials, practical light sources, color, "
                    "shadows and environmental reactions relevant to the action."
                ),
            },
            "sound_plan": {
                "type": "string",
                "minLength": 40,
                "description": (
                    "Specific ambience, synchronized Foley, effects and permitted music; "
                    "keep sound consistent with the selected voice and music mode."
                ),
            },
            "dialogue": {
                "type": "string",
                "description": (
                    "Exact dialogue or lyrics when enabled, using only `(S1): \"...\"` through "
                    "`(S4): \"...\"`; otherwise an empty string. Never use square brackets."
                ),
            },
            "final_state": {
                "type": "string",
                "minLength": 80,
                "description": (
                    "Unmistakable final visible state after all beats: appearance, wardrobe "
                    "or nudity, pose, contact, props, expression, location and composition."
                ),
            },
            "coverage_check": {
                "type": "string",
                "minLength": 40,
                "description": (
                    "Private concise confirmation that every explicit user action assigned "
                    "to this scene appears visibly in the beats and final state."
                ),
            },
        })
        shot["required"].extend([
            "opening_state", "action_beats", "physical_performance", "camera_plan",
            "environment_and_lighting", "sound_plan", "dialogue", "final_state",
            "coverage_check",
        ])
    if include_storyboard:
        shot["properties"]["storyboard_prompt"] = {
            "type": "string",
            "description": (
                "A single static establishing frame for this scene. Describe exact "
                "subjects, persistent wardrobe and props, environment, composition, "
                "shot size, angle, lens, focus and lighting. No motion sequence, "
                "audio, dialogue delivery, captions, labels or multiple moments."
            ),
        }
        shot["required"].append("storyboard_prompt")
    schema = {
        "type": "object",
        "properties": {
            "synopsis": {
                "type": "string",
                "description": "Concise synopsis covering the beginning, development, and ending.",
            },
            "story_bible": {
                "type": "string",
                "description": (
                    "Continuity bible defining subjects, appearance, wardrobe, props, "
                    "locations, visual language, and relationships."
                ),
            },
            "prompt_prefix": {
                "type": "string",
                "description": (
                    "Shared MiniMax H3 instructions prepended to every scene. Define "
                    "each supplied Picture tag and all permanent continuity rules. "
                    "Never include dialogue-planning rules, word limits, line limits, "
                    "or instructions about how dialogue should be written."
                ),
            },
            "shots": {
                "type": "array",
                "minItems": scene_count,
                "maxItems": scene_count,
                "items": shot,
            },
        },
        "required": ["synopsis", "story_bible", "prompt_prefix", "shots"],
        "additionalProperties": False,
    }
    if director_profile == "Gemma" and not is_still:
        schema["properties"]["required_actions"] = {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 12},
            "description": (
                "Private checklist of every explicit user-requested visible action, "
                "transformation and required final state. Do not replace items with mood."
            ),
        }
        schema["required"].append("required_actions")
    if include_storyboard:
        if video_reference_mode == "fl2va_keyframes":
            schema["properties"]["prompt_prefix"]["description"] = (
                "Shared FL2VA video-only instructions. Generated storyboard stills arrive through first_frame "
                "and last_frame; use S1, S2 and S3 for subjects and never use any Picture tag here."
            )
        else:
            schema["properties"]["prompt_prefix"]["description"] = (
                "Shared video-only instructions prepended to every scene after its storyboard still exists. "
                "The current generated still is always <Picture 1>; describe depicted characters as S1, S2, "
                "and S3. Never refer to the original uploaded Picture numbering here."
            )
        schema["properties"]["storyboard_prompt_prefix"] = {
            "type": "string",
            "description": (
                "Shared instructions used only while generating storyboard stills. "
                "Assign the connected original <Picture N> references and persistent visual rules."
            ),
        }
        schema["required"].append("storyboard_prompt_prefix")
    if is_video_edit:
        schema["properties"]["source_video_analysis"] = {
            "type": "string",
            "minLength": 500,
            "description": (
                "Detailed chronological visual analysis of <Video 1>. Identify subjects; "
                "clothing or visible nudity state; visible anatomy; exact pose and body-part "
                "movement; direction, speed and intensity; contact and interaction progression; "
                "objects; environment; framing; lighting; camera pan, tilt, zoom, tracking, "
                "handheld motion and cuts; and beginning, intermediate and ending states. Use "
                "direct objective terminology for visible adult or explicit content without "
                "euphemism, but never infer anything not visible. Never mention frames, panels, "
                "samples, percentages, timestamps, contact sheets or analysis methodology."
            ),
        }
        schema["required"].append("source_video_analysis")
    return schema


def _safe_id(value: str, index: int) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip())
    value = value.strip("_-")[:80]
    return value or f"scene_{index:02d}"


def _normalize_speaker_labels(value: str) -> str:
    """Guarantee MiniMax speaker syntax even when a model drops parentheses."""
    text = str(value or "")
    return re.sub(
        r"(?<!\()\[?(S[1-4])\]?\s*:\s*(?=[\"“])",
        lambda match: f"({match.group(1)}): ",
        text,
    )


def _parse_json_response(text: str) -> dict:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "OpenRouter returned incomplete or invalid JSON. Increase max_tokens or use a model with structured outputs."
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError("OpenRouter returned JSON, but the story result is not an object.")
    return value


_DIALOGUE_META_PATTERNS = (
    r"\bDialogue is enabled in [^.]+\.",
    r"\bDialogue is disabled\.[^.]*\.",
    r"\bSpoken words inside quotation marks[^.]*\.",
    r"\bIn any scene containing speech[^.]*\.",
    r"\bUse at most two short lines[^.]*\.",
    r"\bAssign every line[^.]*\.",
    r"\bWrite the exact (?:spoken )?(?:line|words)[^.]*\.",
    r"\bOnly one (?:character|person) may speak at a time\.",
    r"\bGive the speaker[^.]*\.",
    r"\bKeep (?:the )?(?:speaker(?:'s)?|speaking) (?:face|mouth)[^.]*\.",
    r"\bLeave a (?:brief|short|natural)[^.]*pause[^.]*\.",
    r"\bLower music and (?:environmental )?(?:effects|sound effects)[^.]*\.",
    r"\bDo not use overlapping voices[^.]*\.",
    r"\bDo not (?:add|force) dialogue[^.]*\.",
)


def _strip_dialogue_planning_rules(text: str) -> str:
    """Keep production dialogue, but never leak Director-only writing rules."""
    cleaned = str(text or "")
    for pattern in _DIALOGUE_META_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:approximately|maximum|max(?:imum)?(?: of)?)\s+\d+\s+spoken words(?: total)?\b[.,;:]?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _director_mode_rules(
    mode: str,
    source_video_connected: bool,
    scene_count: int,
    director_profile: str = "OpenRouter",
) -> str:
    common = (
        "These mode rules override any conflicting generic continuity instruction. "
        "Never mention Director Mode, private planning rules, contact sheets, sampled "
        "frames, percentages, JSON, or analysis methodology in a generation prompt.\n"
        "MANDATORY CAST INTEGRITY:\n"
        "- Maintain one persistent cast ledger across the complete plan. Each stable subject "
        "label represents exactly one physical person, never multiple copies.\n"
        "- In every scene prompt, make the intended visible principal-character count and "
        "their subject labels unambiguous. Do not invent background copies of them.\n"
        "- When a new character enters, exactly one new person enters. The entrance changes "
        "that character from absent or off-screen to present; it must not create a second "
        "instance beside an already visible version of the same character.\n"
        "- Never show the same person twice in foreground and background, at both sides of "
        "the frame, or as cloned, twinned, duplicated, merged, or repeated anatomy unless "
        "the user explicitly requests duplicates or twins.\n"
        "- A mirror or reflective surface may show only a physically consistent reflection "
        "of the existing person, never an additional independent body.\n"
        "- Successive action beats describe the same persistent person over time and must "
        "not be interpreted as multiple simultaneous instances."
    )
    if (
        director_profile == "Gemma"
        and (source_video_connected or mode in ("Continuous Story", "Cinematic Cuts"))
    ):
        common += (
            "\nMANDATORY SCENE DETAIL:\n"
            "- Turn every explicit user-requested action or transformation into visible events; "
            "never replace it with a starting pose, implied intent or generic atmosphere.\n"
            "- Give each scene a chronological opening state, several physically achievable "
            "action and reaction beats, and an unmistakable final visual state.\n"
            "- Describe useful body mechanics, hand and object paths, contact, expression, "
            "camera response and synchronized sound. Avoid padding with generic mood language.\n"
            "- If clothing, appearance, pose, props or environment must change, distinguish the "
            "reference's initial state from the requested final state and complete the change on screen."
        )
    if mode == "Cinematic Cuts":
        return f"""{common}
MANDATORY MODE — CINEMATIC CUTS:
- Every scene begins after a hard editorial cut and is a fully independent camera setup.
- Preserve identity, wardrobe, hairstyle, physical changes, relationships, important props,
  geography, time progression and narrative state across scenes.
- Treat connected <Picture N> references as authoritative for immutable subject identity.
  Their wardrobe, pose, expression, props, lighting and background describe the initial state,
  not permanent facts, unless the user explicitly asks to keep them unchanged.
- From scene 2 onward, the latest accepted story state overrides the original Picture references
  for every mutable fact. If a subject changed clothes, hair, accessories, physical appearance,
  location or props, preserve the changed state and never revert to the original reference state.
- Do not repeatedly restate an original reference's wardrobe or environment after the story has
  replaced it. Describe the current state established by the preceding scene instead.
- Do not continue the preceding camera path, framing, exact pose or body movement.
- Freely vary shot size, angle, lens, focus, camera position and camera movement.
- For every scene, deliberately choose and explicitly state: one shot scale; one camera
  height, angle or position; one relational composition; one lens/focus behavior; and one
  camera behavior. A static or locked-off camera is a valid intentional behavior.
- Change at least two of those five cinematography dimensions from the preceding scene.
  Never repeat the exact same setup in consecutive scenes unless the story explicitly
  requires a matched reverse angle or a deliberate visual pattern.
- Motivate every choice with story information, emotion, power relationship, geography,
  reveal, reaction or action readability. Do not select unusual techniques randomly.
- Use Dutch angles, extreme close-ups, macro, POV, whip pans, crash zooms, dolly zooms and
  overhead views sparingly, only when their dramatic meaning fits the moment.
- Preserve readable screen geography and eyelines across the cut. Respect screen direction
  unless crossing the axis is an intentional, clearly motivated disorientation.
- Never request a seamless transition, overlap, morph or continuous take between scenes.
- Scene prompts describe only what occurs after the new shot has already begun.

{CINEMATIC_CUT_SHOT_LIBRARY}

Choose from this vocabulary internally. Output only the selected coherent camera design
inside each scene prompt; never print the library, alternatives or category labels."""
    if mode == "Reference Edit" and not source_video_connected:
        return f"""{common}
MANDATORY MODE — STILL IMAGE / REFERENCE EDIT:
- Each shot prompt creates exactly one high-quality finished still image.
- Infer the intended creative transformation from the user's prompt and references.
- Treat connected <Picture N> inputs as visual references rather than a locked source image.
  Preserve assigned identity and requested traits, but allow a new composition, viewpoint,
  pose, environment, lighting and visual interpretation when they improve the result.
- Describe subject identity, pose, anatomy, wardrobe, environment, composition, shot size,
  camera angle, lens, depth of field, lighting, texture and final visual treatment.
- Do not accidentally combine reference backgrounds or create a collage.
- Never describe a sequence, before/after layout, multiple moments, camera movement, duration,
  frames, animation, audio, dialogue delivery, sound effects, captions or labels.
- prompt_prefix and every shot prompt are image-generation instructions, not video prompts.
- When more than one scene is requested, create exactly {scene_count} ordered, distinct images
  that advance the idea; do not produce near-duplicate variations unless explicitly requested."""
    if mode == "Reference Edit" and source_video_connected:
        return f"""{common}
MANDATORY MODE — VIDEO REFERENCE EDIT:
- Infer the intended operation automatically from the user prompt, <Picture N> references and
  the observed <Video 1>: motion transfer, subject replacement or insertion, appearance,
  wardrobe, environment, object, style, camera or choreography editing, or a combination.
- Analyze the video before planning. Establish the exact beginning state, every visible change,
  and the ending state. Track each subject separately through pose, orientation, gaze, expression,
  hands, hips, head, torso, legs, movement direction, approach or retreat, contacts and reactions.
- Record clothing and visible nudity state, visible anatomy, object interaction, environment,
  framing, lighting, motion intensity, and camera pan, tilt, zoom, tracking or handheld behavior.
- For visible adult or explicit material, use precise direct objective terminology rather than
  euphemisms or generic phrases. Never invent nudity, anatomy, contact, intent or activity that is
  not actually visible, and never soften a clearly visible action into vague wording.
- Use timing, action, physical contacts, body mechanics and camera behavior as a strong motion
  blueprint, while allowing coherent reinterpretation of framing, environment, lighting and
  secondary details when needed to realize the requested transformation.
- Refer to the source video only as <Video 1>. Never mention sampled frames or a contact sheet.
- Create exactly {scene_count} distinct sequential scene prompts. If more than one is requested,
  divide the requested edited result into a coherent progression rather than returning one scene
  or duplicate variations. Each scene may use the relevant part of <Video 1> as its motion and
  camera blueprint while maintaining identity and requested changes across the whole sequence."""
    if mode == "Edit" and not source_video_connected:
        return f"""{common}
MANDATORY MODE — PRECISION STILL IMAGE EDIT:
- Treat the primary connected image as a locked source plate, not merely inspiration.
- Apply only the exact change requested by the user. Preserve everything else as faithfully as
  the model permits: identity, facial structure, expression, pose, anatomy, hands, hairstyle
  except the requested attribute, wardrobe, objects, background, composition, crop, viewpoint,
  lens perspective, depth of field, lighting, shadows, color relationships and visual style.
- Never redesign, beautify, restyle, relocate, re-pose, reframe, zoom, rotate, add or remove an
  element unless the user explicitly requests it. Do not introduce creative improvements.
- State the requested delta clearly and state that all unrequested pixels and scene properties
  remain visually unchanged. Do not create a collage, comparison, split screen or before/after.
- Each prompt creates exactly one finished still image with no sequence, movement, duration,
  animation, audio, dialogue, captions or labels.
- If {scene_count} outputs are requested, apply the same surgical edit consistently to exactly
  {scene_count} ordered source/reference results without inventing unrelated variations."""
    if mode == "Edit" and source_video_connected:
        return f"""{common}
MANDATORY MODE — PRECISION VIDEO EDIT:
- Treat <Video 1> as a locked source plate and temporal blueprint. Apply only the exact change
  requested by the user; this is not motion transfer, restaging or a creative remake.
- Preserve the source duration, chronology, frame composition, crop, viewpoint, camera path,
  cuts, timing, speed, poses, body mechanics, gestures, expressions, contacts, occlusions,
  background, objects, wardrobe, lighting, shadows, texture and audio-visible synchronization,
  except for the single property or explicitly listed properties the user asks to change.
- Track the edited attribute consistently through every pose, angle, occlusion, lighting change
  and source cut. For example, a hair-color edit changes only hair color and preserves hairstyle,
  length, motion, face, skin, clothing, environment and camera exactly.
- Never reframe, change angles, redesign, beautify, add cinematic coverage, alter choreography,
  replace the location or introduce new action unless explicitly requested.
- Analyze the full source before planning, refer to it only as <Video 1>, and never mention
  samples, frames, contact sheets, percentages or analysis methodology in generation prompts.
- Create exactly {scene_count} sequential prompts covering the source progression. Each prompt
  must repeat the same narrow edit constraint and strict preservation requirement."""
    return f"""{common}
MANDATORY MODE — CONTINUOUS STORY:
- Scene 1 establishes the story. Every later scene continues the preceding final pose, movement,
  object state, screen direction, camera trajectory, location, lighting and sound state.
- End every non-final scene on a clear unfinished action that the next scene can continue.
- Preserve identity, wardrobe, props, geography and relationships across the sequence.
- The last scene deliberately resolves the story. Do not introduce an unresolved final action."""


def _gemma_scene_prompt(shot: dict) -> str:
    """Compile Gemma's private scene worksheet into one MiniMax-ready prompt."""
    beats = shot.get("action_beats")
    if not isinstance(beats, list) or not beats:
        return ""
    parts = [
        f"Opening state: {str(shot.get('opening_state') or '').strip()}",
        "Chronological action: " + " ".join(
            f"{index}. {str(beat).strip()}" for index, beat in enumerate(beats, 1)
            if str(beat).strip()
        ),
        f"Physical performance: {str(shot.get('physical_performance') or '').strip()}",
        f"Camera: {str(shot.get('camera_plan') or '').strip()}",
        f"Environment and lighting: {str(shot.get('environment_and_lighting') or '').strip()}",
        f"Sound: {str(shot.get('sound_plan') or '').strip()}",
    ]
    dialogue = str(shot.get("dialogue") or "").strip()
    if dialogue:
        parts.append(f"Dialogue: {dialogue}")
    parts.append(f"Final state: {str(shot.get('final_state') or '').strip()}")
    return "\n".join(part for part in parts if not part.endswith(": "))


def _compile_story(
    raw: dict,
    scene_count: int,
    duration_seconds: float,
    steps: int,
    picture_count: int,
    director_mode: str = "Continuous Story",
) -> tuple[str, str, str, str, str]:
    synopsis = str(raw.get("synopsis") or "").strip()
    story_bible = str(raw.get("story_bible") or "").strip()
    prompt_prefix = _strip_dialogue_planning_rules(raw.get("prompt_prefix") or "")
    storyboard_prompt_prefix = str(raw.get("storyboard_prompt_prefix") or "").strip()
    source_video_analysis = str(raw.get("source_video_analysis") or "").strip()
    shots = raw.get("shots")
    if not synopsis or not story_bible or not prompt_prefix:
        raise RuntimeError("The director response is missing its synopsis, story bible, or shared prompt.")
    if not isinstance(shots, list) or len(shots) != scene_count:
        actual = len(shots) if isinstance(shots, list) else 0
        raise RuntimeError(
            f"The director returned {actual} scenes, but {scene_count} were requested. The plan was not accepted."
        )

    reference_assignment_prefix = storyboard_prompt_prefix or prompt_prefix
    missing_tags = [
        f"<Picture {index}>"
        for index in range(1, picture_count + 1)
        if f"<Picture {index}>" not in reference_assignment_prefix
    ]
    repaired_reference_count = len(missing_tags)
    if missing_tags:
        fallback_assignments = " ".join(
            f"{tag} is an exact connected visual reference; use it for the "
            "subject, object, appearance, environment, or style assigned by the "
            "user request and scene, preserving every relevant visible trait."
            for tag in missing_tags
        )
        if storyboard_prompt_prefix:
            storyboard_prompt_prefix = "\n\n".join((
                storyboard_prompt_prefix, fallback_assignments
            ))
        else:
            prompt_prefix = "\n\n".join((prompt_prefix, fallback_assignments))

    compiled_shots = []
    seen_ids = set()
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            raise RuntimeError(f"Scene {index} is not a structured scene object.")
        shot_id = _safe_id(shot.get("id", ""), index)
        if shot_id in seen_ids:
            shot_id = f"{shot_id}_{index:02d}"
        seen_ids.add(shot_id)
        prompt = _normalize_speaker_labels(_strip_dialogue_planning_rules(
            shot.get("prompt") or _gemma_scene_prompt(shot)
        ))
        if len(prompt) < 80:
            raise RuntimeError(
                f"Scene {index} is too short to be a production-ready continuity prompt."
            )
        compiled_shot = {"id": shot_id, "prompt": prompt}
        storyboard_prompt = str(shot.get("storyboard_prompt") or "").strip()
        if storyboard_prompt:
            if len(storyboard_prompt) < 40:
                raise RuntimeError(
                    f"Scene {index} storyboard prompt is too short to define a stable frame."
                )
            compiled_shot["storyboard_prompt"] = storyboard_prompt
        compiled_shots.append(compiled_shot)

    plan = {
        "prompt_prefix": prompt_prefix,
        "defaults": {
            "duration_seconds": float(duration_seconds),
            "steps": int(steps),
        },
        "shots": compiled_shots,
    }
    if storyboard_prompt_prefix:
        plan["storyboard_prompt_prefix"] = storyboard_prompt_prefix
    if source_video_analysis:
        plan["source_video_analysis"] = source_video_analysis
    validation = (
        f"Valid: {director_mode} · {scene_count} scenes · {picture_count} references · "
        f"{float(duration_seconds):g}s requested per scene · {int(steps)} steps"
    )
    if repaired_reference_count:
        validation += (
            f" · repaired {repaired_reference_count} omitted Picture "
            f"assignment{'s' if repaired_reference_count != 1 else ''}"
        )
    return (
        json.dumps(plan, ensure_ascii=False, indent=2),
        story_bible,
        synopsis,
        validation,
        source_video_analysis,
    )


def _openrouter_request(api_key: str, payload: dict, timeout_seconds: int) -> dict:
    request_data = json.dumps(payload).encode("utf-8")
    retry_delays = (2.0, 6.0)
    retryable_http_codes = {408, 409, 425, 429, 500, 502, 503, 504}

    for attempt in range(len(retry_delays) + 1):
        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=request_data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/elgalardi/ComfyUI-VisionPromptAssistant",
                "X-Title": "ComfyUI H3 Story Director",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            try:
                details = json.loads(details).get("error", {}).get("message", details)
            except (json.JSONDecodeError, AttributeError):
                pass
            if error.code not in retryable_http_codes or attempt >= len(retry_delays):
                raise RuntimeError(
                    f"OpenRouter returned HTTP {error.code}: {details}"
                ) from error
            delay = retry_delays[attempt]
            retry_after = error.headers.get("Retry-After") if error.headers else None
            try:
                delay = max(delay, min(float(retry_after), 30.0))
            except (TypeError, ValueError):
                pass
            print(
                f"[H3 Story Director] OpenRouter HTTP {error.code}; "
                f"retrying {attempt + 2}/{len(retry_delays) + 1} in {delay:g}s."
            )
            time.sleep(delay)
        except (
            ConnectionResetError,
            TimeoutError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as error:
            if attempt >= len(retry_delays):
                reason = getattr(error, "reason", error)
                raise RuntimeError(
                    "OpenRouter connection failed after 3 attempts: "
                    f"{reason}"
                ) from error
            delay = retry_delays[attempt]
            print(
                f"[H3 Story Director] OpenRouter connection interrupted "
                f"({type(error).__name__}); retrying "
                f"{attempt + 2}/{len(retry_delays) + 1} in {delay:g}s."
            )
            time.sleep(delay)

    raise RuntimeError("OpenRouter request failed without a response.")


def _credits(api_key: str, timeout_seconds: int) -> str:
    request = urllib.request.Request(
        OPENROUTER_CREDITS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8")).get("data") or {}
        total = float(data["total_credits"])
        used = float(data["total_usage"])
        return f"Remaining: ${total - used:.3f}"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, urllib.error.URLError):
        return "Credits: not available"


def _chat_completions_url(base_url: str) -> str:
    """Normalize either an OpenAI base URL or a complete chat endpoint."""
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("base_url is required for H3 LLM Model (API).")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


class H3LLMModelAPIConnection:
    """Small OpenAI-compatible LLMMODEL used by the external Director."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: int,
    ):
        self.base_url = str(base_url or "").strip()
        self.model = str(model or "").strip()
        self.api_key = str(api_key or "").strip()
        self.timeout_seconds = int(timeout_seconds)

    def h3_chat_completion(self, payload: dict) -> dict:
        request_payload = dict(payload)
        request_payload["model"] = self.model
        # These fields are OpenRouter-specific and should not leak into a
        # datacenter's otherwise OpenAI-compatible endpoint.
        request_payload.pop("provider", None)
        request_payload.pop("reasoning", None)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            _chat_completions_url(self.base_url),
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM Model (API) returned HTTP {error.code}: {details}"
            ) from error
        except (json.JSONDecodeError, urllib.error.URLError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            raise RuntimeError(f"LLM Model (API) request failed: {reason}") from error


def _ollama_chat_url(server_url: str) -> str:
    url = str(server_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("server_url is required for H3 Ollama Model (Local).")
    if url.endswith("/api/chat"):
        return url
    if url.endswith("/api"):
        return f"{url}/chat"
    return f"{url}/api/chat"


def _ollama_messages(messages: list[dict]) -> list[dict]:
    """Translate OpenAI multimodal message parts to Ollama's native format."""
    converted = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({
                "role": str(message.get("role") or "user"),
                "content": content,
            })
            continue
        text_parts = []
        images = []
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(str(part.get("text") or ""))
            elif part.get("type") == "image_url":
                image_url = part.get("image_url") or {}
                data_url = (
                    image_url.get("url", "")
                    if isinstance(image_url, dict) else str(image_url)
                )
                if "," in data_url and data_url.startswith("data:image/"):
                    images.append(data_url.split(",", 1)[1])
                elif data_url:
                    raise ValueError(
                        "H3 Ollama Model expects embedded image data from the "
                        "Director, not a remote image URL."
                    )
        converted_message = {
            "role": str(message.get("role") or "user"),
            "content": "\n".join(part for part in text_parts if part),
        }
        if images:
            converted_message["images"] = images
        converted.append(converted_message)
    return converted


class H3OllamaModelConnection:
    def __init__(
        self,
        server_url: str,
        model: str,
        api_key: str,
        keep_alive: bool,
        thinking: bool,
        context_length: int,
        timeout_seconds: int,
    ):
        self.server_url = str(server_url or "").strip()
        self.model = str(model or "").strip()
        self.api_key = str(api_key or "").strip()
        self.keep_alive = bool(keep_alive)
        self.thinking = bool(thinking)
        self.context_length = int(context_length)
        self.timeout_seconds = int(timeout_seconds)

    def h3_chat_completion(self, payload: dict) -> dict:
        chat_url = _ollama_chat_url(self.server_url)
        request_payload = {
            "model": self.model,
            "messages": _ollama_messages(payload["messages"]),
            "stream": False,
            "think": self.thinking,
            "format": payload["response_format"]["json_schema"]["schema"],
            "keep_alive": -1 if self.keep_alive else 0,
            "options": {
                "num_ctx": self.context_length,
                "num_predict": int(payload.get("max_tokens", 6144)),
                "temperature": float(payload.get("temperature", 0.45)),
                "seed": int(payload.get("seed", 0)),
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            chat_url,
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {error.code}: {details}"
            ) from error
        except (json.JSONDecodeError, urllib.error.URLError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            raise RuntimeError(f"Ollama request failed: {reason}") from error

        content = str((result.get("message") or {}).get("content") or "")
        if not content.strip():
            raise RuntimeError("Ollama returned an empty response.")
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": result.get("prompt_eval_count"),
                "completion_tokens": result.get("eval_count"),
                "total_tokens": (
                    int(result.get("prompt_eval_count") or 0)
                    + int(result.get("eval_count") or 0)
                ),
            },
        }


def _external_llm_request(llm_model, payload: dict) -> dict:
    """Use either our model connection or YALLM's LLMMODEL contract."""
    if llm_model is None:
        raise ValueError(
            "Connect H3 Ollama Model (Local), H3 LLM Model (API), "
            "YALLM LLM Model (API), or YALLM LLM Provider (API)."
        )
    direct_request = getattr(llm_model, "h3_chat_completion", None)
    if callable(direct_request):
        return direct_request(payload)

    # YALLM's current LLMModel keeps its OpenAI client and selected model on
    # these attributes. Using them when present preserves max_tokens, strict
    # JSON Schema, temperature, seed, and usage reporting. We still retain the
    # public chat_completion fallback below for custom/datacenter adapters.
    yallm_client = getattr(llm_model, "_llm", None)
    yallm_model_name = getattr(llm_model, "_model", None)
    if yallm_client is not None and yallm_model_name:
        try:
            output = yallm_client.chat.completions.create(
                model=yallm_model_name,
                messages=payload["messages"],
                max_tokens=int(payload.get("max_tokens", 6144)),
                temperature=float(payload.get("temperature", 0.45)),
                seed=int(payload.get("seed", 0)),
                response_format=payload.get("response_format"),
            )
            if hasattr(output, "model_dump"):
                return output.model_dump()
            content = output.choices[0].message.content if output.choices else ""
            usage = getattr(output, "usage", None)
            return {
                "choices": [{"message": {"content": str(content or "")}}],
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else {},
            }
        except Exception as error:
            print(
                "[H3 Story Director] The connected YALLM provider did not "
                "accept strict structured-output parameters; falling back to "
                f"its public chat_completion contract ({type(error).__name__})."
            )

    chat_completion = getattr(llm_model, "chat_completion", None)
    if not callable(chat_completion):
        raise TypeError(
            "The connected LLMMODEL does not expose chat_completion()."
        )

    # YALLM intentionally exposes a minimal chat_completion contract and does
    # not accept response_format. Put the exact schema into the system message
    # so the same parser/compiler can still validate its result.
    messages = [dict(message) for message in payload["messages"]]
    schema_instruction = (
        "Return only one valid JSON object. It must match this JSON Schema "
        "exactly; do not use Markdown fences or add commentary:\n"
        + json.dumps(
            payload["response_format"]["json_schema"]["schema"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    messages[0] = dict(messages[0])
    messages[0]["content"] = (
        str(messages[0].get("content") or "").rstrip()
        + "\n\n"
        + schema_instruction
    )
    samplers = [("temperature", float(payload.get("temperature", 0.45)))]
    try:
        content = chat_completion(
            messages,
            samplers=samplers,
            seed=int(payload.get("seed", 0)),
        )
    except TypeError:
        # Permit simpler datacenter adapters that only accept messages.
        content = chat_completion(messages)
    if isinstance(content, (list, tuple)):
        content = content[0] if content else ""
    return {
        "choices": [{"message": {"content": str(content or "")}}],
        "usage": {},
    }
class H3LLMModelAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LLMModelAPI",
            display_name="H3 LLM Model (API)",
            category="text/minimax_h3",
            search_aliases=["llm model api", "yallm", "datacenter llm"],
            description=(
                "Creates an OpenAI-compatible LLMMODEL connection for "
                "H3 Story Director — LLM Model (API). Its output is also "
                "compatible with nodes that accept YALLM's LLMMODEL type."
            ),
            inputs=[
                io.String.Input(
                    "base_url",
                    default="http://127.0.0.1:8080/v1",
                    tooltip=(
                        "OpenAI-compatible base URL ending in /v1, or the full "
                        "/chat/completions endpoint."
                    ),
                ),
                io.String.Input("model", default=""),
                io.String.Input(
                    "api_key",
                    default="",
                    extra_dict={"password": True},
                    tooltip="Optional for trusted internal datacenter endpoints.",
                ),
                io.Int.Input(
                    "timeout_seconds", default=300, min=30, max=1800, advanced=True
                ),
            ],
            outputs=[io.Custom("LLMMODEL").Output("llm_model")],
        )

    @classmethod
    def execute(
        cls, base_url: str, model: str, api_key: str, timeout_seconds: int
    ) -> io.NodeOutput:
        if not str(model or "").strip():
            raise ValueError("model is required for H3 LLM Model (API).")
        connection = H3LLMModelAPIConnection(
            base_url, model, api_key, timeout_seconds
        )
        _chat_completions_url(connection.base_url)
        return io.NodeOutput(connection)


class H3OllamaModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3OllamaModel",
            display_name="H3 Ollama Model (Local)",
            category="text/minimax_h3",
            search_aliases=["ollama h3 director", "local vlm", "ollama model"],
            description=(
                "Connects a local Ollama vision model to H3 Story Director — "
                "LLM Model (API). It uses Ollama's native multimodal and JSON "
                "Schema API, offers optional internal thinking, and can unload the model from "
                "memory immediately after every plan."
            ),
            inputs=[
                io.String.Input(
                    "server_url",
                    default="http://127.0.0.1:11434",
                    tooltip=(
                        "Paste a local, LAN, remote, or hosted Ollama-compatible "
                        "address. Accepts a server root, an address ending in "
                        "/api, or the complete /api/chat endpoint."
                    ),
                ),
                io.String.Input(
                    "model",
                    default="huihui_ai/gemma-4-abliterated:12b",
                ),
                io.String.Input(
                    "api_key",
                    default="",
                    extra_dict={"password": True},
                    tooltip=(
                        "Optional Bearer token for protected remote or hosted "
                        "Ollama-compatible services. Local Ollama needs no key."
                    ),
                ),
                io.Boolean.Input(
                    "keep_alive",
                    default=False,
                    tooltip=(
                        "False unloads the model after every plan and frees VRAM "
                        "for ComfyUI. True keeps it loaded for repeated planning."
                    ),
                ),
                io.Boolean.Input(
                    "thinking",
                    default=False,
                    tooltip=(
                        "Allows supported Ollama models to reason internally before "
                        "returning the final structured plan. This may improve difficult "
                        "scenes but increases generation time and token use."
                    ),
                ),
                io.Int.Input(
                    "context_length",
                    default=32768,
                    min=8192,
                    max=65536,
                    step=1024,
                    advanced=True,
                    tooltip=(
                        "Includes the system prompt, JSON Schema, visual tokens, "
                        "and generated plan. Larger values consume more VRAM."
                    ),
                ),
                io.Int.Input(
                    "timeout_seconds", default=600, min=60, max=3600, advanced=True
                ),
            ],
            outputs=[io.Custom("LLMMODEL").Output("llm_model")],
        )

    @classmethod
    def execute(
        cls,
        server_url: str,
        model: str,
        api_key: str,
        keep_alive: bool,
        thinking: bool,
        context_length: int,
        timeout_seconds: int,
    ) -> io.NodeOutput:
        if not str(model or "").strip():
            raise ValueError("model is required for H3 Ollama Model (Local).")
        connection = H3OllamaModelConnection(
            server_url, model, api_key, keep_alive, thinking,
            context_length, timeout_seconds
        )
        _ollama_chat_url(connection.server_url)
        return io.NodeOutput(connection)


class H3StoryDirector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3StoryDirector",
            display_name="H3 Story Director",
            category="text/minimax_h3",
            search_aliases=["story planner", "h3 director", "openrouter story"],
            description=(
                "Multimodal MiniMax H3 director for continuous stories, cinematic "
                "cuts, still-image generation/editing, storyboard keyframes, and "
                "video editing or motion transfer through OpenRouter."
            ),
            inputs=[
                io.String.Input(
                    "api_key",
                    default="",
                    placeholder="sk-or-v1-...",
                    extra_dict={"password": True},
                    tooltip="OpenRouter API key. Remove it before sharing workflows.",
                ),
                io.String.Input("model", default=DEFAULT_MODEL),
                io.String.Input(
                    "story_idea",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip=(
                        "Optional premise. Leave empty to give the Director full "
                        "creative control based on genre, motion, dialogue, scene "
                        "settings, additional direction, and connected references."
                    ),
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=DEFAULT_SYSTEM_PROMPT,
                ),
                io.Combo.Input(
                    id="director_profile",
                    display_name="Director Profile",
                    options=DIRECTOR_PROFILES,
                    default="OpenRouter",
                    tooltip=(
                        "OpenRouter preserves the established compact Director schema. "
                        "Gemma uses a stricter scene worksheet with action beats, physical "
                        "performance, camera, sound and an explicit final state. The profile "
                        "does not select or connect the model."
                    ),
                ),
                io.Image.Input("image_0", optional=True),
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
                io.Image.Input("image_3", optional=True),
                io.Int.Input(
                    "scene_count",
                    default=5,
                    min=1,
                    max=32,
                    tooltip=(
                        "Use 1 for a standalone I2V shot, or more scenes for a "
                        "connected H3 sequence."
                    ),
                ),
                io.Float.Input(
                    "scene_duration_seconds",
                    default=5.0,
                    min=1.0,
                    max=15.0,
                    step=0.5,
                ),
                io.Int.Input("steps", default=6, min=1, max=100),
                io.Boolean.Input(
                    "draft_only",
                    default=True,
                    tooltip=(
                        "Recommended for the first run. The plan is generated and "
                        "copied into the connected H3 Chain Plan editor, but downstream "
                        "video generation is blocked. Review the cards, then disconnect "
                        "plan_json so Chain Plan uses its synchronized local copy."
                    ),
                ),
                io.Combo.Input(
                    "genre",
                    options=GENRES,
                    default="Auto",
                    tooltip=(
                        "Auto infers the most coherent genre, format, tone, and visual "
                        "language from story_idea, references, source video, edit "
                        "mode, and additional direction. Written intent wins "
                        "when visual clues conflict with the prompt."
                    ),
                ),
                io.Combo.Input(
                    "secondary_genre",
                    options=["None", *GENRES],
                    default="None",
                    tooltip=(
                        "Optionally blend a second genre into the primary genre. "
                        "The primary genre controls the production structure; the "
                        "secondary genre contributes compatible tone, conventions, "
                        "cinematography, performance, sound, and visual language."
                    ),
                ),
                io.Combo.Input(
                    id="language",
                    display_name="Dialogue",
                    options=DIALOGUE_OPTIONS,
                    default="English",
                    tooltip=(
                        "Language for dialogue, lyrics, narration and spoken words. "
                        "No dialogue suppresses speech. The production plan and "
                        "technical directions remain in English."
                    ),
                ),
                io.Combo.Input(
                    "motion_style",
                    options=list(MOTION_STYLES),
                    default="Auto",
                    tooltip=(
                        "Auto infers the best motion and camera language from the "
                        "prompt, references, source video, genre, and mode."
                    ),
                ),
                io.String.Input(
                    "additional_direction",
                    multiline=True,
                    default="",
                ),
                io.Int.Input("max_tokens", default=6144, min=1024, max=16384),
                io.Float.Input("temperature", default=0.45, min=0.0, max=2.0, step=0.05),
                io.Boolean.Input(
                    "reasoning",
                    default=False,
                    tooltip=(
                        "Grok 4.20 can reason before answering. Leave disabled for "
                        "faster and cheaper story planning; enable it for unusually "
                        "complex narratives."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input(
                    "image_max_dimension",
                    default=1024,
                    min=256,
                    max=2048,
                    step=64,
                    advanced=True,
                ),
                io.Int.Input(
                    "timeout_seconds", default=300, min=30, max=900, advanced=True
                ),
                io.Combo.Input(
                    "director_mode",
                    options=DIRECTOR_MODES,
                    default="Continuous Story",
                    tooltip=(
                        "Continuous Story preserves shot continuity. Cinematic Cuts "
                        "starts independent camera setups. Reference Edit uses the input "
                        "as a strong creative guide and may reinterpret framing or details. "
                        "Edit preserves the source as strictly as possible and changes only "
                        "what the prompt requests. Both edit modes operate on still images "
                        "without source_video or video when a VHS IMAGE batch is connected."
                    ),
                ),
                io.Image.Input(
                    "source_video",
                    optional=True,
                    tooltip=(
                        "Optional IMAGE frame batch from VHS Load Video. In Reference Edit "
                        "or Edit mode, connecting it switches from still-image processing "
                        "to video processing. The requested operation is inferred from the prompt."
                    ),
                ),
                io.Int.Input(
                    "video_sample_frames",
                    default=10,
                    min=4,
                    max=16,
                    advanced=True,
                    tooltip=(
                        "Number of frames sampled uniformly from the VHS IMAGE batch "
                        "and sent as separate chronological images for detailed analysis."
                    ),
                ),
                io.Combo.Input(
                    id="audio_content",
                    display_name="Voice and Music",
                    options=AUDIO_CONTENT_MODES,
                    default="Auto",
                    tooltip=(
                        "Controls the permitted voice/music content. Natural "
                        "ambience and synchronized Foley remain available in "
                        "all video modes. Dialogue selects the spoken or sung language."
                    ),
                ),
                io.Boolean.Input(
                    "bypass_director",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Skip OpenRouter completely. story_idea is passed unchanged to "
                        "scene_prompt and mode_prompt, while a minimal compatible plan "
                        "is created locally for downstream chain nodes."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("plan_json"),
                io.String.Output("story_bible"),
                io.String.Output("synopsis"),
                io.String.Output("validation"),
                io.String.Output("usage_stats"),
                io.String.Output("credits_remaining"),
                io.String.Output(
                    "scene_prompt",
                    tooltip=(
                        "Complete prompt for the first scene, with the shared "
                        "prompt prefix included. Connect directly to MiniMax H3 "
                        "I2V when scene_count is 1."
                    ),
                ),
                io.String.Output(
                    "mode_prompt",
                    tooltip=(
                        "First complete prompt adapted to the selected Director Mode. "
                        "Use this for image generation/editing, I2V, or video editing."
                    ),
                ),
                io.String.Output(
                    "source_video_analysis",
                    tooltip=(
                        "Chronological source-motion analysis produced in Video Edit "
                        "mode; empty in other modes."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        api_key: str,
        model: str,
        story_idea: str,
        system_prompt: str,
        director_profile: str,
        scene_count: int,
        scene_duration_seconds: float,
        steps: int,
        draft_only: bool,
        genre: str,
        secondary_genre: str,
        language: str,
        audio_content: str,
        motion_style: str,
        additional_direction: str,
        max_tokens: int,
        temperature: float,
        reasoning: bool,
        seed: int,
        image_max_dimension: int,
        timeout_seconds: int,
        director_mode: str,
        video_sample_frames: int,
        image_0=None,
        image_1=None,
        image_2=None,
        image_3=None,
        source_video=None,
        bypass_director: bool = False,
        llm_model=None,
    ) -> io.NodeOutput:
        if language in {"EspaÃ±ol", "EspaÃƒÂ±ol"}:
            language = "Español"
        language = str(language or "").strip() or "No dialogue"
        audio_content = str(audio_content or "Auto").strip()
        audio_content = AUDIO_CONTENT_LEGACY_ALIASES.get(audio_content, audio_content)
        if audio_content not in AUDIO_CONTENT_MODES:
            audio_content = "Auto"
        motion_style = str(motion_style or "Auto").strip()
        director_profile = str(director_profile or "OpenRouter").strip()
        if director_profile not in DIRECTOR_PROFILES:
            director_profile = "OpenRouter"
        director_mode = str(
            getattr(cls, "FORCED_DIRECTOR_MODE", "")
            or director_mode
            or "Continuous Story"
        ).strip()
        if director_mode in {
            "Still Image / Edit", "Storyboard Frames", "Video Edit / Motion Transfer"
        }:
            director_mode = "Edit"
        if director_mode not in DIRECTOR_MODES:
            director_mode = "Continuous Story"
        motion_direction = MOTION_STYLES.get(
            motion_style,
            "Apply the requested movement style consistently while preserving "
            "physical coherence and readable action.",
        )
        scene_count = int(scene_count)
        if not 1 <= scene_count <= 32:
            raise ValueError("scene_count must be between 1 and 32.")
        story_idea = str(story_idea or "").strip()
        if bool(bypass_director):
            if not story_idea:
                raise ValueError(
                    "story_idea cannot be empty while bypass_director is enabled."
                )
            direct_plan = {
                "prompt_prefix": "",
                "defaults": {
                    "duration_seconds": float(scene_duration_seconds),
                    "steps": int(steps),
                },
                "shots": [
                    {
                        "id": f"direct_scene_{index}",
                        "prompt": story_idea,
                        "duration_seconds": float(scene_duration_seconds),
                        "steps": int(steps),
                    }
                    for index in range(1, scene_count + 1)
                ],
            }
            plan_json = json.dumps(direct_plan, ensure_ascii=False, indent=2)
            validation = (
                f"Bypassed OpenRouter · direct prompt · {scene_count} "
                f"scene{'s' if scene_count != 1 else ''} · {int(steps)} steps"
            )
            return io.NodeOutput(
                plan_json,
                "Prompt Assistant bypassed.",
                story_idea,
                validation,
                "OpenRouter not used",
                "Credits unchanged",
                story_idea,
                story_idea,
                "",
                ui=ui.PreviewText(f"{validation}\n\n{story_idea}"),
            )
        uses_external_llm = llm_model is not None
        api_key = str(api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not uses_external_llm and not api_key:
            raise ValueError(
                "An OpenRouter API key is required in the node or the "
                "OPENROUTER_API_KEY environment variable."
            )
        pictures = [
            image for image in (image_0, image_1, image_2, image_3)
            if image is not None
        ]
        source_video_connected = source_video is not None
        is_edit_mode = director_mode in {"Edit", "Reference Edit"}
        is_video_edit = is_edit_mode and source_video_connected
        is_still_mode = is_edit_mode and not source_video_connected
        genre = str(genre or "Auto").strip()
        secondary_genre = str(secondary_genre or "None").strip()
        auto_genre = genre in {"Auto", "Auto — Infer from References & Prompt"}
        secondary_auto = secondary_genre == "Auto" and not auto_genre
        secondary_enabled = (
            secondary_genre not in {"", "None", "Auto"}
            and secondary_genre != genre
        )
        secondary_direction = (
            f" Secondary genre: {secondary_genre}. Blend its compatible tone, genre "
            "conventions, cinematography, performance, sound, and visual language into "
            "the primary genre without replacing the primary narrative structure."
            if secondary_enabled else ""
        )
        if secondary_auto:
            secondary_direction = (
                " Infer one complementary secondary genre distinct from the selected "
                "primary genre. Blend its compatible tone, conventions, cinematography, "
                "performance, sound, and visual language without replacing the primary "
                "narrative structure. Identify the inferred blend naturally in the "
                "synopsis and story bible; never output Auto as a genre."
            )
        genre_direction = (
            "Genre selection is AUTO. Infer one coherent genre, production format, "
            "tone, audience, and visual language from the user's written premise, "
            "additional direction, connected reference images, source-video evidence, "
            "selected Director Mode, and inferred edit intent. The user's explicit written "
            "intent has priority over ambiguous visual clues. Do not write the word Auto "
            "as the genre; apply the inferred creative direction directly and identify it "
            "naturally in the synopsis and story bible."
            if auto_genre else
            f"Genre: {genre}. Apply this genre consistently to tone, structure, "
            "performance, cinematography, sound, and visual language."
        ) + secondary_direction
        if story_idea:
            story_direction = f"User story premise:\n{story_idea}"
        elif is_still_mode:
            story_direction = (
                "No user premise was provided: FULL CREATIVE CONTROL is enabled. "
                "Invent one compelling finished image concept based on the genre, edit "
                "intent, visual references and additional direction. Do not mention "
                "that the premise was empty or that creative control was enabled."
            )
        elif is_video_edit:
            story_direction = (
                "No user premise was provided: infer the most coherent transformation "
                "from the user's direction, source video and connected references. "
                "Preserve everything not required to change. Do not mention that the "
                "premise was empty or that creative control was enabled."
            )
        else:
            reference_note = (
                "Build the story around the supplied reference subjects and give each "
                "one a clear narrative purpose."
                if pictures else
                "Invent the necessary adult characters or non-human subjects for the story."
            )
            story_direction = (
                "No user story premise was provided: FULL CREATIVE CONTROL is enabled. "
                "Invent an original, visually specific story that strongly expresses the "
                "selected genre and motion style, fits the requested scene count and scene "
                "duration, and has a clear setup, escalation or development, climax, and "
                f"deliberate ending. {reference_note} Do not mention that the premise was "
                "empty or that creative control was enabled in any generated prompt."
            )
        adult_direction = ""
        if "Adults 18+" in genre or (
            secondary_enabled and "Adults 18+" in secondary_genre
        ):
            adult_direction = (
                " This is an adults-only genre. Every depicted participant must be "
                "an explicitly consenting adult aged 18 or older. Never create sexual "
                "content involving a minor or a person whose age is ambiguous."
            )
        elif auto_genre or secondary_auto:
            adult_direction = (
                " If the inferred direction is adult or sexually explicit, every "
                "depicted participant must be an explicitly consenting adult aged 18 "
                "or older. Never infer sexual treatment for a minor or age-ambiguous person."
            )

        if is_still_mode:
            dialogue_direction = (
                "This is a still-image mode. Do not include spoken words, dialogue "
                "delivery, narration, lyrics, audio, Foley, music or sound effects in "
                "prompt_prefix or scene prompts."
            )
        elif audio_content == "Auto":
            language_rule = (
                f"Any spoken dialogue or intelligible sung lyrics must be natural, "
                f"idiomatic {language}."
                if language != "No dialogue" else
                "Do not use spoken dialogue. Singing may use a language explicitly "
                "requested by the story; otherwise prefer instrumental or non-lexical vocals."
            )
            dialogue_direction = (
                "Audio content is AUTO. Infer the most appropriate choice for each scene "
                "and the complete story from the premise, genre, action, references and mood: "
                "dialogue only, dialogue with non-vocal music, sung music, instrumental music, "
                "or intentional ambience without music. Maintain coherent musical and vocal "
                "continuity instead of changing modes randomly. "
                f"{language_rule} Write every actual spoken line or intelligible sung lyric "
                "as `(S1): \"...\"`, `(S2): \"...\"`, `(S3): \"...\"` or `(S4): \"...\"` "
                "using the matching stable speaker label; never use square brackets and never ask MiniMax to invent "
                "unspecified words. Describe the actual music, ambience, Foley and effects "
                "chosen for each scene rather than outputting the word Auto."
            )
        elif audio_content == "Instrumental Music Only":
            dialogue_direction = (
                "Use instrumental music only. Every scene must describe the actual "
                "instrumental score, its mood, instrumentation, rhythm, energy and how it "
                "evolves with the action. Do not include dialogue, narration, voice-over, "
                "singing, lyrics, chants or intelligible background voices. Natural ambience, "
                "synchronized Foley and sound effects remain available beneath the music."
            )
        elif audio_content == "Singing Music Only":
            lyric_language = (
                language if language != "No dialogue"
                else "the language explicitly implied by the story; otherwise use non-lexical vocals"
            )
            dialogue_direction = (
                "Use music with singing as the only vocal form. Do not include spoken dialogue, "
                "narration, voice-over or intelligible spoken background voices. Describe the "
                "actual song style, instrumentation, rhythm, vocal character and progression in "
                f"every scene. Lyrics must be in {lyric_language}. When intelligible lyrics are "
                "used, write the exact sung lyric in quotation marks inside the generation "
                "prompt; never ask MiniMax to invent unspecified lyrics. Natural ambience, Foley "
                "and synchronized effects may remain beneath the song."
            )
        else:
            dialogue_enabled = language != "No dialogue"
            if dialogue_enabled:
                speech_rules = (
                    f"Dialogue is in natural, idiomatic {language}; production directions remain "
                    "in English. The Director must write every actual spoken line in quotation "
                    "marks and assign it to one clearly visible speaker using exactly `(S1):`, "
                    "`(S2):`, `(S3):` or `(S4):` before the quotation. Never use square brackets "
                    "for speaker attribution. Describe tone and delivery outside the quotation. "
                    "Keep each exchange naturally performable within the scene, allow breathing "
                    "and pauses, keep the speaking face readable, and avoid overlapping or "
                    "unattributed voices. Never ask MiniMax to invent unspecified dialogue."
                )
            else:
                speech_rules = (
                    "Spoken dialogue is disabled by the Dialogue selector. Do not include quoted "
                    "speech, narration, voice-over or intelligible background voices."
                )
            if audio_content == "Dialogue Only":
                music_rules = (
                    "Do not include background music, score, singing or lyrics. Preserve only "
                    "natural ambience, synchronized Foley and sound effects around the dialogue."
                )
            else:
                music_rules = (
                    "Include a clearly described non-vocal musical score appropriate to every "
                    "scene, with instrumentation, mood and evolution. Keep it beneath dialogue "
                    "and reduce its level during speech. Do not add sung vocals or lyrics."
                )
            dialogue_direction = f"{speech_rules} {music_rules}"

        mode_rules = _director_mode_rules(
            director_mode, source_video_connected, scene_count, director_profile
        )
        motion_brief = (
            f"Visual energy / pose style: {motion_style}. Translate this into pose, "
            "composition and implied energy without describing temporal movement."
            if is_still_mode else
            f"Motion style: {motion_style}. {motion_direction}"
        )
        duration_brief = (
            "Each prompt describes exactly one finished still image."
            if is_still_mode else
            f"Each scene will be generated for approximately "
            f"{float(scene_duration_seconds):g} seconds."
        )
        content = [{
            "type": "text",
            "text": (
                f"Create exactly {scene_count} "
                f"{'production prompt' if scene_count == 1 else 'production prompts'}. "
                f"Director mode: {director_mode}. "
                f"{genre_direction} "
                f"{motion_brief} "
                "Write the synopsis, story bible, prompt prefix, and every scene "
                "prompt entirely in English. "
                f"Dialogue direction: {dialogue_direction} "
                "Keep MiniMax tags such as <Picture 1> unchanged. "
                f"{duration_brief}\n\n"
                f"{story_direction}\n\n"
                f"Additional direction:\n{str(additional_direction or '').strip()}"
                f"{adult_direction}"
            ),
        }]
        for index, image in enumerate(pictures, 1):
            content.append({
                "type": "text",
                "text": f"The next reference is <Picture {index}>. Use this exact tag.",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(image, int(image_max_dimension))
                },
            })

        video_analysis_note = ""
        if is_video_edit:
            video_samples, video_analysis_note = _frame_batch_sample_data_urls(
                source_video,
                int(video_sample_frames),
                int(image_max_dimension),
            )
            content.append({
                "type": "text",
                "text": (
                    "The following images are separate observations from the connected source "
                    "<Video 1>, ordered strictly from earliest to latest. Reconstruct one "
                    "continuous video progression rather than describing independent pictures. "
                    "Compare adjacent observations and explicitly determine what moves, in which "
                    "direction, how pose and contact change, what remains stable, and how the "
                    "camera behaves. Describe visible clothing or nudity, anatomy and adult or "
                    "explicit actions directly and objectively when present; do not censor them, "
                    "but never infer details that are not visible. In generation prompts refer "
                    "to the source only as <Video 1>, and never expose observation labels or the "
                    f"analysis method. {video_analysis_note}"
                ),
            })
            for order, (frame_index, percentage, image_url) in enumerate(
                video_samples, 1
            ):
                content.append({
                    "type": "text",
                    "text": (
                        f"Chronological observation {order}/{len(video_samples)} "
                        f"({percentage:.1f}% through the source timeline)."
                    ),
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })

        payload = {
            "model": str(model or DEFAULT_MODEL).strip(),
            "messages": [
                {
                    "role": "system",
                    "content": "\n\n".join((
                        str(system_prompt or "").strip(),
                        mode_rules,
                    )).strip(),
                },
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "reasoning": {"enabled": bool(reasoning)},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "minimax_h3_story_plan",
                    "strict": True,
                    "schema": _story_schema(
                        scene_count,
                        include_storyboard=bool(
                            getattr(cls, "STORYBOARD_MODE", False)
                        ),
                        video_reference_mode=str(
                            getattr(cls, "VIDEO_REFERENCE_MODE", "ref2va")
                        ),
                        director_mode=director_mode,
                        source_video_connected=source_video_connected,
                        director_profile=director_profile,
                        scene_duration_seconds=float(scene_duration_seconds),
                    ),
                },
            },
            "provider": {"require_parameters": True},
        }
        result = (
            _external_llm_request(llm_model, payload)
            if uses_external_llm else
            _openrouter_request(api_key, payload, int(timeout_seconds))
        )
        try:
            content_text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("OpenRouter returned an unexpected response.") from error
        raw_story = _parse_json_response(content_text)
        if is_video_edit:
            analysis = str(raw_story.get("source_video_analysis") or "").strip()
            if len(analysis) < 500:
                raise RuntimeError(
                    "The source-video analysis was too vague or incomplete. "
                    "The plan was rejected before generation; retry with more "
                    "video samples or a stronger vision model."
                )
        plan_json, story_bible, synopsis, validation, source_video_analysis = _compile_story(
            raw_story,
            scene_count,
            float(scene_duration_seconds),
            int(steps),
            len(pictures),
            director_mode,
        )
        validation += f" · profile {director_profile}"
        compiled_plan = json.loads(plan_json)
        scene_prompt = "\n\n".join((
            str(compiled_plan["prompt_prefix"]).strip(),
            str(compiled_plan["shots"][0]["prompt"]).strip(),
        ))
        if is_video_edit and "<Video 1>" not in scene_prompt:
            raise RuntimeError(
                "The Video Edit prompt did not preserve the required <Video 1> tag. "
                "The plan was rejected before generation."
            )

        usage = result.get("usage") or {}
        usage_stats = (
            f"input: {usage.get('prompt_tokens', '?')} · "
            f"output: {usage.get('completion_tokens', '?')} · "
            f"total: {usage.get('total_tokens', '?')}"
            if usage else
            "External LLM · usage not reported"
        )
        credits = (
            "External LLM · credits not available"
            if uses_external_llm else
            _credits(api_key, min(30, int(timeout_seconds)))
        )
        preview = f"{synopsis}\n\n{validation}\n\n--- PLAN JSON ---\n{plan_json}"
        plan_output = ExecutionBlocker(None) if draft_only else plan_json
        return io.NodeOutput(
            plan_output,
            story_bible,
            synopsis,
            validation,
            usage_stats,
            credits,
            scene_prompt,
            scene_prompt,
            source_video_analysis,
            ui=ui.PreviewText(preview),
        )


class H3StoryDirectorLLMAPI(H3StoryDirector):
    """H3 Director variant driven by a wired YALLM-compatible LLMMODEL."""

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3StoryDirectorLLMAPI"
        schema.display_name = "H3 Story Director — LLM Model (API)"
        schema.search_aliases = [
            "h3 external llm director", "yallm h3 director",
            "datacenter story director",
        ]
        schema.description = (
            "The same multimodal H3 Story Director, driven by a connected "
            "LLMMODEL instead of OpenRouter. Connect H3 Ollama Model (Local), "
            "H3 LLM Model (API), YALLM LLM Model (API), or YALLM LLM "
            "Provider (API)."
        )
        hidden_openrouter_inputs = {
            "api_key", "model", "reasoning", "timeout_seconds",
        }
        schema.inputs = [
            io.Custom("LLMMODEL").Input(
                "llm_model",
                tooltip=(
                    "Accepts H3 Ollama Model (Local), H3 LLM Model (API), and "
                    "YALLM-compatible LLM Model / Provider outputs."
                ),
            ),
            *[
                item for item in schema.inputs
                if item.id not in hidden_openrouter_inputs
            ],
        ]
        return schema

    @classmethod
    def execute(cls, llm_model, **kwargs) -> io.NodeOutput:
        return H3StoryDirector.execute.__func__(
            cls,
            api_key="",
            model="",
            reasoning=False,
            timeout_seconds=300,
            llm_model=llm_model,
            **kwargs,
        )


class H3StoryDirectorCleanCuts(H3StoryDirector):
    STORYBOARD_MODE = True
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3StoryDirectorCleanCuts"
        schema.display_name = "H3 Story Director — Clean Cuts"
        schema.search_aliases = [
            "h3 cut director", "clean cut story planner", "shot director"
        ]
        schema.description = (
            "Creates independent cinematic shots joined by hard cuts while "
            "preserving identity and story state through a selected Picture tag."
        )
        schema.inputs.append(io.Combo.Input(
            "cut_continuity_reference",
            options=[f"<Picture {index}>" for index in range(2, 10)],
            default="<Picture 4>",
            tooltip=(
                "Must match the continuity sheet input position on MiniMax H3 "
                "Reference to Video. The tag is used from scene 2 onward."
            ),
        ))
        return schema

    @classmethod
    def execute(cls, cut_continuity_reference: str, **kwargs) -> io.NodeOutput:
        tag = str(cut_continuity_reference or "<Picture 4>").strip()
        clean_cut_rules = f"""

MANDATORY CLEAN-CUT DIRECTION:
- Every scene is an independent cinematic shot beginning after a hard cut.
- Preserve narrative state, identity, wardrobe, hairstyle, accessories, physical
  changes, and persistent props, but never continue the previous camera path,
  composition, exact pose, or body motion.
- Freely choose a new shot size, camera position, angle, lens, depth of field,
  framing, focus strategy, and camera movement appropriate to each scene.
- Never describe a seamless transition, continuous take, or continuation of the
  preceding final frame.
- Scene 1 must not reference {tag} because no preceding scene exists.
- At the start of every scene from scene 2 onward, use the exact tag {tag} as the
  primary visual-state reference from the previously accepted scene. It is
  authoritative for all mutable facts: current wardrobe, hair, accessories,
  physical changes, props and location state. Connected original Picture tags
  remain secondary identity anchors only and must never restore an obsolete
  outfit, pose, expression, prop, background or lighting state. Treat the new
  scene as a different camera setup after a hard cut.
- For every scene, storyboard_prompt must describe exactly one static first-frame
  composition. It must preserve the same identity and persistent story state, but
  contain no temporal sequence, camera movement, audio, dialogue delivery, labels,
  captions, panel numbers, or multiple moments. Do not mention {tag} inside a
  storyboard_prompt; that tag exists only during the later video loop.
""".strip()
        kwargs["system_prompt"] = "\n\n".join((
            str(kwargs.get("system_prompt") or "").strip(),
            clean_cut_rules,
        )).strip()
        return super().execute(**kwargs)


class H3StoryDirectorStoryboardCuts(H3StoryDirector):
    STORYBOARD_MODE = True

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3StoryDirectorStoryboardCuts"
        schema.display_name = "H3 Story Director — Storyboard Cuts"
        schema.search_aliases = [
            "h3 storyboard director", "storyboard cuts", "shot sheet director"
        ]
        schema.description = (
            "Plans independent shots plus one static storyboard frame per scene. "
            "The generated sheet becomes the FL2VA first-frame source."
        )
        return schema

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        storyboard_rules = """
MANDATORY STORYBOARD-CUT DIRECTION:
- Every scene is an independent cinematic shot beginning after a hard cut.
- Preserve identity, wardrobe, hairstyle, accessories, physical changes,
  persistent props, geography and narrative state across the complete story.
- Never continue the previous camera trajectory, composition, exact pose or body
  motion. Freely choose the best new shot size, angle, lens, focus and movement.
- Never describe a seamless transition or one continuous take.
- Produce two completely separate reference vocabularies.
- storyboard_prompt_prefix is used only to generate still images. It must assign
  every connected original <Picture N> reference to stable subject labels S1,
  S2 and S3 as applicable, plus permanent visual rules.
- prompt_prefix is used only for video generation after the still exists. In it,
  <Picture 1> always means the complete generated storyboard image for the
  current scene, never an original uploaded reference. Define S1, S2 and S3 as
  the depicted subjects inside that generated image. Never mention <Picture 2>,
  <Picture 3> or any higher Picture tag in prompt_prefix.
- The normal scene prompt describes action, camera movement, dialogue and sound
  starting from its approved generated storyboard image. Refer to characters
  only as S1, S2 and S3. Do not use any <Picture N> tag inside a scene prompt.
- storyboard_prompt describes exactly one static first-frame composition for its
  scene. It may use the original Picture tags assigned by
  storyboard_prompt_prefix and must include the visible subjects, persistent wardrobe and props,
  environment, shot size, angle, lens, focus and lighting, but no temporal
  sequence, camera movement, audio, dialogue delivery, captions, panel numbers,
  labels or multiple moments.
""".strip()
        kwargs["system_prompt"] = "\n\n".join((
            str(kwargs.get("system_prompt") or "").strip(),
            storyboard_rules,
        )).strip()
        return super().execute(**kwargs)


class H3StoryDirectorStoryboardFL2VA(H3StoryDirector):
    STORYBOARD_MODE = True
    VIDEO_REFERENCE_MODE = "fl2va_keyframes"

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3StoryDirectorStoryboardFL2VA"
        schema.display_name = "H3 Story Director — FL2VA Keyframe Storyboard"
        schema.search_aliases = ["h3 fl2va storyboard", "first last frame director"]
        schema.description = (
            "Plans individual storyboard stills and FL2VA first-to-last-frame video bridges."
        )
        return schema

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        rules = """
MANDATORY FL2VA KEYFRAME STORYBOARD DIRECTION:
- Produce storyboard_prompt_prefix for still generation only. It assigns every
  connected original <Picture N> to stable subject labels S1, S2 and S3.
- prompt_prefix is video-only. Never mention any <Picture N> tag in it. Refer to
  characters only as S1, S2 and S3 and define permanent identity, wardrobe,
  environment, prop, sound and narrative rules.
- storyboard_prompt describes one high-quality static composition per scene and
  may use the original Picture tags. No movement, audio, captions or multiple moments.
- Treat the ordered storyboard images as exact sequential keyframes. storyboard_prompt
  N is the opening image for video clip N and, when N is greater than 1, also the
  destination image of video clip N-1. Design every adjacent pair together.
- Before writing each non-final scene prompt, internally classify the change to
  the next storyboard as one of: continuous action, reframing/angle change,
  major action-state change, or location/time change. Do not output this as a
  new JSON field; express the appropriate transition naturally inside prompt.
- For continuous action, preserve screen direction, body mechanics, momentum,
  camera trajectory, object contact, lighting and sound while reaching the next pose.
- For reframing or angle changes, use a motivated cinematic bridge such as a
  subject crossing close to lens, foreground occlusion, whip pan, rapid rack
  focus, camera orbit, push through a doorway, or a natural movement that lets
  the camera settle into the exact next composition. Preserve action phase and
  identity; never dissolve faces, bodies or wardrobe into the new framing.
- For major action-state changes, describe the complete intermediate mechanics:
  initiating movement, weight transfer, hand and object paths, contact, reaction,
  secondary motion, and the exact resulting pose visible in the next keyframe.
- For location or time changes, build a deliberate visual and sound transition
  through darkness, bright light wash, doorway, moving foreground object,
  reflective surface, match movement, whip pan, or other motivated concealment.
  Establish the new location only after the old view is concealed, then settle
  into the exact destination image. Persistent identity, wardrobe, physical
  changes, relationships and story-critical props must survive the transition.
- Vary shot scale and visual language across the story when dramatically useful:
  establishing wide shots, full shots, medium shots, close-ups, inserts, low or
  high angles, over-the-shoulder views and controlled moving-camera shots. Do not
  repeat the same framing unless repetition is intentional.
- Every storyboard_prompt must be a strong standalone composition, not merely a
  minor variation of the previous image. At the same time, adjacent images must
  remain bridgeable within the selected clip duration.
- Each normal scene prompt is written for FL2VA. The current scene's generated
  storyboard image will be connected as first_frame. Except for the last scene,
  the following scene's storyboard image will be connected as last_frame.
- For every non-final scene, describe one temporally coherent action or motivated
  cinematic transition that starts exactly at the current composition and reaches
  the next planned composition only near the end. Do not request unexplained hard
  cuts, teleportation, resets, morphing, duplicate subjects or early arrival.
- Include environment, lighting, camera, synchronized physical sound, ambience,
  dialogue when useful, and evolving music in every video prompt. Sound must also
  bridge location and shot changes rather than restarting accidentally.
- The final scene has only first_frame. Give it freedom to resolve the story with
  a coherent deliberate ending and no new unresolved event.
- Normal scene prompts and prompt_prefix must never use Picture tags. The keyframe
  inputs themselves carry the generated scene images.
""".strip()
        kwargs["system_prompt"] = "\n\n".join((
            str(kwargs.get("system_prompt") or "").strip(), rules,
        )).strip()
        return super().execute(**kwargs)


__all__ = [
    "H3StoryDirector",
    "H3StoryDirectorCleanCuts",
    "H3StoryDirectorStoryboardCuts",
    "H3StoryDirectorStoryboardFL2VA",
]
