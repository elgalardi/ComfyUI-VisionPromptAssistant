from __future__ import annotations

import base64
import http.client
import io as binary_io
import json
import math
import os
import re
import threading
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
_DIRECTOR_HOLD_CACHE = {}
_DIRECTOR_HOLD_LOCK = threading.Lock()
_DIRECTOR_HOLD_CACHE_LOADED = False


def _director_hold_cache_path() -> str:
    """Return a local, non-repository path for persistent held plans."""
    try:
        import folder_paths

        user_directory = folder_paths.get_user_directory()
    except Exception:
        user_directory = os.path.join(os.path.dirname(__file__), ".runtime")
    cache_directory = os.path.join(
        user_directory, "vision_prompt_assistant", "director_hold"
    )
    os.makedirs(cache_directory, exist_ok=True)
    return os.path.join(cache_directory, "plans.json")


def _load_director_hold_cache_locked() -> None:
    global _DIRECTOR_HOLD_CACHE_LOADED
    if _DIRECTOR_HOLD_CACHE_LOADED:
        return
    _DIRECTOR_HOLD_CACHE_LOADED = True
    path = _director_hold_cache_path()
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if isinstance(saved, dict):
            _DIRECTOR_HOLD_CACHE.update(
                (str(key), value)
                for key, value in saved.items()
                if isinstance(value, dict)
            )
    except Exception as error:
        print(
            "[H3 Story Director] Could not load the persistent Hold cache: "
            f"{error}"
        )


def _save_director_hold_cache_locked() -> None:
    path = _director_hold_cache_path()
    temporary_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(_DIRECTOR_HOLD_CACHE, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    except Exception as error:
        try:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        except OSError:
            pass
        print(
            "[H3 Story Director] Could not save the persistent Hold cache: "
            f"{error}"
        )


def _get_held_director_plan(cache_key: str):
    with _DIRECTOR_HOLD_LOCK:
        _load_director_hold_cache_locked()
        held = _DIRECTOR_HOLD_CACHE.get(cache_key)
        return dict(held) if isinstance(held, dict) else None


def _set_held_director_plan(cache_key: str, record: dict) -> None:
    with _DIRECTOR_HOLD_LOCK:
        _load_director_hold_cache_locked()
        _DIRECTOR_HOLD_CACHE[cache_key] = dict(record)
        _save_director_hold_cache_locked()
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
    "Image to Video",
    "Reference Edit",
    "Edit",
]
VISUAL_LOOKS = {
    "Auto": (
        "Infer the most suitable capture aesthetic from the written request, genre, "
        "references and production mode. Apply it concretely; never output Auto."
    ),
    "Cinematic": (
        "Use deliberate feature-film composition, controlled lighting, motivated lens and "
        "focus choices, polished color, intentional camera movement and refined production design."
    ),
    "Naturalistic": (
        "Use believable available light, restrained grading, ordinary lens behavior, natural "
        "performance and unobtrusive camera operation without artificial glamour."
    ),
    "Raw / Unpolished": (
        "Use direct observational presentation, practical lighting, ordinary locations, visible "
        "imperfections and minimally choreographed operation. Avoid cinematic polish."
    ),
    "Amateur Home Video": (
        "Use personal home-video capture, imperfect but readable framing, available light, casual "
        "operator reactions, ordinary room acoustics and no professional staging."
    ),
    "Smartphone / Cellphone": (
        "Use handheld phone capture, broad depth of field, responsive autofocus and autoexposure, "
        "minor rolling shutter, casual reframing and realistic phone compression."
    ),
    "Vertical Social Media / UGC": (
        "Use spontaneous creator-style recording, direct-to-camera presence, practical lighting, "
        "quick readable framing and authentic user-generated presentation. Preserve output dimensions."
    ),
    "Webcam / Livestream": (
        "Use a fixed consumer webcam viewpoint, broad focus, screen-lit exposure, modest dynamic "
        "range, room ambience and the immediacy of an uninterrupted live feed."
    ),
    "Consumer Camcorder": (
        "Use consumer camcorder optics, deep focus, reactive zoom and autofocus, handheld operator "
        "movement, practical color and straightforward event recording."
    ),
    "MiniDV": (
        "Use early-digital MiniDV character: crisp edges, limited highlights, mild interlaced-video "
        "feel, consumer autofocus, handheld framing and period-appropriate color response."
    ),
    "VHS / Analog Tape": (
        "Use soft analog-video detail, chroma bleed, tape grain, mild tracking noise and unstable "
        "color response while preserving the selected canvas and subject readability."
    ),
    "Super 8 Film": (
        "Use intimate small-gauge film texture, visible grain, gentle flicker, warm color drift, "
        "soft detail and tactile handheld home-movie operation."
    ),
    "16mm Film": (
        "Use textured 16mm grain, organic highlight roll-off, restrained detail, practical lighting "
        "and documentary or independent-film camera behavior."
    ),
    "Documentary": (
        "Use evidence-focused observational coverage, motivated reframing, available light, credible "
        "locations, readable geography and no beautifying intervention."
    ),
    "Found Footage": (
        "Use in-world operator footage with imperfect framing, reactive movement, occlusion, exposure "
        "correction and recording artifacts, while keeping the central action legible."
    ),
    "Reality TV": (
        "Use reactive unscripted coverage, quick reframing, practical interiors, compressed telephoto "
        "observation, candid reactions and broadcast-reality immediacy."
    ),
    "CCTV / Security Camera": (
        "Use a fixed high-corner surveillance viewpoint, deep focus, wide coverage, flat practical "
        "exposure, limited detail and continuous impersonal observation."
    ),
    "Bodycam": (
        "Use chest-mounted first-person capture, wide-angle distortion, movement tied to the wearer's "
        "body, abrupt occlusion, autoexposure and raw situational audio."
    ),
    "Action Camera": (
        "Use an ultra-wide action-camera view, deep focus, strong stabilization or body-mounted motion, "
        "high environmental clarity and energetic spatial movement."
    ),
    "Broadcast Television": (
        "Use clean broadcast exposure, controlled multi-purpose framing, deep readable focus, studio "
        "or location television lighting and polished but non-filmic video color."
    ),
    "Commercial / Glossy": (
        "Use pristine product-grade lighting, controlled reflections, polished surfaces, precise "
        "camera movement, clean color separation and premium advertising finish."
    ),
    "Fashion Editorial": (
        "Use deliberate fashion posing, graphic composition, directional beauty lighting, selective "
        "texture, confident lens choices and magazine-editorial visual rhythm."
    ),
    "Dreamlike / Soft Focus": (
        "Use diffused highlights, gentle soft focus, floating tonal transitions, restrained contrast "
        "and subtly unreal atmosphere while keeping identities and actions coherent."
    ),
    "Surveillance / Hidden Camera": (
        "Use an obstructed or discreet fixed viewpoint, imperfect angle, practical exposure, broad "
        "focus and detached observational behavior without cinematic staging."
    ),
    "Laptop / PC Webcam": (
        "Use a built-in laptop or desktop webcam at monitor height, fixed wide framing, screen-led "
        "facial illumination, broad focus, modest dynamic range and realistic compressed video."
    ),
    "Streamer / RGB Gaming Setup": (
        "Use a fixed streamer-camera composition with monitor and RGB practicals, desk microphone, "
        "gaming-room depth, screen spill, direct-to-camera presence and clean livestream exposure."
    ),
    "Screenlife / Desktop Capture": (
        "Present the event through a computer-screen visual language: stable webcam window, desktop "
        "or application framing, screen illumination and readable digital-interface context."
    ),
    "Video Call / Conference Webcam": (
        "Use an ordinary video-call viewpoint with centered laptop framing, automatic exposure and "
        "white balance, modest bitrate, room ambience and believable network-video character."
    ),
    "Casting / Audition Tape": (
        "Use a straightforward audition setup with a neutral practical room, fixed eye-level camera, "
        "clear full-body or medium framing, simple lighting and an unembellished performance-recording feel."
    ),
    "Adult Casting / Audition (Consenting Adults 18+)": (
        "Use a clearly consensual adults-only casting-room setup with a fixed practical camera, direct "
        "performance framing, plain production space, neutral work lights and non-cinematic recording."
    ),
    "Adult Studio Production (Consenting Adults 18+)": (
        "Use a professional consenting-adult studio setup with readable staging, controlled practical "
        "lighting, stable multi-purpose framing, clean exposure and visible production-grade image quality."
    ),
    "Creator Bedroom Camera": (
        "Use an intimate creator-operated bedroom camera, practical lamps or ring light, fixed or casually "
        "adjusted framing, authentic room detail and polished but personal online-content presentation."
    ),
    "Dashcam / Vehicle Camera": (
        "Use a fixed dashboard or windshield-mounted wide view, deep focus, vehicle vibration, changing "
        "exposure through glass, road reflections and continuous observational recording."
    ),
    "Doorbell / Fixed Home Camera": (
        "Use a fixed wide-angle residential security viewpoint, mild barrel distortion, deep focus, automatic "
        "exposure, compressed detail and an impersonal always-on recording style."
    ),
    "Behind the Scenes / Production Diary": (
        "Use candid behind-the-scenes coverage with visible production context, practical work lights, reactive "
        "handheld reframing, informal performances and documentary production-diary immediacy."
    ),
    "Podcast / Talk Show Studio": (
        "Use a clean studio-camera setup with microphones, practical set lighting, stable conversational "
        "coverage, controlled exposure and a polished podcast or talk-show visual language."
    ),
    "1940s Black-and-White Studio": (
        "Use monochrome studio-era photography, sculpted hard key light, deep shadows, restrained camera "
        "movement, formal blocking, period lenses and visible fine film grain."
    ),
    "1950s Technicolor Studio": (
        "Use saturated three-strip-Technicolor-inspired color, bright controlled studio lighting, formal "
        "classical composition, polished production design and measured heavy-camera movement."
    ),
    "1960s Mod / Pop Cinema": (
        "Use bold period color, harder frontal or graphic lighting, zoom-lens energy, playful composition, "
        "stylized production design and crisp mid-century film texture."
    ),
    "1970s Gritty New Hollywood": (
        "Use earthy color, pushed 16mm or 35mm grain, practical low-light interiors, available-light texture, "
        "imperfect zooms or handheld observation and candid naturalistic staging."
    ),
    "1980s Analog Neon Video": (
        "Use saturated neon and tungsten color, analog-video softness, blooming highlights, period practicals, "
        "smoky atmosphere and assertive music-video or broadcast-era framing."
    ),
    "1990s Cable TV / Consumer Video": (
        "Use late-analog cable or Hi8 character, direct flash or practical light, reactive consumer zoom, "
        "soft detail, date-era color and casual television or home-video composition."
    ),
    "2000s Y2K Digital Camera": (
        "Use early compact-digital or DV sharpness, clipped highlights, direct on-camera flash, cool auto white "
        "balance, modest resolution and candid turn-of-the-millennium framing."
    ),
    "2010s DSLR / YouTube": (
        "Use early creator-era DSLR video with shallow depth of field, clean but lightly compressed 1080p detail, "
        "softbox or window light, tripod framing and recognizable YouTube production polish."
    ),
    "Flat LOG / Ungraded Digital": (
        "Use a deliberately ungraded logarithmic-camera appearance with very low contrast, restrained saturation, "
        "lifted shadow information, protected highlights and broad neutral tonal latitude; do not add a finished LUT."
    ),
    "Neutral Rec.709 / Broadcast Color": (
        "Use balanced display-ready Rec.709-style contrast, neutral whites, natural skin color, controlled legal-looking "
        "saturation and clean broadcast tonal separation without a strong creative color bias."
    ),
    "Teal and Orange Blockbuster": (
        "Use a controlled complementary grade with cyan-teal shadows and environments, warm amber-orange skin and "
        "highlights, strong subject separation, polished contrast and restrained saturation outside the key palette."
    ),
    "Bleach Bypass / Silver Retention": (
        "Use a silver-retention-inspired finish with increased contrast, dense dark shadows, reduced color saturation, "
        "harder highlights, visible grain and a metallic tactile image without losing essential subject detail."
    ),
    "Cross-Processed Reversal Film": (
        "Use cross-processed reversal-film character with shifted color relationships, punchy contrast, unusual cyan, "
        "green or magenta casts, compressed tonal transitions and expressive photochemical unpredictability."
    ),
    "Film Print Emulation": (
        "Use a restrained theatrical film-print finish with rich but controlled blacks, smooth highlight roll-off, "
        "gentle color crosstalk, moderate saturation, fine grain and cohesive photochemical density."
    ),
    "Two-Strip Technicolor": (
        "Use a stylized early two-color palette dominated by warm reds, coral skin, cyan-green shadows and limited blues, "
        "with vintage studio contrast and deliberately restricted chromatic separation."
    ),
    "High-Key Pastel": (
        "Use bright soft illumination, open shadows, low-to-moderate contrast, creamy highlights and a controlled pastel "
        "palette while retaining readable edges and natural subject separation."
    ),
    "Low-Key Desaturated": (
        "Use sparse motivated light, deep shaped shadows, restrained saturation, subdued highlights and selective color "
        "accents for a tense, intimate and minimally illuminated finish."
    ),
    "Muted Earth Tones": (
        "Use restrained ochre, olive, brown, clay and weathered neutral colors, soft saturation, natural skin separation "
        "and gently compressed highlights for an organic grounded palette."
    ),
    "Warm Golden Grade": (
        "Use warm amber highlights, honeyed skin tones, gently cool neutral shadows, soft highlight bloom and a luminous "
        "golden-hour-inspired grade without turning the whole image uniformly orange."
    ),
    "Cool Moonlight Grade": (
        "Use deep blue-cyan night ambience, neutral protected skin, cool shadow separation, restrained highlights and "
        "localized motivated practical lights while preserving believable nighttime exposure."
    ),
    "Day for Night": (
        "Simulate night from daylight with lowered exposure, cool blue-biased ambience, controlled bright sky, deepened "
        "shadows and selective warm practicals; avoid an evenly blue filter over skin and highlights."
    ),
    "Warm Tungsten Interior": (
        "Use warm tungsten practicals, amber highlights, natural falloff into cooler or neutral shadows, soft skin response "
        "and believable mixed-color interior lighting."
    ),
    "Sodium Vapor Urban Night": (
        "Use dirty amber-orange street lighting, deep cyan or neutral night shadows, hard pools of practical illumination, "
        "limited color rendering and gritty urban contrast."
    ),
    "Fluorescent Institutional Green": (
        "Use cool overhead fluorescent lighting, subtle green-cyan contamination, pale highlights, flat institutional "
        "surfaces and controlled sickly color separation without destroying skin readability."
    ),
    "Neon Magenta and Cyan": (
        "Use strong motivated magenta and cyan practicals, saturated colored edge light, deep neutral separation and glossy "
        "night contrast while preventing uncontrolled color spill across every surface."
    ),
    "Monochrome High Contrast": (
        "Use pure black-and-white rendering with hard tonal separation, deep blacks, bright shaped highlights, crisp texture "
        "and expressive film grain; introduce no residual color."
    ),
    "Monochrome Soft Silver": (
        "Use nuanced black-and-white tonality with silvery midtones, open shadow detail, gentle highlight roll-off, soft skin "
        "response and fine restrained grain."
    ),
    "Sepia / Faded Archive": (
        "Use warm sepia-brown monochromatic color, faded density, softened contrast, aged print texture, restrained flicker "
        "and archival wear while keeping the image legible."
    ),
    "Faded Vintage Print": (
        "Use lifted blacks, softened highlights, reduced saturation, gentle dye fading, warm paper-like neutrals, subtle grain "
        "and an aged photochemical print character."
    ),
    "Crushed Blacks / Punchy Contrast": (
        "Use deliberate dense blacks, steep contrast, bright controlled highlights and bold color separation while preserving "
        "the essential silhouette, face and action rather than losing them in clipped shadow."
    ),
    "Lifted Blacks / Matte Grade": (
        "Use raised black levels, compressed contrast, soft highlight response, restrained saturation and a modern matte finish "
        "with enough local contrast to keep subjects dimensional."
    ),
    "Clean HDR / Modern Digital": (
        "Use high dynamic range, clean shadow detail, protected bright highlights, precise neutral color, crisp microcontrast "
        "and polished contemporary digital-camera clarity without artificial oversharpening."
    ),
}
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
    "Body Horror",
    "Cosmic Horror",
    "Creature Feature",
    "Monster Transformation",
    "Gothic Horror",
    "Dark Fantasy",
    "Surrealist Film",
    "Cyberpunk",
    "Post-Apocalyptic",
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
    "Twitch Stream / Gaming Creator",
    "Livestream Event",
    "ASMR Creator Video",
    "Reaction Video",
    "Dance Challenge",
    "Prank / Social Experiment",
    "Livestream Shopping",
    "Machinima / Virtual Creator",
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
    "Voyeur / Hidden Camera Fantasy (Consenting Adults 18+)",
    "Adult Webcam Show (Consenting Adults 18+)",
    "Adult Livestream Creator (Consenting Adults 18+)",
    "Erotic Cosplay (Consenting Adults 18+)",
    "Adult ASMR Roleplay (Consenting Adults 18+)",
    "Explicit POV Roleplay (Consenting Adults 18+)",
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
    "SnorriCam / Body-Mounted Camera": (
        "Mount the camera rigidly to the subject so their face or torso stays fixed while the "
        "environment swings and surges around them. Keep attachment geometry and identity stable."
    ),
    "Bullet Time Arc": (
        "Suspend one decisive action in extreme slow motion while the viewpoint travels around it, "
        "revealing depth, pose, particles and environment from a coherent continuous arc."
    ),
    "Reverse Motion": (
        "Design physically legible action that unfolds backward: fragments reassemble, spills return, "
        "footsteps retract or gestures unwind, with consistent reverse causality and screen direction."
    ),
    "Freeze and Resume": (
        "Move naturally into a motivated frozen instant, hold the entire scene as a stable tableau, "
        "then resume from precisely the same pose and spatial state without morphing."
    ),
    "Seamless Loop Motion": (
        "Choreograph movement so the final pose, camera placement, lighting and environmental state "
        "naturally reconnect to the opening frame for an unobtrusive repeatable loop."
    ),
    "Dutch Roll Camera": (
        "Use controlled rolling rotation around the lens axis with motivated recovery, preserving "
        "subject readability and spatial geography while creating disorientation or instability."
    ),
    "Body-Sway POV": (
        "Use an embodied first-person camera whose steps, breathing, balance and head turns produce "
        "natural rhythmic sway, while hands and interactions remain spatially coherent."
    ),
    "Macro Probe Camera": (
        "Move an extreme close-focus camera slowly across tactile surface details, maintaining precise "
        "focus pulls, scale cues and a readable relationship to the complete subject."
    ),
    "Mechanical Robotic Motion": (
        "Use exact segmented movement, hard starts and stops, repeated calibrated arcs and machine-like "
        "timing while keeping joints, contact points and object mechanics physically consistent."
    ),
    "Creature Crawl Motion": (
        "Use low grounded locomotion with coordinated limb placement, shifting weight, gripping contact "
        "and predatory changes of direction while preserving coherent anatomy."
    ),
    "Elastic Exaggerated Motion": (
        "Use stylized anticipation, stretch, overshoot and recoil with expressive arcs and readable poses, "
        "then restore stable anatomy at the completion of every action beat."
    ),
    "Choreographed Dance Camera": (
        "Synchronize precise full-body choreography with camera travel, musical accents, formation changes "
        "and clean silhouette staging so steps and performers remain readable."
    ),
    "Parkour Pursuit": (
        "Use athletic vaults, jumps, climbs and landings with clear anticipation and impact, accompanied by "
        "a responsive pursuit camera that preserves route geography and body mechanics."
    ),
    "Intimate Micro-Motion": (
        "Concentrate on breath, eye focus, fingertips, skin contact and minute facial responses with an "
        "almost still camera, allowing subtle physical changes to carry the scene."
    ),
    "Staccato Music-Video Motion": (
        "Use sharply punctuated poses, brief bursts of movement, snap reframing and rhythmic visual accents, "
        "with clean rests between beats rather than uncontrolled jitter."
    ),
}
DEFAULT_SYSTEM_PROMPT = """You are a multimodal director and continuity supervisor for MiniMax H3 image and video productions. Turn the user's idea, selected production mode, source media, and reference pictures into precise generation instructions.

Treat each connected <Picture N> as a visual source, not automatically as one person. A Picture may contain several distinct people, and several persistent subjects may therefore originate from the same Picture. Define reusable visible identities with `<Subject 1>`, `<Subject 2>`, `<Subject 3>` and `<Subject 4>` in the shared prompt, explicitly grounding each referenced Subject in the correct `<Picture N>`. Subject and Picture numbers are independent. Use stable natural role descriptions for important characters without a connected image. Preserve identity, current wardrobe, props, geography, lighting logic, screen direction and relationships throughout the story only when the user has not explicitly replaced those mutable traits. A requested new wardrobe, world, period, environment, visual style or appearance overrides the conflicting source-image trait from the first frame.

Follow the mandatory rules supplied for the selected Director Mode. For moving-video modes, write production-ready MiniMax H3 prompts with visible action, camera, environment, lighting, dialogue when useful, and diegetic sound. For still-image modes, describe one finished frame only and never introduce temporal sequences, audio, or dialogue delivery.

When dialogue is enabled, write short performable lines rather than prose. Use the stable speaker label only for voice attribution and render every spoken or sung line in official MiniMax form: `(S1) says: <d>[English] exact words</d>`, substituting the selected language and correct speaker number. `<Subject N>` owns visual identity; `(S1)`, `(S2)`, `(S3)` and `(S4)` never replace a visual Subject or natural role. Put only the language tag and exact spoken words inside `<d>`, with tone and delivery in English outside it. Square brackets used as speaker IDs, bare speaker names, translations and duplicate quotations are forbidden. Allow only one person to speak at a time and leave a natural pause before and after each line. Throughout every line, keep the assigned speaker's face and unobstructed mouth visibly readable, with continuous natural lip, jaw and cheek articulation precisely synchronized to every spoken syllable; the voice must visibly originate from that speaker, never from a closed mouth or an off-screen source unless the user explicitly requests voice-over. Keep non-speakers' mouths still while listening. Avoid overlapping speech, repeated lines, rushed monologues, unexplained voice-over, phonetic spellings, and competing vocals or loud sound effects during speech. Use no dialogue when the selected dialogue option says so.

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


def _scene_prompt_budget(duration_seconds: float) -> tuple[int, int, int]:
    """Return a practical word range and action-beat cap for one H3 clip.

    Persistent identity, wardrobe and look belong in prompt_prefix; this budget
    is reserved for changing shot-specific information.
    """
    seconds = max(1.0, float(duration_seconds))
    if seconds <= 6.0:
        return 80, 130, 1
    if seconds <= 10.0:
        return 130, 200, 3
    return 180, 280, 4


def _story_schema(
    scene_count: int,
    include_storyboard: bool = False,
    video_reference_mode: str = "ref2va",
    director_mode: str = "Continuous Story",
    source_video_connected: bool = False,
    director_profile: str = "OpenRouter",
    scene_duration_seconds: float = 5.0,
    dialogue_language: str = "English",
    primary_genre: str = "Auto",
    secondary_genre: str = "None",
    primary_motion_style: str = "Auto",
    secondary_motion_style: str = "None",
    visual_look: str = "Auto",
    secondary_visual_look: str = "None",
    toolkit_prompt_rules: bool = False,
    split_global: bool = False,
    power_prompt_rules: bool = False,
    mood_assets_connected: bool = False,
) -> dict:
    is_edit_mode = director_mode in {"Edit", "Reference Edit"}
    is_i2v = director_mode == "Image to Video"
    is_still = is_edit_mode and not source_video_connected
    is_video_edit = is_edit_mode and source_video_connected
    is_continuous = director_mode == "Continuous Story"
    min_words, max_words, max_beats = _scene_prompt_budget(scene_duration_seconds)
    if power_prompt_rules and not is_still:
        # Power keeps a detailed private blueprint, but H3 receives concise visual
        # prose. Longer planning text does not make the diffusion prompt stronger.
        if float(scene_duration_seconds) <= 6.0:
            min_words, max_words, max_beats = 55, 105, 1
        elif float(scene_duration_seconds) <= 10.0:
            min_words, max_words, max_beats = 75, 145, 2
        else:
            min_words, max_words, max_beats = 95, 185, 3
    compact_scene_contract = (
        f" Target {min_words}-{max_words} English words for this "
        f"{float(scene_duration_seconds):g}-second scene and use no more than "
        f"{max_beats} main action beat{'s' if max_beats != 1 else ''}. "
        "Treat this as a compact chronological production brief, not literary prose. "
        "Do not repeat identities, wardrobe, location, lighting, visual look, music or "
        "continuity facts already established by prompt_prefix unless they change in this "
        "scene or are required to disambiguate an action. Spend the available detail on "
        "visible mechanics, camera changes, synchronized sound and the inherited final state."
    )
    continuous_scene_contract = (
        " This is one invisible time partition of a single unbroken take. Keep the "
        "same location, traversable route, action phase, screen direction, subject and "
        "camera velocity, rig, height, axis, lens behavior, framing distance, lighting "
        "and sound phase. Begin at the exact next instant after the preceding final "
        "state and never introduce a cut, new angle, location jump, camera reset, "
        "restarted action or invented escalation."
        if is_continuous else ""
    )
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
                    ((
                        "Complete MiniMax H3 FL2VA motion prompt describing how the supplied "
                        "first frame evolves through one shot. Treat the first frame as the literal "
                        "opening frame and an optional second image as the literal final frame. "
                        "Describe visible motion, camera evolution, environmental response and "
                        "synchronized audio without redescribing or reinterpreting the source "
                        "composition. Never emit Picture or Subject tags."
                    ) if is_i2v else (
                        "Complete production prompt for this scene, including continuity, "
                        "visible action, camera, lighting, dialogue when useful, and sound. "
                        "Use <Subject N> for persistent visible identities and reserve (S1), "
                        "(S2), (S3), and (S4) exclusively for dialogue or singing attribution. "
                        "Mention exactly the Subjects visible in this scene, omit future or absent "
                        "characters completely. Put globally persistent wardrobe/appearance "
                        "overrides in prompt_prefix and repeat only changes or details needed to "
                        "make the current action unambiguous."
                    ))
                    + compact_scene_contract
                    + continuous_scene_contract
                ),
            },
        },
        "required": ["id", "prompt"],
        "additionalProperties": False,
    }
    if not is_still and not is_i2v and director_profile != "Gemma":
        shot["properties"]["compact_state"] = {
            "type": "object",
            "properties": {
                "opening": {
                    "type": "string",
                    "minLength": 24,
                    "maxLength": 320,
                    "description": (
                        "Private compact state at the first visible instant: present cast, "
                        "current pose/contact, latest changed wardrobe or exposure, held props, "
                        "and exact location only where needed. For Continuous Story inherit the "
                        "preceding closing state without resetting completed changes. Use direct "
                        "positive visual facts, not labels, rules or explanations."
                    ),
                },
                "action": {
                    "type": "string",
                    "minLength": 40,
                    "maxLength": max(520, max_words * 7),
                    "description": (
                        "The scene's chronological visible action in compact natural English. "
                        "Name actor, physical mechanics, affected person or object, observable "
                        "result, camera behavior, lighting changes, synchronized physical sound "
                        "and any correctly attributed dialogue needed for this scene. Fit "
                        f"no more than {max_beats} main action beat"
                        f"{'s' if max_beats != 1 else ''} into the available duration."
                    ),
                },
                "closing": {
                    "type": "string",
                    "minLength": 24,
                    "maxLength": 320,
                    "description": (
                        "Private compact state after all action finishes: final pose/contact, "
                        "latest wardrobe or exposure, props, position, camera and active sound "
                        "phase only where changed. This is the authoritative opening state for "
                        "the next Continuous Story scene. End the last scene on a resolved beat."
                    ),
                },
            },
            "required": ["opening", "action", "closing"],
            "additionalProperties": False,
            "description": (
                "Compact internal continuity handoff. It is compiled into the final H3 prompt "
                "and is not exposed as planning metadata. Keep it literal and non-redundant."
            ),
        }
        shot["required"].append("compact_state")
        # The compact state is the renderable scene source. Keeping a second
        # free-form prompt makes capable models repeat the entire action and can
        # contradict the inherited closing state at a masked seam.
        shot["properties"].pop("prompt", None)
        shot["required"] = [
            name for name in shot["required"] if name != "prompt"
        ]
    if not is_still and "prompt" in shot["properties"]:
        # Character limits make the duration budget machine-readable for both
        # OpenRouter structured output and local OpenAI-compatible backends.
        shot["properties"]["prompt"]["minLength"] = max(160, min_words * 4)
        shot["properties"]["prompt"]["maxLength"] = max_words * 8
    if (toolkit_prompt_rules or split_global) and not is_still:
        shot["properties"]["visible_cast"] = {
            "type": "array",
            "minItems": 0,
            "maxItems": 12,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 3},
            "description": (
                "Exact exhaustive list of physical people visibly present in this "
                "scene, including background people. Use `<Subject N>` for every "
                "referenced identity and a short stable natural role for an invented "
                "person. List each physical individual exactly once. Do not list a "
                "Picture, body part, reflection, screen image, prop, or off-screen voice "
                "as another person. Return [] only when no person is visible."
            ),
        }
        shot["required"].append("visible_cast")
    if split_global and not is_still:
        shot["properties"]["scene_global"] = {
            "type": "string",
            "minLength": 80,
            "description": (
                "Positive scene-specific global state prepended only to this scene. "
                "Define exactly the visible Subjects and their Picture grounding, current "
                "wardrobe or exposure, current location and geography, active props, selected "
                "production format, motion treatment, visual look, lighting, camera baseline, "
                "music/ambience phase and consent/age rule when applicable. Carry completed "
                "state changes from the preceding scene and omit future people, places, props "
                "and events. Use concrete renderable facts, not planning language or negative "
                "instructions."
            ),
        }
        shot["required"].append("scene_global")
    if director_profile == "Gemma" and not is_still and not is_i2v:
        shot["properties"]["prompt"] = {
            "type": "string",
            "minLength": max(160, min_words * 4),
            "maxLength": max_words * 8,
            "description": (
                "One complete MiniMax H3 production prompt in natural English prose, "
                "with no headings, JSON keys, worksheet labels or placeholders. Begin "
                "with the exact visible opening state and cast. In visual prose, identify "
                "every referenced person naturally as `the man from <Picture 1>`, `the woman "
                "from <Picture 1>`, or the correct role and Picture number. A single Picture "
                "may contain several distinct people, and several roles may therefore use the "
                "same Picture tag; never assume character number equals Picture number. Never use bare S1, "
                "S2, S3 or S4 as visual character names; reserve `(S1)` only for spoken lines. "
                f"Continue with up to {max_beats} chronological literal visible action "
                "beats appropriate to the available screen time, including concrete body "
                "or object mechanics and the observable result of each beat. Maintain one "
                "physically coherent body configuration: name the acting subject, affected "
                "subject, exact contact, direction and resulting position; never call one act "
                "a kiss when a different contact is visible, jump between incompatible poses, "
                "or restart an action already completed by the preceding scene. Use literal "
                "anatomical and material terms when the request is explicit; do not substitute "
                "metaphors such as milk unless the user explicitly requests that substance. Include "
                "only scene-changing or action-critical camera, environment and lighting details; "
                f"execute the selected genres ({primary_genre}; {secondary_genre}), motion styles "
                f"({primary_motion_style}; {secondary_motion_style}), and visual looks "
                f"({visual_look}; {secondary_visual_look}) without restating their unchanged descriptions. Describe only "
                "synchronized sound events that occur or change in this scene. Include dialogue "
                f"only in natural {dialogue_language}, formatted `(S1) says: <d>[{dialogue_language}] exact words</d>`, without "
                "translation or repetition. End with the unmistakable final visible state "
                "that the next scene inherits. Preserve the user's semantic specificity."
                + compact_scene_contract
                + continuous_scene_contract
            ),
        }
        shot["properties"]["coverage_check"] = {
            "type": "string",
            "minLength": 60,
            "description": (
                "Private confirmation that the scene prompt contains every assigned user "
                "action, correct cast and references, opening state, appropriately limited action "
                "beats, genre/look execution, sound/dialogue rules, and final state."
            ),
        }
        shot["required"].append("coverage_check")
    if include_storyboard:
        shot["properties"]["storyboard_prompt"] = {
            "type": "string",
            "description": (
                "One decisive static first frame for this scene, representing a "
                "different chronological action phase and dramatic purpose from every "
                "adjacent panel. State the exact visible action state, expressions, "
                "subjects, current wardrobe and props, environment, shot scale, camera "
                "side, height, angle, lens character, foreground/midground/background "
                "staging, subject arrangement, eyelines, focal priority and lighting. "
                "Adjacent prompts may not describe the same pose/action with only a "
                "small crop or angle change. Across adjacent panels, change at least four "
                "of these: shot scale, camera axis, camera height, lens character, subject "
                "arrangement, and foreground-depth pattern. Build deliberate coverage with "
                "establishing geography, relationship coverage, action inserts/details, "
                "reversals and a visually strongest payoff as appropriate; never default "
                "repeatedly to centered eye-level medium two-shots. Treat clothing requested by the user "
                "as a deliberate new restyle; never claim that new clothing came from a Picture reference "
                "unless it is actually visible there. The selected genre must visibly control acting, body "
                "language, interpersonal distance, touch, gaze, emotional intensity and camera relationship, "
                "not merely color, lighting or texture. For an OnlyFans-style creator genre, direct confident "
                "seductive creator energy, charged mutual attraction and passionate physical proximity appropriate "
                "to the current story beat instead of generic cheerful posing. Describe one opening instant only. No motion sequence, audio, dialogue "
                "delivery, captions, labels or multiple moments."
            ),
        }
        shot["required"].append("storyboard_prompt")
    if power_prompt_rules and not is_still:
        shot["properties"].update({
            "scene_context": {
                "type": "string",
                "minLength": 60,
                "description": (
                    "Compact positive context prepended only to this scene: visible Subjects and "
                    "their current wardrobe/exposure, exact location and geography, active props, "
                    "lighting, camera setup and audio phase at the opening instant. State what is "
                    "present directly; never mention source traits, discarded alternatives, negative "
                    "instructions, validation, continuity rules or planning terminology."
                ),
            },
            "entry_state": {
                "type": "string",
                "minLength": 80,
                "description": (
                    "Private continuity ledger at the first rendered instant: visible cast, "
                    "body pose and contact, wardrobe and exposure state, held props and object "
                    "state, location and geography, screen direction, camera setup, lighting, "
                    "and current audio phase. For Continuous Story, copy the preceding scene's "
                    "exit_state verbatim before adding no new event."
                ),
            },
            "timeline_beats": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_beats,
                "items": {"type": "string", "minLength": 24},
                "description": (
                    "Private chronological cause-to-effect beats that fit the scene duration. "
                    "Each beat names the actor, physical action, direction/contact when relevant, "
                    "observable result, and synchronized camera or sound change."
                ),
            },
            "exit_state": {
                "type": "string",
                "minLength": 80,
                "description": (
                    "Private continuity ledger at the final rendered instant after every scene "
                    "action: cast, pose/contact, wardrobe/exposure, props, location/geography, "
                    "screen direction, camera, lighting, and audio phase. It must be concrete "
                    "enough to become the next Continuous Story entry_state verbatim."
                ),
            },
            "reference_contract": {
                "type": "string",
                "minLength": 30,
                "description": (
                    "Private scene-scoped reference ownership: which Subject/Picture/Video/Audio "
                    "is active, exactly what each contributes, and what unrelated attributes it "
                    "must not transfer. Omit assets not active in this scene."
                ),
            },
            "power_check": {
                "type": "string",
                "minLength": 80,
                "description": (
                    "Private verification of literal user-intent coverage, feasible pacing, "
                    "reference ownership, continuity or deliberate cut logic, camera readability, "
                    "speaker/dialogue syntax, sound separation, and a complete final state."
                ),
            },
        })
        shot["required"].extend((
            "scene_context", "entry_state", "timeline_beats", "exit_state",
            "reference_contract", "power_check",
        ))
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
                    + (
                        " For Continuous Story, also lock one immutable take ledger: "
                        "traversable route, action phase, screen direction, subject speed, "
                        "camera rig/height/axis/lens/framing/direction/speed, lighting and "
                        "sound phase; do not redesign these per scene."
                        if is_continuous else ""
                    )
                ),
            },
            "prompt_prefix": {
                "type": "string",
                "description": (
                    "Shared MiniMax H3 instructions prepended to every scene. Define "
                    "persistent visible identities with official Subject/Picture grammar, "
                    "for example `<Subject 1> is the woman grounded in <Picture 1>`. "
                    "Subject and Picture numbers are independent and multiple Subjects may "
                    "share one Picture. Include only Subjects visible in every requested scene; "
                    "future or temporary characters belong only in their active scene prompts. "
                    "Spell out every user-requested wardrobe and appearance override literally once; "
                    "never replace it with `preserve wardrobe` or `same outfit`. State all "
                    "permanent continuity rules. Keep this shared prefix compact because it is "
                    "prepended to every scene; never repeat its unchanged facts in scene prompts. "
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
    if split_global:
        schema["properties"]["prompt_prefix"]["description"] = (
            "Return an empty string. Split Global places all renderable shared and "
            "continuity state inside each scene_global instead of one shared prefix."
        )
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
    elif director_profile == "OpenRouter":
        schema["properties"]["persistent_visual_overrides"] = {
            "type": "string",
            "description": (
                "Exact concrete user-requested wardrobe, colors, garment fit, covered or "
                "exposed areas, hair, accessories, footwear and persistent props for Subjects "
                "visible from scene 1. These override conflicting mutable details in reference "
                "images. Extract only facts explicitly supplied by the user: do not infer, "
                "remove, expose, replace or intensify clothing, nudity, anatomy, contact or "
                "activity. In particular, clothing remains clothing unless the user explicitly "
                "requests its removal or nudity. Never use vague phrases such as preserve "
                "wardrobe or same outfit. "
                "Exclude future characters, later locations and events. Return an empty string "
                "only when the user supplied no persistent visible override."
            ),
        }
        schema["required"].append("persistent_visual_overrides")
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
                "Assign the connected original <Picture N> references and preserve "
                "identity plus genuinely persistent requested visual rules. Reference "
                "pose, framing, camera, expression and background are not persistent "
                "and must never be locked unless the user explicitly requests them."
            ),
        }
        schema["required"].append("storyboard_prompt_prefix")
    if power_prompt_rules:
        schema["properties"]["power_blueprint"] = {
            "type": "object",
            "properties": {
                "creative_objective": {
                    "type": "string",
                    "minLength": 50,
                    "description": "The exact audience-visible purpose and payoff of the complete generation.",
                },
                "user_action_spine": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 12},
                    "description": (
                        "The user's visible actions and state changes in their original order, "
                        "without stylistic rewriting, duplication or replacement."
                    ),
                },
                "reference_ownership": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 20},
                    "description": (
                        "One affirmative ownership rule per connected asset: what it contributes, "
                        "where it is active, and which unrelated traits are excluded from transfer."
                    ),
                },
                "continuity_ledger": {
                    "type": "string",
                    "minLength": 100,
                    "description": (
                        "Immutable and mutable production state: cast identities, wardrobe changes, "
                        "props, geography, screen direction, camera axis, lighting logic, action phase, "
                        "audio phase, and the ending obligation."
                    ),
                },
                "source_attribute_policy": {
                    "type": "string",
                    "minLength": 50,
                    "description": (
                        "Explicit boundary between source-reference identity and target design. "
                        "State which source attributes remain authoritative and which source "
                        "wardrobe, setting, lighting, pose, composition or style attributes are "
                        "discarded because the user requested a new design."
                    ),
                },
                "source_visual_inventory": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Private inventory of source-only wardrobe, setting, lighting, pose and "
                        "capture traits that must not leak into a requested redesign."
                    ),
                },
                "target_world_design": {
                    "type": "string",
                    "minLength": 50,
                    "description": (
                        "Concrete target environment and visual-world design: location, architecture, "
                        "materials, geography, lighting, palette, medium and atmosphere. Translate any "
                        "named universe or style into observable production details instead of leaving "
                        "it as a label. Use N/A with a clear preservation reason only when the user "
                        "requests no world, setting or style change."
                    ),
                },
                "target_subject_designs": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 40},
                    "description": (
                        "One concrete target design per affected subject: role/Subject binding, exact "
                        "new wardrobe pieces, materials, colors, footwear, accessories, grooming and "
                        "other requested appearance changes. Never copy source clothing when the user "
                        "requests new wardrobe. Empty only when no subject redesign is requested."
                    ),
                },
                "design_change_map": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One explicit source -> target replacement per affected subject and world. "
                        "The target side must be visibly different and newly invented."
                    ),
                },
                "coverage_map": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 20},
                    "description": (
                        "Every explicit user-requested visible action, transformation, dialogue, "
                        "sound, style, and required ending mapped to the scene that executes it."
                    ),
                },
            },
            "required": [
                "creative_objective", "user_action_spine", "reference_ownership",
                "continuity_ledger", "source_attribute_policy", "source_visual_inventory",
                "target_world_design", "target_subject_designs", "design_change_map", "coverage_map",
            ],
            "additionalProperties": False,
        }
        schema["required"].append("power_blueprint")
        schema["properties"]["prompt_prefix"]["description"] = (
            "Minimal shared prefix for Power mode. Include only immutable Subject-to-Picture "
            "identity bindings and a concise persistent capture style. Do not put wardrobe, "
            "location, pose, props, exposure, source-attribute policies, negative instructions "
            "or mutable continuity state here; those belong in each shot's scene_context."
        )
    if mood_assets_connected:
        schema["properties"]["mood_analysis"] = {
            "type": "string",
            "minLength": 80,
            "description": (
                "Concise private mood-board interpretation. Map each supplied <Mood N> "
                "or <Mood Video> only to the attributes explicitly requested by the user, "
                "such as wardrobe, pose, action, staging, environment, composition, camera, "
                "lighting, color, texture or pacing. State what is borrowed and what remains "
                "owned by the production Pictures. Mood assets never establish identity and "
                "their tags must never appear in prompt_prefix or any scene prompt."
            ),
        }
        schema["required"].append("mood_analysis")
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


def _normalize_h3_dialogue(value: str, language: str) -> str:
    """Render legacy quoted dialogue with MiniMax H3's official <d> grammar."""
    text = str(value or "")
    spoken_language = str(language or "English").strip()
    if spoken_language == "No dialogue":
        return text
    pattern = re.compile(
        r"\((S[1-4])\)\s*(?:says\s*)?:\s*[\"“]([^\"”]+)[\"”]",
        flags=re.IGNORECASE,
    )
    return pattern.sub(
        lambda match: (
            f"({match.group(1).upper()}) says: "
            f"<d>[{spoken_language}] {match.group(2).strip()}</d>"
        ),
        text,
    )


def _ensure_dialogue_lipsync(value: str) -> str:
    """Add one production cue only when an actual attributed line exists."""
    text = str(value or "").strip()
    has_dialogue = re.search(
        r"\(S[1-4]\)(?:\s*:\s*[\"“][^\"”]+[\"”]|\s+says:\s*<d>\[[^\]]+\].+?</d>)",
        text,
        flags=re.IGNORECASE,
    )
    if not has_dialogue:
        return text
    if re.search(
        r"\b(?:lip[ -]?sync|lip articulation|mouth articulation|spoken syllable)",
        text,
        flags=re.IGNORECASE,
    ):
        return text
    cue = (
        "The visible assigned speaker articulates every quoted syllable with precise "
        "natural lip sync; listeners keep their mouths still."
    )
    return f"{text} {cue}".strip()


def _ensure_final_scene_closure(value: str, director_mode: str) -> str:
    """Give the delivered final H3 clip an explicit terminal performance beat."""
    text = str(value or "").strip()
    if str(director_mode) not in {"Continuous Story", "Cinematic Cuts"}:
        return text
    if "FINAL SCENE CLOSURE:" in text:
        return text
    cue = (
        "FINAL SCENE CLOSURE: Complete the requested outcome, introduce no new "
        "action, and hold the resolved final state as camera and sound settle."
    )
    return f"{text} {cue}".strip()


def _normalize_visual_subject_labels(value: str) -> str:
    """Convert bare Gemma cast labels to H3 visual Subject tags.

    Parenthesized speaker labels are intentionally preserved for dialogue.
    Subject and Picture numbers are independent: several Subjects may be
    grounded in the same Picture when one reference contains several people.
    """
    return re.sub(
        r"(?<!\()\bS([1-4])\b",
        lambda match: f"<Subject {match.group(1)}>",
        str(value or ""),
        flags=re.IGNORECASE,
    )


def _route_scoped_subject_prefix(
    prompt_prefix: str, shots: list[dict]
) -> tuple[str, list[str], int]:
    """Keep future/temporary cast definitions out of the shared H3 prefix.

    A model may correctly delay a character in scene prompts while defining that
    character globally. MiniMax then sees the future identity from frame one and
    may render it early. Route every Subject-bearing prefix unit only to scenes
    that actually mention all Subjects in that unit; keep it global only when all
    referenced Subjects are active in every scene.
    """
    normalized_shots = [
        _normalize_visual_subject_labels(str(shot.get("prompt") or ""))
        if isinstance(shot, dict) else ""
        for shot in shots
    ]
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?;])\s+|[\r\n]+", str(prompt_prefix or ""))
        if unit.strip()
    ]
    shared: list[str] = []
    scoped: list[list[str]] = [[] for _ in shots]
    routed = 0
    for unit in units:
        subjects = sorted(set(re.findall(
            r"<Subject\s+([1-4])>", unit, flags=re.IGNORECASE
        )))
        if not subjects:
            shared.append(unit)
            continue
        active = [
            all(re.search(
                rf"<Subject\s+{re.escape(subject)}>", scene,
                flags=re.IGNORECASE,
            ) for subject in subjects)
            for scene in normalized_shots
        ]
        if active and all(active):
            shared.append(unit)
            continue
        for index, is_active in enumerate(active):
            if is_active:
                scoped[index].append(unit)
        routed += 1
    return (
        "\n".join(shared).strip(),
        ["\n".join(items).strip() for items in scoped],
        routed,
    )


def _toolkit_visible_cast_lock(value) -> str:
    """Render one short positive occupancy statement for MiniMax H3."""
    if not isinstance(value, list):
        return ""
    cast = []
    seen = set()
    for item in value:
        role = _normalize_visual_subject_labels(str(item or "").strip())
        role = re.sub(r"\s+", " ", role).strip(" ,.;")
        key = role.casefold()
        if not role or key in seen:
            continue
        seen.add(key)
        cast.append(role)
    if not cast:
        return "The frame contains no visible people."
    if len(cast) == 1:
        return (
            f"Exactly one physical person is visible in this scene: {cast[0]}. "
            "This identity occupies one body."
        )
    names = ", ".join(cast[:-1]) + f" and {cast[-1]}"
    return (
        f"Exactly {len(cast)} physical people are visible in this scene: {names}. "
        "Each listed identity occupies one distinct body."
    )


def _director_style_contract(
    genre: str,
    secondary_genre: str,
    motion_style: str,
    secondary_motion_style: str,
    visual_look: str,
    secondary_visual_look: str,
    compact: bool = False,
) -> str:
    """Create a deterministic style lock for local models with aesthetic bias."""
    primary = str(genre or "Auto").strip()
    secondary = str(secondary_genre or "None").strip()
    motion = str(motion_style or "Auto").strip()
    secondary_motion = str(secondary_motion_style or "None").strip()
    look = str(visual_look or "Auto").strip()
    secondary_look = str(secondary_visual_look or "None").strip()
    if compact:
        genre_text = (
            "infer one concrete primary genre from the user request and target design"
            if primary in {"Auto", "Auto — Infer from References & Prompt"} else
            primary
        )
        if secondary not in {"", "None", "Auto", primary}:
            genre_text += f", supported by {secondary}"
        motion_text = motion if motion != "Auto" else "infer motion from the requested action"
        if secondary_motion not in {"", "None", "Auto", motion}:
            motion_text += f" with {secondary_motion} as a secondary treatment"
        look_text = look if look != "Auto" else "infer one specific capture look"
        if secondary_look not in {"", "None", "Auto", look}:
            look_text += f" with {secondary_look} as a restrained finish"
        return (
            f"Production style: {genre_text}. Motion: {motion_text}. Visual look: {look_text}. "
            "Express these choices through concrete genre-specific staging, camera, lighting, "
            "pacing and sound."
        )
    parts = [
        "MANDATORY STYLE CONTRACT FOR EVERY SCENE:",
        (
            "Infer one primary genre from the request and references, then express its "
            "recognizable production format through concrete staging, performance, camera, "
            "lighting, setting, pacing and sound."
            if primary in {"Auto", "Auto — Infer from References & Prompt"} else
            f"Primary genre is `{primary}`. Its recognizable production format must control "
            "staging, performance, camera, lighting, setting, pacing and sound; it is not "
            "merely a label or mood adjective."
        ),
    ]
    if secondary == "Auto" and primary not in {
        "Auto", "Auto — Infer from References & Prompt"
    }:
        parts.append(
            "Infer one compatible secondary genre and apply it only as a clearly visible "
            "supporting layer without replacing the primary format."
        )
    elif secondary not in {"", "None", "Auto", primary}:
        parts.append(
            f"Secondary genre is `{secondary}`. Express its compatible conventions as a "
            "supporting layer without replacing the primary format."
        )
    if secondary_motion == "Auto" and motion != "Auto":
        parts.append(
            "Infer one compatible secondary motion treatment. It may add camera operation, "
            "stabilization or rhythmic nuance, but the primary motion remains authoritative."
        )
    elif secondary_motion not in {"", "None", "Auto", motion}:
        parts.append(
            f"Secondary motion style is {secondary_motion}: "
            f"{MOTION_STYLES[secondary_motion]} Apply it only as a compatible supporting "
            "behavior; if it conflicts with the primary motion, preserve the primary."
        )
    if look == "Auto":
        parts.append(
            "Infer a capture look from the chosen genre, request and references; keep it "
            "specific and consistent instead of defaulting to generic film polish."
        )
    else:
        parts.append(
            f"Capture look is `{look}`: {VISUAL_LOOKS[look]}"
        )
    if secondary_look == "Auto" and look != "Auto":
        parts.append(
            "Infer one compatible secondary visual treatment and apply it as a restrained "
            "finishing accent without replacing the primary capture medium."
        )
    elif secondary_look not in {"", "None", "Auto", look}:
        parts.append(
            f"Secondary visual look is {secondary_look}: {VISUAL_LOOKS[secondary_look]} "
            "Blend only compatible color, texture or finishing traits. The primary look controls "
            "capture medium, viewpoint and base image behavior whenever the two conflict."
        )
    parts.append(
        "Do not default to warm cinematic lighting, amber glow, moody cinematic atmosphere, "
        "shallow depth of field, polished studio lighting or a slow dramatic push-in. Use any "
        "of those only when the selected genre, capture look, user request or visible reference "
        "specifically motivates it. Describe the actual genre-specific alternative in each scene."
    )
    return "\n".join(parts)


def _parse_json_response(text: str) -> dict:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The language model returned incomplete or invalid JSON. Increase max_tokens, "
            "reduce the scene count, or use a model with reliable structured outputs."
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(
            "The language model returned JSON, but the story result is not an object."
        )
    return value


def _parse_direct_prompt_lines(text: str, scene_count: int) -> list[str]:
    """Normalize a schema-free Director answer into one prompt block per scene."""
    value = str(text or "").strip()
    value = re.sub(r"^```(?:text|markdown)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    expected = int(scene_count)
    tagged_separator = r"<<<H3_SCENE(?:_\d+)?>>>"

    # Accept both the requested between-scene token and the numbered visual
    # headers emitted by some models (and by our own readable output).
    if re.search(tagged_separator, value, flags=re.IGNORECASE):
        chunks = re.split(tagged_separator, value, flags=re.IGNORECASE)
    else:
        marked = re.split(
            r"(?:^|\n)\s*(?:#{1,4}\s*)?SCENE\s+\d+\s*[:.\-—]*\s*",
            value,
            flags=re.IGNORECASE,
        )
        if len(marked) > 1:
            chunks = marked[1:]
        else:
            # Structured H3 blocks are also an unambiguous boundary when a
            # provider forgets the requested separator between complete scenes.
            roots = list(re.finditer(
                r"(?im)^\s*(?:\*\*)?(?:subject_definitions|"
                r"integrated_multimodal_description)\s*:(?:\*\*)?",
                value,
            ))
            if len(roots) > 1:
                chunks = [
                    value[root.start():(
                        roots[index + 1].start()
                        if index + 1 < len(roots) else len(value)
                    )]
                    for index, root in enumerate(roots)
                ]
            else:
                # One requested scene may freely contain paragraphs. For old
                # line-list responses, split only when the count is exact.
                lines = [line for line in value.splitlines() if line.strip()]
                chunks = lines if expected > 1 and len(lines) == expected else [value]
    prompts = []
    for chunk in chunks:
        cleaned = re.sub(
            r"^\s*(?:[-*]\s*)?(?:(?:SCENE\s*)?\d+\s*[:.)\-—]\s*)?",
            "", str(chunk), flags=re.IGNORECASE,
        )
        # Preserve the official H3 section layout inside each scene while
        # normalizing noisy spacing produced by chat models.
        cleaned = "\n".join(
            re.sub(r"[ \t]+", " ", line).strip()
            for line in cleaned.splitlines()
        ).strip()
        cleaned = cleaned.replace(r"\<", "<").replace(r"\>", ">")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        if cleaned:
            prompts.append(cleaned)
    if len(prompts) != expected:
        raise RuntimeError(
            f"Direct Prompt Director returned {len(prompts)} usable prompts; "
            f"exactly {expected} were requested. The response is incomplete or "
            "its scene separators are missing."
        )
    return prompts


def _parse_direct_prompt_response(content, scene_count: int) -> list[str]:
    """Read the minimal structured transport, with legacy text fallback."""
    expected = int(scene_count)
    value = content
    if isinstance(value, str):
        stripped = value.strip()
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return _parse_direct_prompt_lines(stripped, expected)
    if isinstance(value, dict) and isinstance(value.get("scene_prompts"), list):
        raw_prompts = value["scene_prompts"]
        if len(raw_prompts) != expected:
            raise RuntimeError(
                f"Direct Prompt Director returned {len(raw_prompts)} scene_prompts; "
                f"exactly {expected} were requested."
            )
        prompts = []
        for index, prompt in enumerate(raw_prompts, 1):
            if not isinstance(prompt, str) or not prompt.strip():
                raise RuntimeError(
                    f"Direct Prompt Director scene_prompts[{index - 1}] is empty."
                )
            prompts.append(_parse_direct_prompt_lines(prompt, 1)[0])
        return prompts
    return _parse_direct_prompt_lines(str(content or ""), expected)


def _format_direct_prompt_block(prompts: list[str]) -> str:
    """Render readable scene blocks without leaking headers into generation."""
    return "\n\n".join(
        f"<<<H3_SCENE_{index:02d}>>>\n{prompt}"
        for index, prompt in enumerate(prompts, 1)
    )


def _format_direct_chain_payload(prompts: list[str], duration: float,
                                 steps: int, seed: int,
                                 director_mode: str) -> str:
    """Carry chain settings through the same type-safe STRING connection."""
    metadata = json.dumps({
        "scene_duration_seconds": float(duration),
        "steps": int(steps),
        "base_seed": int(seed),
        "director_mode": str(director_mode),
    }, ensure_ascii=False, separators=(",", ":"))
    return f"<<<H3_CHAIN_META {metadata}>>>\n" + _format_direct_prompt_block(prompts)


def _extract_direct_chain_metadata(value: str) -> tuple[dict, str]:
    """Remove optional internal metadata before prompts reach MiniMax."""
    text = str(value or "").strip()
    match = re.match(r"^<<<H3_CHAIN_META\s+(\{[^\n]*\})>>>\s*", text)
    if not match:
        return {}, text
    try:
        metadata = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ValueError("Direct prompt chain metadata is invalid.") from error
    return metadata, text[match.end():].strip()


def _split_direct_prompt_block(value: str) -> list[str]:
    """Read numbered scene blocks, legacy delimiters, or old line lists."""
    _metadata, text = _extract_direct_chain_metadata(value)
    numbered_header = r"(?:^|\n)\s*<<<H3_SCENE_\d+>>>\s*(?:\n|$)"
    if re.search(numbered_header, text, flags=re.IGNORECASE):
        chunks = re.split(numbered_header, text, flags=re.IGNORECASE)[1:]
    elif "<<<H3_SCENE>>>" in text:
        chunks = text.split("<<<H3_SCENE>>>")
    else:
        # Backward compatibility with the first experimental workflows.
        chunks = text.splitlines()
    return [chunk.strip() for chunk in chunks if chunk.strip()]


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


_WORKSHEET_LEAK_MARKERS = re.compile(
    r"(?:[\"']?\s*[,}]?\s*)?(?:physical_performance|camera_plan|"
    r"environment_and_lighting|genre_execution|visual_look_execution|sound_plan|"
    r"dialogue|final_state|coverage_check)[\"']?\s*[:&]",
    flags=re.IGNORECASE,
)
_INLINE_SPEAKER_LINE = re.compile(
    r"\s*(?:[.?!]\s*)?[\"']?\s*\(?S[1-4]\)?\s*:?\s*"
    r"(?:[\"“][^\"”\r\n]*[\"”]|[^.?!\r\n]*(?:[.?!]|$))",
    flags=re.IGNORECASE,
)


def _clean_gemma_prose(value, *, allow_dialogue: bool = False) -> str:
    """Remove worksheet serialization leaks without rewriting scene content."""
    text = str(value or "").strip()
    leak = _WORKSHEET_LEAK_MARKERS.search(text)
    if leak:
        text = text[:leak.start()].rstrip(" \t\r\n,;:'\"")
    if not allow_dialogue:
        text = _INLINE_SPEAKER_LINE.sub("", text)
    text = text.replace("**", "")
    text = re.sub(r"[\"']\s*,\s*[.]", ".", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?:\s*\1)+", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(
        r"(?:\s+|^)(?:First|Then|Next|After that|Finally),?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(" \t\r\n,;")


def _clean_gemma_dialogue(value) -> str:
    """Keep valid speaker lines once and discard duplicated model chatter."""
    text = _clean_gemma_prose(value, allow_dialogue=True)
    matches = re.findall(
        r"\(?S([1-4])\)?\s*:\s*[\"“]([^\"”\r\n]+)[\"”]",
        text,
        flags=re.IGNORECASE,
    )
    unique = []
    seen_speakers = set()
    for speaker, words in matches:
        words = re.sub(r"\s+", " ", words).strip()
        # Gemma sometimes writes the requested line and immediately repeats a
        # translation with the same speaker. The requested-language line is
        # consistently first; accept one exact utterance per speaker rather
        # than leaking a bilingual duplicate into MiniMax.
        if words and speaker not in seen_speakers:
            seen_speakers.add(speaker)
            unique.append(f'(S{speaker}): "{words}"')
    return " ".join(unique)


def _dedupe_gemma_inline_dialogue(text: str) -> str:
    """Remove an immediately repeated same-speaker translation from prose."""
    previous = None
    # Standardize `S2:`, `[S2]:` and the already-correct `(S2):` before
    # looking for repetitions; otherwise two equivalent source forms become
    # identical only after this function has already run.
    cleaned = _normalize_speaker_labels(str(text or ""))
    # Normalize Gemma's common `(S2) "line"` variant before deduplication.
    cleaned = re.sub(
        r"\(S([1-4])\)\s*(?=[\"“])",
        lambda match: f"(S{match.group(1)}): ",
        cleaned,
        flags=re.IGNORECASE,
    )
    pattern = re.compile(
        r'(\(S([1-4])\)\s*:\s*["“][^"”\r\n]+["”])'
        r'(?:\s+\(S\2\)\s*:\s*["“][^"”\r\n]+["”])+',
        flags=re.IGNORECASE,
    )
    while previous != cleaned:
        previous = cleaned
        cleaned = pattern.sub(r"\1", cleaned)
    # A final bare speaker marker is model chatter, not performable dialogue.
    cleaned = re.sub(
        r"\s+\(S([1-4])\)\s*\.?(?=\s|$)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


_GEMMA_FIELD_NAMES = (
    r"physical_performance|camera_plan|environment_and_lighting|"
    r"genre_execution|visual_look_execution|sound_plan|final_state|coverage_check"
)
_GEMMA_BROKEN_VALUE = re.compile(
    # Reject leaked snake_case worksheet prose such as
    # `physical_performance_subject_moves...`, or a bare schema-field label.
    # Do not reject ordinary prose containing phrases like "final state".
    rf"(?:\b(?:{_GEMMA_FIELD_NAMES})(?:_[a-z][a-z0-9_]*)+\b|"
    rf"^\s*(?:{_GEMMA_FIELD_NAMES})\s*:?\s*$|^[\s\"'\[\]{{}},:]+$)",
    flags=re.IGNORECASE,
)


def _gemma_scene_issues(raw: dict, scene_count: int) -> list[str]:
    """Reject truncated worksheets before they become expensive video prompts."""
    shots = raw.get("shots") if isinstance(raw, dict) else None
    if not isinstance(shots, list) or len(shots) != int(scene_count):
        return [f"expected {scene_count} complete scenes"]
    issues = []
    prose_minimums = {
        "opening_state": 45,
        "physical_performance": 45,
        "camera_plan": 35,
        "environment_and_lighting": 40,
        "genre_execution": 35,
        "visual_look_execution": 35,
        "sound_plan": 25,
        "final_state": 45,
    }
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            issues.append(f"scene {index} is not an object")
            continue
        direct_prompt = _clean_gemma_prose(
            shot.get("prompt"), allow_dialogue=True
        )
        if direct_prompt:
            sentences = [
                part for part in re.split(r"(?<=[.!?])\s+", direct_prompt)
                if len(part.strip()) >= 15
            ]
            # A useful H3 shot can be concise. Character-count targets made
            # valid four-scene plans fail twice simply because Gemma wrote
            # compact prose. Reject structural damage, not writing style.
            if len(direct_prompt) < 160 or len(sentences) < 3:
                issues.append(
                    f"scene {index} has an incomplete direct production prompt"
                )
            elif _GEMMA_BROKEN_VALUE.search(direct_prompt):
                issues.append(
                    f"scene {index} contains broken worksheet fragments"
                )
            continue
        for field, minimum in prose_minimums.items():
            cleaned = _clean_gemma_prose(shot.get(field))
            if len(cleaned) < minimum or _GEMMA_BROKEN_VALUE.search(cleaned):
                issues.append(f"scene {index} has broken or incomplete {field}")
        beats = shot.get("action_beats")
        usable = 0
        if isinstance(beats, list):
            for beat in beats:
                if isinstance(beat, dict):
                    candidate = " ".join(str(beat.get(key) or "") for key in (
                        "visible_action", "physical_mechanics", "visible_result"
                    ))
                else:
                    candidate = str(beat or "")
                cleaned = _clean_gemma_prose(candidate)
                if len(cleaned) >= 45 and not _GEMMA_BROKEN_VALUE.search(cleaned):
                    usable += 1
        if usable < 2:
            issues.append(f"scene {index} has fewer than two complete visible action beats")
    return issues


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
        "- Maintain one persistent cast ledger across the complete plan. Use `<Subject N>` "
        "for reusable visible identities. Each Subject represents exactly one physical person, "
        "never multiple copies. Use stable natural visual roles for characters without images.\n"
        "- Ground every referenced Subject in the correct `<Picture N>` inside prompt_prefix. "
        "Subject and Picture numbers are independent: one Picture may ground several Subjects.\n"
        "- Reserve `(S1)`, `(S2)`, `(S3)` and `(S4)` exclusively for dialogue or singing "
        "attribution; never use bare S labels as visual character names.\n"
        "- In every scene prompt, make the intended visible principal-character count and "
        "their Subject tags or natural visual roles unambiguous. Do not invent background copies.\n"
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
    if source_video_connected or mode in ("Continuous Story", "Cinematic Cuts"):
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
    if mode == "Image to Video":
        return f"""{common}
MANDATORY MODE — NATIVE IMAGE TO VIDEO (FL2VA):
- Produce exactly one scene prompt for one uninterrupted I2V clip.
- The first connected image is the literal first frame, not a loose identity reference and not
  a Picture-tagged Ref2VA asset. If a second image is connected, it is the literal last frame.
- Never output `<Picture N>`, `<Subject N>`, reference-assignment prose, contact-sheet language,
  image-analysis methodology, or instructions to recreate the source frame.
- Begin from the exact visible source state: identity, anatomy, wardrobe, pose, expression,
  objects, composition, crop, camera position, lighting, shadows and background geometry.
- Describe what starts moving immediately after that frame, using one physically achievable
  progression. Name subject movement, body and object mechanics, camera motion, environmental
  response and synchronized native audio. Preserve face, hands, clothing and scene geometry.
- Do not spend prompt space inventorying static source details the model already receives.
  Mention a source detail only when it must remain stable during motion or changes by request.
- If an optional last frame exists, describe a plausible continuous path that arrives at its
  exact composition and visible state. Do not request a cut, teleport, morph, dissolve or reset.
- Follow the user's requested action literally. Do not invent an unrelated narrative, extra
  characters, wardrobe changes, camera angles or scene changes.
- End with a clear final motion state and allow natural settle time; do not leave the principal
  requested action unfinished."""
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
- The result must read as the same source photograph after the complete requested edit, never
  as an unrelated regenerated variation. The user may request one change or a coordinated set
  of changes; apply all of them accurately while avoiding changes they did not request.
- Apply the exact requested transformation. Preserve everything else as faithfully as
  the model permits: identity, facial structure, expression, pose, anatomy, hands, hairstyle
  except the requested attribute, wardrobe, objects, background, composition, crop, viewpoint,
  lens perspective, depth of field, lighting, shadows, color relationships and visual style.
- Never redesign, beautify, restyle, relocate, re-pose, reframe, zoom, rotate, add or remove an
  element unless the user explicitly requests it. Do not introduce creative improvements.
- State the requested delta clearly and state that all unrequested pixels and scene properties
  remain visually unchanged. Do not create a collage, comparison, split screen or before/after.
- The requested transformation may freely alter any combination of subjects, appearance,
  wardrobe, objects, environment, lighting, style, composition, or other visible properties.
  Treat every explicitly requested change as authorized, but retain all properties outside that
  requested scope. Additional connected pictures may supply identities or any visual material
  explicitly requested by the user; they must not introduce unrelated content on their own.
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
- Treat the complete multi-scene result as ONE UNBROKEN TAKE. Scene boundaries are invisible
  time partitions required by generation, never editorial cuts, new coverage, establishing
  shots, narrative chapters, location resets or opportunities to redesign the camera.
- Before writing scenes, privately lock one take ledger: exact location and traversable route;
  subject identity, wardrobe and props; screen direction and action phase; camera rig, height,
  side of axis, lens behavior, framing distance, movement vector and speed; lighting direction,
  exposure, ambience, music and sound perspective. Keep that ledger unchanged unless the user
  explicitly requests an observable change, and then show the complete change continuously.
- The selected genre controls performance, staging, texture and sound inside this same take.
  The selected motion style controls the speed and energy of the same continuous action. Neither
  selection authorizes a cut, a new angle, a new lens, a new location, a camera reversal, a
  sudden acceleration, a dramatic push-in or an invented escalation.
- Scene 1 begins the requested action without adding a larger story arc. Every later scene must
  begin at the exact next instant after the preceding final state: same body configuration,
  planted foot, hand and object positions, momentum, facing, gaze, fabric and hair motion,
  camera trajectory, framing distance, environment, lighting and sound phase.
- Keep camera and subject velocity continuous across every boundary. A tracking camera keeps
  tracking in the same direction at the same height, distance and pace; it must not tilt,
  reframe, slow, reverse or become a push-in merely because a new scene begins.
- Never teleport between a sidewalk, alley, doorway, room or other space. A location may change
  only when the user requests it or the subject visibly traverses the complete connected route;
  the preceding scene must reach the exact threshold state inherited by the next scene.
- End every non-final scene during one clearly unfinished physical action. Start the next scene
  by continuing that exact action, not by restating, restarting, intensifying or replacing it.
- For a simple repeated request such as walking, dancing or observing, sustain that action and
  its established setting/camera across all scenes. Do not invent seduction, contact, dialogue,
  destinations, wardrobe changes, plot escalation or a final encounter unless requested.
- Preserve identity, wardrobe, props, geography and relationships across the sequence.
- The final scene is the actual ending, not another bridge. Complete every remaining action and
  outcome explicitly requested by the user before the shot ends, then hold the resolved visible
  state for a final beat while camera motion and sound settle naturally. Never finish at the
  beginning, anticipation or midpoint of that final action. For an intentionally ongoing repeated
  action, sustain it through the shot and finish on a stable natural beat. Do not introduce a new
  camera setup, destination, contact or story event merely to manufacture a dramatic resolution.
- Do not write meta phrases such as `Scene 1 establishes`, `Scene 2 continues`, `the story
  intensifies`, or `the final scene resolves`. Describe only the uninterrupted visible take."""


def _gemma_scene_prompt(shot: dict) -> str:
    """Compile Gemma's private scene worksheet into one MiniMax-ready prompt."""
    beats = shot.get("action_beats")
    if not isinstance(beats, list) or not beats:
        return ""
    def sentence(value) -> str:
        text = _clean_gemma_prose(value)
        return text if not text or text.endswith((".", "!", "?", '"', "”")) else f"{text}."

    clean_beats = []
    for beat in beats:
        if isinstance(beat, dict):
            phase = str(beat.get("phase") or "").strip().lower()
            action = sentence(beat.get("visible_action"))
            mechanics = sentence(beat.get("physical_mechanics"))
            result = sentence(beat.get("visible_result"))
            rendered = " ".join(part for part in (action, mechanics, result) if part)
            if rendered:
                clean_beats.append((phase, rendered))
        elif _clean_gemma_prose(beat):
            # Backward compatibility for plans saved by the first Gemma worksheet.
            clean_beats.append(("", sentence(beat)))
    transitions = ("First,", "Then,", "Next,", "After that,", "Finally,")
    action_prose = " ".join(
        f"{transitions[min(index, len(transitions) - 1)]} "
        f"{rendered}"
        for index, (_phase, rendered) in enumerate(clean_beats)
    )
    opening = sentence(shot.get("opening_state"))
    parts = [
        f"At the opening of the shot, {opening.lstrip()}" if opening else "",
        action_prose,
        " ".join(filter(None, (
            sentence(shot.get("physical_performance")),
            sentence(shot.get("camera_plan")),
        ))),
        " ".join(filter(None, (
            sentence(shot.get("environment_and_lighting")),
            sentence(shot.get("genre_execution")),
            sentence(shot.get("visual_look_execution")),
        ))),
        sentence(shot.get("sound_plan")),
    ]
    dialogue = _clean_gemma_dialogue(shot.get("dialogue"))
    if dialogue:
        parts.append(dialogue)
    final_state = sentence(shot.get("final_state"))
    if final_state:
        parts.append(f"By the end of the shot, {final_state}")
    return "\n\n".join(part for part in parts if str(part).strip())


def _sanitize_generic_reference_observations(value) -> str:
    """Remove visual facts inferred from a replaceable input frame."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"AUTHORITATIVE USER VISUAL OVERRIDES\b.*?(?=MANDATORY STYLE CONTRACT\b|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    observation = re.compile(
        r"\b(?:supplied|input|source)\s+(?:first|last)?\s*frame\b.*?"
        r"\b(?:shows?|wears?|wearing|dressed|has|visible|appearance|hair|"
        r"on the left|on the right)\b",
        re.IGNORECASE,
    )
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept = [part.strip() for part in parts if part.strip() and not observation.search(part)]
    text = "\n".join(kept).strip()

    # A reusable action must not freeze garment details inferred from the
    # currently connected image. Collapse colours, materials and garment types
    # to neutral wardrobe language even when the model leaks them into a beat.
    garment = (
        r"(?:(?:leather|denim|cotton|silk|satin|lace|black|white|red|blue|green|"
        r"gray|grey|pink|purple|brown|dark|light|cropped|oversized|tight|loose|"
        r"hooded|ribbed|metallic|holographic|ultra-short|short|long)\s+)*"
        r"(?:jacket|coat|hoodie|sweater|shirt|t-shirt|tee|top|blouse|dress|"
        r"skirt|miniskirt|shorts|jeans|pants|trousers|leggings|underwear|bra|"
        r"panties|stockings|socks|shoes|sneakers|boots)\b"
    )
    text = re.sub(garment, "clothes", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\bclothes(?:\s*,\s*clothes)+(?:\s*,?\s*and\s+clothes)?\b|"
        r"\bclothes\s+and\s+clothes\b",
        "clothes",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _power_visual_redesign_requested(value: str) -> bool:
    return bool(re.search(
        r"\b(?:ambientad[oa]s?|universo|mundo|estilo|vestuarios?|bestuarios?|"
        r"atuendos?|trajes?|outfits?|wardrobes?|costumes?|setting|environment|world|"
        r"visual\s+(?:style|look)|(?:new|nuevo|capture)\s+look|look\s+(?:visual|de)|"
        r"redesign|re-?dress|change\s+(?:their\s+)?clothes|cambi(?:a|ar)\s+(?:la\s+)?ropa|"
        r"\bwears?\b|\bviste\b)",
        str(value or ""),
        flags=re.IGNORECASE,
    ))


def _power_plan_issues(
    raw: dict,
    scene_count: int,
    director_mode: str,
    is_still: bool = False,
    user_request: str = "",
) -> list[str]:
    """Validate the private production controls used by experimental Power mode."""
    issues = []
    blueprint = raw.get("power_blueprint")
    if not isinstance(blueprint, dict):
        issues.append("power_blueprint is missing")
    else:
        action_spine = blueprint.get("user_action_spine")
        if not isinstance(action_spine, list) or not action_spine:
            issues.append("power_blueprint.user_action_spine is empty")
        for key in (
            "creative_objective", "continuity_ledger", "source_attribute_policy",
            "target_world_design",
        ):
            if len(str(blueprint.get(key) or "").strip()) < 50:
                issues.append(f"power_blueprint.{key} is incomplete")
        if not isinstance(blueprint.get("coverage_map"), list) or not blueprint.get("coverage_map"):
            issues.append("power_blueprint.coverage_map is empty")
        if _power_visual_redesign_requested(user_request):
            inventory = blueprint.get("source_visual_inventory")
            if not isinstance(inventory, list) or not inventory:
                issues.append("the visual redesign lacks a private source_visual_inventory")
            designs = blueprint.get("target_subject_designs")
            if not isinstance(designs, list) or not designs:
                issues.append(
                    "the user requested a visual redesign but target_subject_designs is empty"
                )
            world = str(blueprint.get("target_world_design") or "").strip()
            if len(world) < 120:
                issues.append(
                    "the requested target world/look is not translated into a concrete design"
                )
            change_map = blueprint.get("design_change_map")
            if not isinstance(change_map, list) or not change_map:
                issues.append("the visual redesign lacks an explicit design_change_map")
            # Its wording is intentionally not linted: synonyms and natural prose
            # must never trigger another paid repair or block a usable plan.

    shots = raw.get("shots")
    if not isinstance(shots, list) or len(shots) != scene_count:
        return issues + [f"expected {scene_count} complete Power scenes"]

    previous_exit = ""
    previous_compact_closing = ""
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            issues.append(f"scene {index} is not structured")
            continue
        if is_still:
            if len(str(shot.get("prompt") or "").strip()) < 80:
                issues.append(f"image {index} edit/generation prompt is too short")
            continue
        entry = str(shot.get("entry_state") or "").strip()
        exit_state = str(shot.get("exit_state") or "").strip()
        scene_context = str(shot.get("scene_context") or "").strip()
        beats = shot.get("timeline_beats")
        reference_contract = str(shot.get("reference_contract") or "").strip()
        check = str(shot.get("power_check") or "").strip()
        prompt = str(shot.get("prompt") or "").strip()
        if len(scene_context) < 60:
            issues.append(f"scene {index} scene_context is incomplete")
        if re.search(
            r"\b(?:ignore|discard|do not|don't|must not|instead of|source (?:image|wardrobe|"
            r"setting|look)|reference (?:wardrobe|clothes|setting|look)|negative prompt)\b",
            scene_context,
            flags=re.IGNORECASE,
        ):
            issues.append(f"scene {index} scene_context contains negative/source instructions")
        if len(entry) < 80:
            issues.append(f"scene {index} entry_state is incomplete")
        if len(exit_state) < 80:
            issues.append(f"scene {index} exit_state is incomplete")
        if not isinstance(beats, list) or not beats:
            issues.append(f"scene {index} has no chronological timeline beats")
        if len(reference_contract) < 30:
            issues.append(f"scene {index} reference ownership is incomplete")
        if len(check) < 80:
            issues.append(f"scene {index} power_check is incomplete")
        if len(prompt) < 160:
            issues.append(f"scene {index} rendered prompt is too short")
        if re.search(
            r"\b(?:entry_state|exit_state|timeline_beats|power_check|coverage_map)\b",
            prompt,
            flags=re.IGNORECASE,
        ):
            issues.append(f"scene {index} leaked private worksheet labels into its prompt")
        if prompt.count(";") > 2:
            issues.append(f"scene {index} overuses semicolons instead of chronological prose")
        if "<d>" in prompt and not re.search(
            r"\(S\d+\).*?<d>\[", prompt, flags=re.IGNORECASE | re.DOTALL
        ):
            issues.append(f"scene {index} dialogue is not bound to a stable speaker")
        if director_mode == "Continuous Story" and re.search(
            r"\b(?:camera\s+cuts?|cut\s+to|hard\s+cut|new\s+camera\s+setup)\b",
            prompt,
            flags=re.IGNORECASE,
        ):
            issues.append(f"scene {index} introduces a cut inside Continuous Story")
        if director_mode == "Continuous Story" and index > 1:
            if entry != previous_exit:
                issues.append(
                    f"scene {index} entry_state does not exactly inherit scene {index - 1} exit_state"
                )
        previous_exit = exit_state

    coverage = " ".join(
        str(item) for item in ((blueprint or {}).get("coverage_map") or [])
    ).lower()
    if scene_count > 1 and not re.search(r"scene\s*1|scene_?01|shot\s*1", coverage):
        issues.append("coverage_map does not assign requests to scenes")
    return issues


def _apply_power_target_design(raw: dict, user_request: str) -> None:
    """Keep redesign policy private and expose only positive per-scene state."""
    if not _power_visual_redesign_requested(user_request):
        return
    existing = str(raw.get("prompt_prefix") or "").strip()
    pairs = []
    for subject, picture in re.findall(
        r"<Subject\s+([1-4])>[^\n.!?]{0,240}?<Picture\s+([1-4])>",
        existing,
        flags=re.IGNORECASE,
    ):
        pair = (int(subject), int(picture))
        if pair not in pairs:
            pairs.append(pair)
    raw["persistent_visual_overrides"] = ""
    identity_prefix = "\n".join(
        f"<Subject {subject}> is the persistent visual identity grounded in <Picture {picture}>."
        for subject, picture in pairs
    )
    raw["prompt_prefix"] = (
        identity_prefix
        or "Persistent identities are grounded in their connected Picture references."
    )


def _compile_story(
    raw: dict,
    scene_count: int,
    duration_seconds: float,
    steps: int,
    picture_count: int,
    director_mode: str = "Continuous Story",
    director_profile: str = "OpenRouter",
    toolkit_prompt_rules: bool = False,
    dialogue_language: str = "English",
    generic_mode: bool = False,
    power_prompt_rules: bool = False,
    split_global: bool = False,
) -> tuple[str, str, str, str, str]:
    synopsis = str(raw.get("synopsis") or "").strip()
    story_bible = str(raw.get("story_bible") or "").strip()
    prompt_prefix = _strip_dialogue_planning_rules(raw.get("prompt_prefix") or "")
    # Both planning backends occasionally compress a real reference assignment
    # into `[S1] <Picture 1>`. Normalize every backend to one H3 contract.
    # Subject and Picture captures are intentionally independent so multiple
    # people from one reference remain representable.
    prompt_prefix = re.sub(
        r"\[S([1-4])\]\s*(<Picture\s+([1-4])>)",
        lambda match: (
            f"<Subject {match.group(1)}> is the exact persistent subject defined by "
            f"{match.group(2)}; preserve identity, appearance, wardrobe, "
            "anatomy, and all relevant visible traits whenever present"
        ),
        prompt_prefix,
        flags=re.IGNORECASE,
    )
    prompt_prefix = _normalize_visual_subject_labels(prompt_prefix)
    persistent_visual_overrides = str(
        raw.get("persistent_visual_overrides") or ""
    ).strip()
    if generic_mode:
        # A reusable plan must never freeze incidental traits inferred from the
        # currently connected image. User-requested changes remain in the shot
        # instructions, while the replacement image supplies its own wardrobe,
        # hair, accessories and appearance at execution time.
        persistent_visual_overrides = ""
    if persistent_visual_overrides and not split_global:
        prompt_prefix = "\n\n".join((
            "AUTHORITATIVE USER VISUAL OVERRIDES — these mutable details replace "
            "conflicting clothing, appearance and prop details in reference images:\n"
            + persistent_visual_overrides,
            prompt_prefix,
        ))
    storyboard_prompt_prefix = str(raw.get("storyboard_prompt_prefix") or "").strip()
    source_video_analysis = str(raw.get("source_video_analysis") or "").strip()
    shots = raw.get("shots")
    if (not synopsis or not story_bible
            or (not prompt_prefix and not split_global)):
        raise RuntimeError("The director response is missing its synopsis, story bible, or shared prompt.")
    if not isinstance(shots, list) or len(shots) != scene_count:
        actual = len(shots) if isinstance(shots, list) else 0
        raise RuntimeError(
            f"The director returned {actual} scenes, but {scene_count} were requested. The plan was not accepted."
        )

    scoped_subject_prefixes = ["" for _ in shots]
    routed_subject_units = 0
    if not storyboard_prompt_prefix:
        prompt_prefix, scoped_subject_prefixes, routed_subject_units = (
            _route_scoped_subject_prefix(prompt_prefix, shots)
        )

    reference_assignment_prefix = storyboard_prompt_prefix or "\n".join((
        prompt_prefix,
        *scoped_subject_prefixes,
        *("\n".join((
            str(shot.get("scene_global") or ""),
            str(shot.get("prompt") or ""),
        )) for shot in shots
          if isinstance(shot, dict)),
    ))
    missing_tags = [
        f"<Picture {index}>"
        for index in range(1, picture_count + 1)
        if f"<Picture {index}>" not in reference_assignment_prefix
    ]
    repaired_reference_count = len(missing_tags)
    if missing_tags:
        fallback_assignments = " ".join(
            (
                f"<Picture {index}> is the connected interchangeable visual source assigned "
                "to this role; use the current image without describing its incidental visible traits."
                if generic_mode else
                f"<Picture {index}> is an exact connected visual source. Use it only for "
                "the person, object, place or style assigned to it by the user request and "
                "story bible; preserve the relevant visible traits whenever that assignment "
                "is active."
            )
            for index in range(1, picture_count + 1)
            if f"<Picture {index}>" in missing_tags
        )
        if storyboard_prompt_prefix:
            storyboard_prompt_prefix = "\n\n".join((
                storyboard_prompt_prefix, fallback_assignments
            ))
        else:
            prompt_prefix = "\n\n".join((prompt_prefix, fallback_assignments))

    compiled_shots = []
    # OpenRouter may optionally return the compact opening/action/closing
    # contract. Keep the previous closing local to this compilation so every
    # profile and schema path starts from a defined state.
    previous_compact_closing = ""
    seen_ids = set()
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            raise RuntimeError(f"Scene {index} is not a structured scene object.")
        shot_id = _safe_id(shot.get("id", ""), index)
        if shot_id in seen_ids:
            shot_id = f"{shot_id}_{index:02d}"
        seen_ids.add(shot_id)
        generated_prompt = _gemma_scene_prompt(shot) if director_profile == "Gemma" else ""
        source_prompt = shot.get("prompt") or generated_prompt
        compact_state = shot.get("compact_state")
        if director_profile != "Gemma" and isinstance(compact_state, dict):
            opening = str(compact_state.get("opening") or "").strip()
            action = str(compact_state.get("action") or "").strip()
            closing = str(compact_state.get("closing") or "").strip()
            if opening and action and closing:
                if director_mode == "Continuous Story" and previous_compact_closing:
                    # Continuity is deterministic in the compiler: do not trust a
                    # second LLM-authored paraphrase to preserve the exact state.
                    opening = previous_compact_closing
                source_prompt = " ".join((opening, action, closing))
                previous_compact_closing = closing
        if director_profile == "Gemma":
            source_prompt = _dedupe_gemma_inline_dialogue(source_prompt)
        source_prompt = _normalize_speaker_labels(source_prompt)
        source_prompt = _normalize_visual_subject_labels(source_prompt)
        if (toolkit_prompt_rules or split_global) and not power_prompt_rules:
            cast_lock = _toolkit_visible_cast_lock(shot.get("visible_cast"))
            if cast_lock:
                source_prompt = "\n\n".join((cast_lock, source_prompt))
        if split_global:
            scene_global = _normalize_visual_subject_labels(
                str(shot.get("scene_global") or "").strip()
            )
            if len(scene_global) < 80:
                raise RuntimeError(
                    f"Scene {index} Split Global state is missing or too short."
                )
            source_prompt = "\n\n".join((scene_global, source_prompt))
        if power_prompt_rules:
            scene_context = _normalize_visual_subject_labels(
                str(shot.get("scene_context") or "").strip()
            )
            if scene_context:
                source_prompt = "\n\n".join((scene_context, source_prompt))
        scoped_prefix = scoped_subject_prefixes[index - 1]
        if scoped_prefix:
            source_prompt = "\n\n".join((scoped_prefix, source_prompt))
        prompt = _normalize_speaker_labels(_strip_dialogue_planning_rules(
            source_prompt
        ))
        prompt = _normalize_h3_dialogue(prompt, dialogue_language)
        prompt = _ensure_dialogue_lipsync(prompt)
        if index == len(shots):
            prompt = _ensure_final_scene_closure(prompt, director_mode)
        if power_prompt_rules:
            prompt = re.sub(
                r"\s*\(\s*\d+\s+words?\s*\)\s*",
                " ",
                prompt,
                flags=re.IGNORECASE,
            )
            prompt = re.sub(
                r"(?i)\b(?:opening state|final state|continuity lock|"
                r"final scene closure)\s*:\s*",
                "",
                prompt,
            )
            prompt = re.sub(r"[ \t]{2,}", " ", prompt).strip()
        if len(prompt) < 80:
            raise RuntimeError(
                f"Scene {index} is too short to be a production-ready continuity prompt."
            )
        compiled_shot = {"id": shot_id, "prompt": prompt}
        if power_prompt_rules:
            compiled_shot["power_state"] = {
                "context": str(shot.get("scene_context") or "").strip(),
                "entry": str(shot.get("entry_state") or "").strip(),
                "beats": [
                    str(beat).strip() for beat in (shot.get("timeline_beats") or [])
                    if str(beat).strip()
                ],
                "exit": str(shot.get("exit_state") or "").strip(),
                "reference_contract": str(
                    shot.get("reference_contract") or ""
                ).strip(),
            }
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
        "director_mode": director_mode,
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
    if power_prompt_rules:
        plan["power_blueprint"] = raw.get("power_blueprint") or {}
    validation = (
        f"Valid: {director_mode} · {scene_count} scenes · {picture_count} references · "
        f"{float(duration_seconds):g}s requested per scene · {int(steps)} steps"
    )
    if routed_subject_units:
        validation += (
            f" · routed {routed_subject_units} temporally scoped Subject "
            f"definition{'s' if routed_subject_units != 1 else ''}"
        )
    if repaired_reference_count:
        validation += (
            f" · repaired {repaired_reference_count} omitted Picture "
            f"assignment{'s' if repaired_reference_count != 1 else ''}"
        )
    if toolkit_prompt_rules:
        findings = []
        combined_prompts = [
            "\n\n".join((prompt_prefix, str(shot.get("prompt") or "")))
            for shot in compiled_shots
        ]
        forbidden = (
            (r"\bno cuts?\b|\bwithout (?:any )?cuts?\b|\bnever cuts?\b", "names cuts negatively"),
            (r"\bno camera movement\b|\bno angle change\b", "names unwanted camera behavior"),
            (r"\bcontinu(?:e|es|ing) (?:the )?(?:same|previous)\b|\bthe same (?:room|shot|scene)\b", "relies on relative continuity wording"),
        )
        for scene_index, text in enumerate(combined_prompts, 1):
            for pattern, label in forbidden:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    findings.append(f"scene {scene_index} {label}")
        if any(
            re.search(r"<Picture\s+\d+>\s*(?:is|:).*\b(?:fully|partially)_preserved\b", text, re.IGNORECASE)
            for text in combined_prompts
        ):
            findings.append("retention appears bound to Picture rather than Subject")
        validation += (
            " · H3 Toolkit lint clean"
            if not findings else
            " · H3 Toolkit lint warnings: " + "; ".join(dict.fromkeys(findings))
        )
    if power_prompt_rules:
        validation += " · Power blueprint + state ledger validated"
    if split_global:
        validation += " · Split Global scene-scoped continuity"
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

    def build_request_payload(self, payload: dict) -> dict:
        return {
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

    def h3_chat_completion(self, payload: dict) -> dict:
        chat_url = _ollama_chat_url(self.server_url)
        request_payload = self.build_request_payload(payload)
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
                io.Image.Input(
                    "mood_image_1", optional=True,
                    tooltip=(
                        "Private Director mood reference. Address it as @mood1. It can "
                        "inspire requested wardrobe, pose, setting, composition or look, "
                        "but never becomes a MiniMax Picture or transfers identity."
                    ),
                ),
                io.Image.Input("mood_image_2", optional=True,
                               tooltip="Private Director mood reference: @mood2."),
                io.Image.Input("mood_image_3", optional=True,
                               tooltip="Private Director mood reference: @mood3."),
                io.Image.Input("mood_image_4", optional=True,
                               tooltip="Private Director mood reference: @mood4."),
                io.Image.Input(
                    "mood_video", optional=True,
                    tooltip=(
                        "Private IMAGE frame batch from VHS Load Video. Address it as "
                        "@moodvideo to borrow requested action, pose progression, camera "
                        "movement or pacing without making it a MiniMax video reference."
                    ),
                ),
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
                io.Combo.Input(
                    "secondary_motion_style",
                    options=["None", *MOTION_STYLES],
                    default="None",
                    tooltip=(
                        "Adds one compatible motion or camera behavior. The primary motion "
                        "style remains authoritative whenever the two selections conflict."
                    ),
                ),
                io.Combo.Input(
                    id="visual_look",
                    display_name="Visual Look",
                    options=list(VISUAL_LOOKS),
                    default="Auto",
                    tooltip=(
                        "Controls the capture aesthetic independently from genre and motion. "
                        "Non-cinematic choices explicitly suppress generic film polish."
                    ),
                ),
                io.Combo.Input(
                    id="secondary_visual_look",
                    display_name="Secondary Visual Look",
                    options=["None", *VISUAL_LOOKS],
                    default="None",
                    tooltip=(
                        "Adds compatible color, texture, processing, or finishing traits. "
                        "The primary look continues to control the base capture aesthetic."
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
                        "starts independent camera setups. Image to Video treats image_0 "
                        "as the literal first frame and image_1 as an optional last frame. "
                        "Reference Edit uses the input "
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
                io.Boolean.Input(
                    "toolkit_prompt_rules",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Experimental H3 Toolkit rules: self-contained rendered prompts, "
                        "positive camera wording, Subject-bound retention, and prompt lint."
                    ),
                ),
                io.Boolean.Input(
                    "split_global",
                    display_name="Split Global",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Experimental scene-scoped continuity. Replaces one heavy shared global "
                        "prompt with a dedicated positive global state for every scene, carrying "
                        "the latest cast, wardrobe, location, props, look, camera and audio while "
                        "keeping future elements out. Works with or without Toolkit."
                    ),
                ),
                io.Boolean.Input(
                    "generic_mode",
                    display_name="Generic Mode",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Create reusable prompts that bind generic roles such as the man, "
                        "woman, object, or place from <Picture N>, without copying incidental "
                        "facial, body, wardrobe, color, or background details from the image."
                    ),
                ),
                io.Boolean.Input(
                    "compact_mode",
                    display_name="Compact Mode",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Ask the LLM for a shorter production plan and cap its effective "
                        "output budget. It preserves actions, references, continuity, dialogue "
                        "and ending state while removing repetition to reduce token cost."
                    ),
                ),
                io.Boolean.Input(
                    "hold_plan",
                    display_name="Hold Plan",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Reuse this node's last successful plan and prompts exactly, without "
                        "calling OpenRouter or the connected LLM again. Image, prompt and "
                        "setting changes are ignored until Hold Plan is disabled."
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
                io.String.Output(
                    "mood_analysis",
                    tooltip=(
                        "Private interpretation of connected mood images/video. Mood tags "
                        "are resolved into prose and never passed to MiniMax."
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
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
        secondary_motion_style: str,
        visual_look: str,
        secondary_visual_look: str,
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
        mood_image_1=None,
        mood_image_2=None,
        mood_image_3=None,
        mood_image_4=None,
        mood_video=None,
        source_video=None,
        bypass_director: bool = False,
        toolkit_prompt_rules: bool = False,
        split_global: bool = False,
        generic_mode: bool = False,
        compact_mode: bool = False,
        hold_plan: bool = False,
        llm_model=None,
        power_prompt_rules: bool = False,
        unique_id=None,
    ) -> io.NodeOutput:
        # Power Mode was retired in favor of the orthogonal Split Global
        # contract. Accept the legacy keyword so old API workflows still load,
        # but never activate its former blueprint behavior.
        power_prompt_rules = False
        node_unique_id = str(unique_id or "default")
        hold_cache_key = f"{cls.__name__}:{node_unique_id}"
        if bool(hold_plan):
            held = _get_held_director_plan(hold_cache_key)
            if held is not None:
                if bool(getattr(cls, "PROMPT_LIST_MODE", False)):
                    if held.get("cache_kind") == "direct_scene_prompts":
                        held_validation = f"{held['validation']} · HOLD"
                        return io.NodeOutput(
                            held["prompt_lines"],
                            held["first_prompt"],
                            int(held["prompt_count"]),
                            held_validation,
                            "Held prompts · LLM not called",
                            "Credits unchanged",
                            held.get("mood_analysis", ""),
                            ui=ui.PreviewText(held["preview"]),
                        )
                    print(
                        "[H3 Direct Prompt Director] Ignoring an incompatible "
                        "legacy Hold record for this node."
                    )
                else:
                    held_validation = f"{held['validation']} · HOLD"
                    held_preview = (
                        f"{held['synopsis']}\n\n{held_validation}\n\n"
                        f"--- HELD PLAN JSON ---\n{held['plan_json']}"
                    )
                    plan_output = (
                        ExecutionBlocker(None) if draft_only else held["plan_json"]
                    )
                    return io.NodeOutput(
                        plan_output,
                        held["story_bible"],
                        held["synopsis"],
                        held_validation,
                        "Held plan · LLM not called",
                        "Credits unchanged",
                        held["scene_prompt"],
                        held["mode_prompt"],
                        held["source_video_analysis"],
                        held.get("mood_analysis", ""),
                        ui=ui.PreviewText(held_preview),
                    )
        if language in {"EspaÃ±ol", "EspaÃƒÂ±ol"}:
            language = "Español"
        language = str(language or "").strip() or "No dialogue"
        audio_content = str(audio_content or "Auto").strip()
        audio_content = AUDIO_CONTENT_LEGACY_ALIASES.get(audio_content, audio_content)
        if audio_content not in AUDIO_CONTENT_MODES:
            audio_content = "Auto"
        motion_style = str(motion_style or "Auto").strip()
        secondary_motion_style = str(secondary_motion_style or "None").strip()
        if secondary_motion_style not in {"None", *MOTION_STYLES}:
            secondary_motion_style = "None"
        if secondary_motion_style == motion_style:
            secondary_motion_style = "None"
        visual_look = str(visual_look or "Auto").strip()
        if visual_look not in VISUAL_LOOKS:
            visual_look = "Auto"
        secondary_visual_look = str(secondary_visual_look or "None").strip()
        if secondary_visual_look not in {"None", *VISUAL_LOOKS}:
            secondary_visual_look = "None"
        if secondary_visual_look == visual_look:
            secondary_visual_look = "None"
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
        secondary_motion_direction = ""
        if secondary_motion_style == "Auto":
            secondary_motion_direction = (
                " Infer one distinct compatible secondary motion treatment. Use it only "
                "as a supporting camera, stabilization, or rhythmic behavior; the primary "
                "motion remains authoritative."
            )
        elif secondary_motion_style != "None":
            secondary_motion_direction = (
                f" Secondary motion style: {secondary_motion_style}. "
                f"{MOTION_STYLES[secondary_motion_style]} Apply it as a supporting layer; "
                "discard any trait that contradicts the primary motion or Director Mode."
            )
        scene_count = int(scene_count)
        if not 1 <= scene_count <= 32:
            raise ValueError("scene_count must be between 1 and 32.")
        if director_mode == "Image to Video":
            scene_count = 1
            if image_0 is None:
                raise ValueError(
                    "Image to Video requires image_0 as the literal first frame. "
                    "image_1 is optional and becomes the literal last frame."
                )
        story_idea = str(story_idea or "").strip()
        reference_alias_rules = (
            "REFERENCE ALIASES: The user may identify connected inputs as @image1, "
            "@image2, @image3, @image4, image 1-4, or naturally as the first, second, "
            "third, or fourth image. Interpret each expression as the corresponding "
            "connected <Picture N>. Preserve the user's role assignment (for example, "
            "`the woman in the first image`) and use the exact MiniMax <Picture N> tag "
            "in the compiled generation prompt. Never confuse an image slot with a "
            "character number."
        )
        for index, ordinal in enumerate(
            ("first", "second", "third", "fourth"), 1
        ):
            if director_mode == "Image to Video":
                destination = (
                    "the supplied first frame" if index == 1
                    else "the supplied last frame" if index == 2
                    else f"image {index}"
                )
            else:
                destination = f"<Picture {index}>"
            for pattern in (
                rf"@image\s*{index}\b",
                rf"\bimage\s*{index}\b",
                rf"\b(?:the\s+)?{ordinal}\s+(?:reference\s+)?image\b",
            ):
                story_idea = re.sub(
                    pattern, destination, story_idea, flags=re.IGNORECASE
                )
        for index in range(1, 5):
            for pattern in (
                rf"@mood\s*{index}\b",
                rf"\bmood\s*(?:image\s*)?{index}\b",
            ):
                story_idea = re.sub(
                    pattern, f"<Mood {index}>", story_idea,
                    flags=re.IGNORECASE,
                )
        story_idea = re.sub(
            r"@mood\s*(?:video)?\b|\bmood\s+video\b",
            "<Mood Video>", story_idea, flags=re.IGNORECASE,
        )
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
            bypass_record = {
                "plan_json": plan_json,
                "story_bible": "Prompt Assistant bypassed.",
                "synopsis": story_idea,
                "validation": validation,
                "usage_stats": "OpenRouter not used",
                "credits_remaining": "Credits unchanged",
                "scene_prompt": story_idea,
                "mode_prompt": story_idea,
                "source_video_analysis": "",
                "mood_analysis": "",
            }
            # Bypass is commonly used for local wiring/validation checks. It must
            # never replace a paid Director plan stored by Hold.
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
        mood_images = [
            (index, image) for index, image in enumerate(
                (mood_image_1, mood_image_2, mood_image_3, mood_image_4), 1
            ) if image is not None
        ]
        mood_assets_connected = bool(mood_images or mood_video is not None)
        source_video_connected = source_video is not None
        is_edit_mode = director_mode in {"Edit", "Reference Edit"}
        is_video_edit = is_edit_mode and source_video_connected
        is_still_mode = is_edit_mode and not source_video_connected
        is_i2v_mode = director_mode == "Image to Video"
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
                "as `(S1) says: <d>[Language] exact words</d>` using the matching "
                "stable speaker label; never use square brackets and never ask MiniMax to invent "
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
                    "in English. The Director must write every actual spoken line using MiniMax "
                    "`<d>` markup and assign it to one clearly visible speaker using exactly "
                    "`(S1) says: <d>[Language] exact words</d>` with the matching speaker number. "
                    "Never use square brackets "
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
        generic_rules = ""
        if bool(generic_mode):
            if director_mode == "Image to Video":
                generic_rules = """
EXPERIMENTAL GENERIC MODE — REUSABLE I2V PROMPT:
- Build a reusable action and direction template whose start frame can be replaced later without rewriting the prompt.
- Inspect the supplied first frame only enough to choose coarse functional roles such as the man, the woman, the person, the couple, the animal, the object, the vehicle, the product, or the location in the supplied first frame.
- The planning model will not receive the replaceable frame. Never invent or infer its wardrobe, body, face, hair, accessories, props, setting, lighting, composition, or background. If the user does not state a visual fact, omit it completely and let the frame supply it at generation time.
- Do not output `<Picture N>` or `<Subject N>` in native I2V mode. Refer naturally to `the man in the supplied first frame`, `the woman in the supplied first frame`, or another concise coarse role. If a role is ambiguous, use `the person` or `the subject`.
- Do not copy or mention incidental source-frame traits such as facial structure, age, ethnicity, body measurements, hair style or color, clothing design or color, accessories, lighting, camera framing, or background unless the user's written request explicitly requires that detail.
- In action beats, never name individual garments observed in the frame. Use neutral functional words such as `clothes`, `clothing`, or `footwear`. For example, write `removes the woman's clothes`, never `removes her leather jacket, black top, and pants`. This applies even when several garments are visibly distinguishable.
- Do not turn an observed room into a permanent setting. Say `the current location`, `the room`, or omit the setting unless the user explicitly requests a location or the selected genre requires one.
- User-written details remain authoritative. Include requested clothing, transformations, props, locations, actions, relationships, camera behavior, dialogue, sound, genre, motion, and visual look normally.
- Describe motion from the literal supplied first-frame state while leaving appearance resolution to whichever frame is connected when the reusable prompt runs.
""".strip()
            else:
                generic_rules = """
EXPERIMENTAL GENERIC MODE — REUSABLE REFERENCE PROMPTS:
- Build a reusable action and direction template whose connected images can be replaced later without rewriting the plan.
- Inspect each connected Picture only enough to determine its coarse functional role: for example the man, the woman, the person, the couple, the animal, the object, the vehicle, the product, or the location from <Picture N>.
- The planning model will receive only replaceable Picture slots, not their pixels. Never invent or infer wardrobe, body, face, hair, accessories, props, setting, lighting, composition, or background. If the user does not state a visual fact, omit it completely and let the connected Picture supply it at generation time.
- Ground reusable identities with concise definitions such as `<Subject 1> is the man from <Picture 1>`. If gender or role is genuinely ambiguous, use `the person` or `the subject`; never guess a detailed identity.
- Do not copy or mention incidental source-image traits such as facial structure, age, ethnicity, body measurements, hair style or color, clothing design or color, accessories, pose, expression, lighting, camera framing, or background unless the user's written request explicitly requires that detail.
- In action beats, never name individual garments observed in a Picture. Use neutral functional words such as `clothes`, `clothing`, or `footwear`; for example `removes <Subject 2>'s clothes`, not a list of jacket, top, pants, colours, or materials.
- Do not turn an observed room into a permanent setting. Say `the current location`, `the room`, or omit the setting unless the user explicitly requests a location or the selected genre requires one.
- User-written details remain authoritative. Include requested clothing, transformations, props, locations, actions, relationships, camera behavior, dialogue, sound, genre, motion, and visual look normally.
- Scene prompts must refer to visual participants by their generic natural role and `<Subject N>` binding, never by a detailed visual description inferred from an uploaded image.
- Preserve role assignment and action continuity across scenes, but leave appearance resolution to whichever Picture is connected when the reusable plan runs.
""".strip()
        visual_look_direction = (
            f"Visual Look: {visual_look}. {VISUAL_LOOKS[visual_look]}"
        )
        if visual_look not in {"Auto", "Cinematic"}:
            visual_look_direction += (
                " Do not reinterpret this selection as cinematic. Avoid generic cinematic "
                "atmosphere, sweeping dolly language, artificial shallow depth of field, "
                "dramatic color grading and polished studio lighting unless the selected "
                "primary or secondary look explicitly requires one of those properties."
            )
        if secondary_visual_look == "Auto":
            visual_look_direction += (
                " Infer one distinct compatible secondary visual treatment and apply it only "
                "as a restrained color, texture, processing, or finishing accent."
            )
        elif secondary_visual_look != "None":
            visual_look_direction += (
                f" Secondary Visual Look: {secondary_visual_look}. "
                f"{VISUAL_LOOKS[secondary_visual_look]} Blend only compatible finishing traits; "
                "the primary look controls capture medium, viewpoint, and base image behavior."
            )
        profile_rules = ""
        style_contract = _director_style_contract(
            genre,
            secondary_genre,
            motion_style,
            secondary_motion_style,
            visual_look,
            secondary_visual_look,
            compact=bool(split_global or toolkit_prompt_rules),
        )
        if director_profile == "Gemma":
            profile_genre_rule = (
                "- Infer one primary genre from the request and references, then execute it in "
                "every scene through concrete action, performance, setting, camera and sound."
                if auto_genre else
                f"- Execute primary genre `{genre}` in every scene through concrete action, "
                "performance, setting, camera and sound; never treat it as a decorative label."
            )
            if secondary_genre == "Auto" and not auto_genre:
                profile_secondary_rule = (
                    "- Infer one compatible secondary genre and use it only as a supporting layer; "
                    "never let it erase the primary genre."
                )
            elif secondary_enabled:
                profile_secondary_rule = (
                    f"- Execute secondary genre `{secondary_genre}` only as a compatible layer and "
                    "never let it erase the primary genre."
                )
            else:
                profile_secondary_rule = (
                    "- No secondary genre is active; do not invent a competing genre identity."
                )
            profile_visual_rule = (
                "- Infer one coherent capture aesthetic and execute its camera, focus, lighting, "
                "exposure, color and texture consistently; never output Auto."
                if visual_look == "Auto" else
                f"- Execute Visual Look `{visual_look}` literally and consistently."
            )
            profile_language_rule = (
                "- Dialogue is disabled: dialogue must be an empty string in every scene."
                if language == "No dialogue" else
                f"- Any dialogue or intelligible lyrics must be exclusively natural idiomatic "
                f"{language}, with no translation, bilingual repetition or language mixing."
            )
            profile_rules = (
                "MANDATORY GEMMA PROFILE:\n"
                f"{profile_genre_rule}\n"
                f"{profile_secondary_rule}\n"
                f"{profile_visual_rule}\n"
                f"{profile_language_rule}\n"
                "- Preserve the user's requested action at the same semantic specificity. Never "
                "downgrade a concrete act into mood, implication, seduction, posing, atmosphere "
                "or a generic interaction; never intensify it beyond what was requested.\n"
                "- Build each action beat as cause -> physical mechanics -> observable result. "
                "Name the acting subject, affected subject or object, direction, contact and "
                "result whenever they are visually relevant.\n"
                "- Complete every private worksheet field with scene-specific evidence. The "
                "coverage check must verify actions, genre, visual look and final state.\n"
                f"{style_contract}"
            )
        else:
            profile_rules = (
                "MANDATORY OPENROUTER PROFILE:\n"
                "- Treat every explicit user-specified visible attribute as authoritative: "
                "wardrobe, colors, garment type and fit, exposed or covered body areas, hair, "
                "accessories, footwear, props and location. These instructions override "
                "conflicting mutable details visible in a reference image.\n"
                "- Preserve every requested action at the same semantic specificity. Enrich "
                "camera, staging, performance, lighting and sound, but never replace or escalate "
                "the action itself. A kiss remains a kiss; clothing remains worn; covered areas "
                "remain covered unless the user explicitly requests otherwise. Do not introduce "
                "nudity, exposed anatomy or a more explicit form of contact by inference.\n"
                "- Never compress an explicit attribute into vague phrases such as `preserve "
                "wardrobe`, `same outfit`, `as described`, or `keep appearance`. State the exact "
                "requested visible details in prompt_prefix when they apply to every scene and "
                "place them once in prompt_prefix when globally persistent. Do not repeat those "
                "unchanged details in every scene; restate only scene-scoped or newly changed facts.\n"
                "- Keep the complete cast and future events in the private story_bible, but put "
                "only Subjects visible in every scene inside prompt_prefix. A character who "
                "enters later must not be named, described, tagged, implied in the background, "
                "or grounded in a Picture inside prompt_prefix. Define that character only in "
                "the first scene where they become visible and in later scenes where present.\n"
                "- Every scene prompt must name exactly the Subjects actually visible in that "
                "scene. Omit future, absent, departed and off-screen characters completely; do "
                "not mention them even in a negative instruction.\n"
                "- Preserve mutable continuity from the latest completed scene, not from an "
                "obsolete reference state. Never restore replaced clothing, props, hairstyle, "
                "location or physical state merely because it appears in <Picture N>.\n"
                "- Translate every requested event into literal visible action with an opening "
                "state, physically achievable progression and unmistakable final state.\n"
                f"{style_contract}"
            )
        motion_brief = (
            f"Visual energy / pose style: {motion_style}. Translate this into pose, "
            "composition and implied energy without describing temporal movement. "
            + (
                "Infer one compatible secondary pose or compositional energy."
                if secondary_motion_style == "Auto" else
                (
                    f"Secondary motion style: {secondary_motion_style}. Translate only its "
                    "compatible energy into pose and composition, not temporal movement."
                    if secondary_motion_style != "None" else ""
                )
            )
            if is_still_mode else
            f"Motion style: {motion_style}. {motion_direction}{secondary_motion_direction}"
        )
        duration_brief = (
            "Each prompt describes exactly one finished still image."
            if is_still_mode else
            f"Each scene will be generated for approximately "
            f"{float(scene_duration_seconds):g} seconds. "
            f"Write each scene as a compact {_scene_prompt_budget(scene_duration_seconds)[0]}-"
            f"{_scene_prompt_budget(scene_duration_seconds)[1]} word production brief with no "
            f"more than {_scene_prompt_budget(scene_duration_seconds)[2]} main action "
            f"beat{'s' if _scene_prompt_budget(scene_duration_seconds)[2] != 1 else ''}. "
            + (
                "Put the current identity, wardrobe, setting, look, camera and sound state "
                "inside that scene's scene_global; its prompt contains the visible action and "
                "result occurring during that screen time."
                if bool(split_global) else
                "Put invariant identity, wardrobe, setting, look and sound rules in prompt_prefix "
                "once; scene prompts contain only the opening state, changes, camera/sound events "
                "and final state that occur during their own screen time."
            )
        )
        compact_state_direction = ""
        if (not is_still_mode and director_mode != "Image to Video"
                and director_profile != "Gemma"):
            compact_state_direction = (
                "\n\nCOMPACT INTERNAL SCENE STATE:\n"
                "- Fill compact_state before writing each scene prompt. Keep opening and closing "
                "short, literal and limited to mutable physical facts required for continuity.\n"
                "- For Continuous Story, scene N opening inherits scene N-1 closing exactly in "
                "meaning. Preserve completed wardrobe, exposure, prop, pose, contact and location "
                "changes; never reset them from a reference image.\n"
                "- Put chronological visible mechanics, materially useful camera and lighting, "
                "synchronized sound and correctly attributed dialogue in compact_state.action.\n"
                "- Do not print field names, continuity terminology, checks or explanations. The "
                "compiler converts these private fields into one ordinary H3 scene prompt."
            )
        toolkit_direction = ""
        if bool(toolkit_prompt_rules):
            toolkit_direction = (
                "\n\nEXPERIMENTAL H3 TOOLKIT PROMPT CONTRACT:\n"
                "- The rendered prompt for each scene is prompt_prefix plus that scene prompt; "
                "together they must be fully self-contained without relying on any previous prompt.\n"
                "- State the exact visible opening state and current location. Avoid relative wording "
                "such as `the same room`, `continue the previous shot`, `as before`, or `still there`.\n"
                "- Describe what the camera positively does. Never name an unwanted failure with "
                "phrases such as `no cuts`, `never cut`, `no camera movement`, or `no angle change`.\n"
                "- Bind retained identity and mutable appearance to <Subject N>. Use <Picture N> only "
                "as the visual source defining that Subject, never as the retained entity.\n"
                "- Fill visible_cast with every physical person actually visible in that scene, "
                "including unnamed or background people, exactly once. One Subject means one body; "
                "a reflection, screen image or body part is not an additional cast member.\n"
                "- Each independent scene states every changing visual state required at its first "
                "frame; prompt_prefix supplies invariant identity, wardrobe, environment and look."
            )
        split_direction = ""
        if bool(split_global):
            split_direction = (
                "\n\nEXPERIMENTAL SPLIT GLOBAL CONTRACT:\n"
                "- Return prompt_prefix as an empty string. Every scene owns its complete "
                "positive scene_global and receives only that state plus its action prompt.\n"
                "- scene_global must state the exact visible cast and Subject/Picture bindings, "
                "current wardrobe or exposure, exact location/geography, active props, selected "
                "genre format, motion treatment, visual look, lighting, camera baseline and "
                "current music/ambience phase.\n"
                "- Scene 1 establishes the requested initial state. Every later scene_global "
                "inherits the latest completed physical state from the preceding scene, then "
                "updates only facts that have visibly changed.\n"
                "- Keep future characters, locations, props, wardrobe changes and events out of "
                "earlier scene_global values. A person appears only in scenes whose visible_cast "
                "lists that physical individual.\n"
                "- Use direct renderable facts only. Do not print continuity terminology, source "
                "alternatives, negative prompts, validation rules or explanations.\n"
                "- Split Global is independent from Toolkit. When Toolkit is enabled, also obey "
                "its prompt-form and cast-cardinality rules."
            )
        compact_direction = ""
        if bool(compact_mode):
            compact_direction = (
                "\n\nCOMPACT OUTPUT MODE — TOKEN-EFFICIENT PRODUCTION PLAN:\n"
                "- Return the complete required JSON schema, but use concise direct English. "
                "Never omit a requested scene, required field, explicit user action, active "
                "reference, continuity change, dialogue line, sound cue or final state.\n"
                "- Keep synopsis to one or two short sentences and story_bible to only the "
                "stable cast, identity bindings and continuity facts needed by later scenes.\n"
                "- State invariant production facts once in prompt_prefix or scene_global. "
                "Do not repeat genre labels, wardrobe, location, lighting, camera baseline or "
                "reference definitions inside every scene unless they change there.\n"
                "- Each scene is one compact chronological production paragraph: opening "
                "state, visible action mechanics, essential camera/audio/dialogue, and concrete "
                "closing state. Remove commentary, rationale, coverage prose and synonyms.\n"
                "- Prefer specific verbs and nouns over decorative adjectives. Compact means "
                "less repetition, never less semantic fidelity or an unfinished action."
            )
        power_direction = ""
        if bool(power_prompt_rules):
            power_mode_behavior = (
                "Treat all scenes as invisible time partitions of one uninterrupted audiovisual "
                "take. Scene N entry_state must be copied verbatim from scene N-1 exit_state. "
                "Carry pose, contact, wardrobe/exposure, props, geography, screen direction, "
                "camera axis and velocity, lighting, action phase, music phase and ambience "
                "without resets."
                if director_mode == "Continuous Story" else
                "Treat each scene as a deliberate new camera setup while preserving the latest "
                "world state, identity, wardrobe/exposure, props, relationships and causal story "
                "facts. A cut may change framing, angle, lens, axis or location only when that "
                "change adds visible narrative information."
                if director_mode == "Cinematic Cuts" else
                "Honor the selected edit/keyframe mode exactly. Separate source-plate properties "
                "that stay authoritative from the explicit requested delta, and never let an "
                "auxiliary reference overwrite unrelated source motion, composition or identity."
            )
            power_direction = (
                "\n\nEXPERIMENTAL POWER DIRECTOR CONTRACT:\n"
                "- First complete power_blueprint privately. Convert every explicit user request "
                "into one coverage_map item assigned to the exact scene that visibly executes it; "
                "mood never substitutes for an action, transformation, dialogue or ending.\n"
                "- Copy the user's requested visible actions and state changes into user_action_spine "
                "in their original order before adding creative detail. This spine is authoritative: "
                "do not replace, soften, summarize, reorder or perform the same beat again in a later "
                "scene. Distribute it across scenes as one advancing sequence. When more scenes are "
                "requested than the user supplied beats, invent a compatible next consequence or "
                "resolution instead of repeating the preceding action.\n"
                "- Give each connected reference one primary production job and a scene scope. "
                "Define reusable visible identity as <Subject N> grounded in <Picture N>. Use "
                "<Picture N> for the concrete file and (Sx) only for a vocal source. State what "
                "each asset contributes and which unrelated identity, wardrobe, pose, setting, "
                "lighting, motion, camera or audio traits must not transfer.\n"
                "- Distinguish SOURCE IDENTITY from TARGET DESIGN before writing prompts. When the "
                "user requests a new universe, setting, look, style, wardrobe, outfit or appearance, "
                "the connected Picture supplies identity only unless the user explicitly preserves "
                "another trait. Discard conflicting source clothing, environment, lighting, pose, "
                "composition and capture style. Never copy those source traits into "
                "persistent_visual_overrides.\n"
                "- Before designing, fill source_visual_inventory privately with the visible source "
                "wardrobe, environment, lighting, pose and capture style. Then fill design_change_map "
                "with explicit `source trait -> newly invented target trait` replacements for the "
                "world and every affected subject. The two sides must be visibly different; merely "
                "renaming, re-describing or declaring the source clothes to be the target costume is "
                "a failed redesign. Never expose source_visual_inventory in rendered prompts.\n"
                "- Translate every requested named universe, period or aesthetic into a concrete "
                "target_world_design: architecture, landscape/geography, materials, set dressing, "
                "lighting logic, palette, rendering/capture medium and atmosphere. A franchise or "
                "style name alone is not a design and must never appear as the only environment rule.\n"
                "- When new wardrobe is requested, target_subject_designs must define every affected "
                "subject separately with exact garment pieces, silhouette, layers, materials, colors, "
                "footwear and accessories appropriate to the target world. These designs replace the "
                "reference wardrobe from the first rendered instant and remain authoritative until a "
                "later visible wardrobe change. Never write `current wardrobe`, `same clothes`, or "
                "`preserve wardrobe` in place of the actual new design.\n"
                "- Keep complete cast knowledge in the private blueprint, but expose a Subject or "
                "reference in a rendered scene only while it is actually visible or audible. Future "
                "characters, locations, props and events must not leak into earlier prompts.\n"
                "- Build each scene from entry_state -> chronological cause/effect timeline_beats -> "
                "exit_state. Fit one dominant action beat per roughly 1-3 seconds and prefer fewer "
                "fully readable actions over compressed choreography. Every action names actor, "
                "affected subject/object, physical mechanics, direction/contact where relevant, "
                "observable result and the resulting body/object state.\n"
                "- Use chronological conjunctions such as `as`, `while`, and `then` to distinguish "
                "simultaneous motion from sequential action. Prefer ordinary sentences and commas; "
                "avoid semicolon-heavy instruction chains that fragment model attention.\n"
                f"- {power_mode_behavior}\n"
                "- Direct composition, subject position, foreground/midground/background geography, "
                "camera path, amplitude and speed in natural positive prose. Preserve eyelines and "
                "screen direction. Use a cut only for new narrative information, never as decoration.\n"
                "- Treat user-requested wardrobe, appearance and prop changes as mutable state. Once "
                "changed on screen, the newest completed state supersedes the obsolete Picture state "
                "until another visible change occurs.\n"
                "- Use official speech form `<Subject N> (S1) says: <d>[Language] exact words</d>` "
                "when a referenced subject speaks. Keep delivery outside <d>, keep speaker IDs stable, "
                "show readable mouth movement, and fit the line plus a natural breath/pause inside the "
                "available duration. Do not repeat dialogue or translate it.\n"
                "- Put synchronized physical sound with its visible event. Keep ambience/non-verbal "
                "sound conceptually separate from audience-only non-diegetic music; preserve their "
                "phase and level across Continuous Story boundaries.\n"
                "- Make prompt_prefix compact and invariant. Each scene prompt must be a literal, "
                "self-contained production instruction, not a plot summary, adjective stack, worksheet, "
                "negative-prompt list or explanation. End every scene in a concrete renderable state; "
                "the final scene visibly resolves the user's promised outcome.\n"
                "- RENDERED POWER PROMPT STYLE: keep the complex reasoning in power_blueprint and "
                "write the final H3 prompt as compact chronological prose. In the first sentence, "
                "establish the concrete location and bind each visible identity once as `<Subject N> "
                "grounded in <Picture N>`, including its target wardrobe only on first appearance or "
                "when that wardrobe changes. Then describe only what visibly happens, in physical order, "
                "using short direct sentences joined by `as`, `while`, and `then`. State the resulting "
                "pose, exposure, prop and location state exactly where it changes so the next scene can "
                "inherit it. Do not repeat static clothing, scenery, style, consent, camera, lighting or "
                "music in every paragraph when prompt_prefix already establishes it. Add camera and sound "
                "only when requested or materially useful. Never print word counts, planning commentary, "
                "contracts, checks, summaries, labels such as opening state, or ornamental filler.\n"
                "- power_check must silently verify user-intent coverage, exact reference ownership, "
                "duration feasibility, continuity/cut logic, physical coherence, camera readability, "
                "dialogue markup, sound separation and final closure before returning JSON."
            )
        content = [{
            "type": "text",
            "text": (
                f"Create exactly {scene_count} "
                f"{'production prompt' if scene_count == 1 else 'production prompts'}. "
                f"Director mode: {director_mode}. "
                f"{genre_direction} "
                f"{motion_brief} "
                f"{visual_look_direction} "
                "Write the synopsis, story bible, prompt prefix, and every scene "
                "prompt entirely in English. "
                f"Dialogue direction: {dialogue_direction} "
                "Keep MiniMax tags such as <Picture 1> unchanged. "
                f"FINAL SCENE CONTRACT: Scene {scene_count} is the actual ending, "
                "not another transition. It must visibly complete every remaining "
                "action and outcome explicitly requested for the ending, then hold "
                "a resolved physical state for a final beat while camera and sound "
                "settle. Never end the final scene with a newly started, anticipated "
                "or half-completed action. For an intentionally ongoing repeated "
                "action, sustain it and finish on a stable natural beat without "
                "inventing a new event. Non-final scenes may remain open for their "
                "successor; the final scene may not. "
                f"{duration_brief}\n\n"
                f"{story_direction}\n\n"
                f"{reference_alias_rules}\n\n"
                f"Additional direction:\n{str(additional_direction or '').strip()}"
                f"{adult_direction}\n\n"
                f"{style_contract}"
                f"{toolkit_direction}"
                f"{compact_state_direction}"
            ),
        }]
        for index, image in enumerate(pictures, 1):
            if bool(generic_mode):
                # Generic Mode must be reusable with a different image. Sending
                # pixels to the planner inevitably leaks wardrobe, location and
                # lighting into the supposedly generic prompt, so expose only a
                # replaceable slot. MiniMax still receives the real image later.
                if director_mode != "Image to Video":
                    content.append({
                        "type": "text",
                        "text": (
                            f"A replaceable visual slot exists as <Picture {index}>. "
                            "Use the exact tag, but do not infer any unseen visual detail."
                        ),
                    })
                continue
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

        mood_rules = ""
        if mood_assets_connected:
            mood_rules = (
                "PRIVATE MOOD-BOARD CONTRACT:\n"
                "- <Mood 1> through <Mood 4> and <Mood Video> are visible only to the "
                "Director. They are not MiniMax references and must never appear in "
                "prompt_prefix, scene_global, a scene prompt, synopsis or story bible.\n"
                "- Resolve every requested mood attribute into explicit natural English "
                "generation prose. Never write `from the mood image`, `as in the reference`, "
                "an @mood alias or a <Mood ...> tag in generation text.\n"
                "- Mood assets never transfer facial identity, body identity or character "
                "ownership. Production identity remains exclusively attached to <Picture N>.\n"
                "- Borrow only the attribute the user requests. An outfit request borrows "
                "garment design, materials, colors and fit; a pose request borrows body "
                "arrangement; an action/video request borrows chronological motion and camera.\n"
                "- Record the resolved borrowing decisions in mood_analysis."
            )
            for mood_index, mood_image in mood_images:
                content.append({
                    "type": "text",
                    "text": (
                        f"The next private Director-only mood reference is <Mood {mood_index}>. "
                        "Inspect it only for attributes explicitly assigned by the user. It "
                        "does not define identity and its tag cannot enter generation prompts."
                    ),
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(mood_image, min(
                            int(image_max_dimension), 640
                        ))
                    },
                })
            if mood_video is not None:
                mood_samples, mood_video_note = _frame_batch_sample_data_urls(
                    mood_video,
                    min(int(video_sample_frames), 10),
                    min(int(image_max_dimension), 640),
                )
                content.append({
                    "type": "text",
                    "text": (
                        "The following chronological observations form private <Mood Video>. "
                        "Infer only user-requested action, pose progression, interaction, camera "
                        "movement, rhythm or staging. Do not copy identity automatically and do "
                        "not expose this tag or the observation method in generation text. "
                        f"{mood_video_note}"
                    ),
                })
                for order, (_frame_index, percentage, image_url) in enumerate(
                    mood_samples, 1
                ):
                    content.append({
                        "type": "text",
                        "text": (
                            f"Private mood-video observation {order}/{len(mood_samples)} "
                            f"({percentage:.1f}% through its motion)."
                        ),
                    })
                    content.append({
                        "type": "image_url", "image_url": {"url": image_url},
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

        # Optional metadata must have a stable value in every execution route.
        # The direct-prompt variant intentionally skips the JSON story object
        # from which the standard Director normally reads mood_analysis.
        mood_analysis = ""

        payload = {
            "model": str(model or DEFAULT_MODEL).strip(),
            "messages": [
                {
                    "role": "system",
                    "content": "\n\n".join((
                        str(system_prompt or "").strip(),
                        mode_rules,
                        profile_rules,
                        generic_rules,
                        split_direction,
                        compact_direction,
                        power_direction,
                        mood_rules,
                    )).strip(),
                },
                {"role": "user", "content": content},
            ],
            "max_tokens": (
                min(
                    int(max_tokens),
                    max(2200, 1800 + (int(scene_count) * 550)),
                )
                if bool(compact_mode) else int(max_tokens)
            ),
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
                        dialogue_language=language,
                        primary_genre=genre,
                        secondary_genre=secondary_genre,
                        primary_motion_style=motion_style,
                        secondary_motion_style=secondary_motion_style,
                        visual_look=visual_look,
                        secondary_visual_look=secondary_visual_look,
                        toolkit_prompt_rules=bool(toolkit_prompt_rules),
                        split_global=bool(split_global),
                        power_prompt_rules=bool(power_prompt_rules),
                        mood_assets_connected=mood_assets_connected,
                    ),
                },
            },
            "provider": {"require_parameters": True},
        }
        if bool(getattr(cls, "PROMPT_LIST_MODE", False)):
            duration = float(scene_duration_seconds)
            if duration <= 6.0:
                action_budget = (
                    "one primary action with one immediately visible reaction; "
                    "do not begin a second major action"
                )
            elif duration <= 10.0:
                action_budget = (
                    "at most two causally connected action phases with visible completion"
                )
            else:
                action_budget = (
                    "at most three causally connected action phases with readable transitions"
                )
            if pictures:
                required_prompt_structure = """
Each prompt must use these exact MiniMax H3 full-reference fields in this order:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

Keep subject_definitions compact: one stable <Subject N> line per visible reusable
entity, citing the source <Picture N>. In retention_analysis use only the official
visual relationship values fully_preserved, partially_preserved, attribute_transfer,
or weak_reference. Use <Picture N> as a concrete frame/composition anchor only when
it truly serves that role; visual identity belongs to <Subject N>.
""".strip()
            else:
                required_prompt_structure = """
Each prompt must use these exact MiniMax H3 base fields in this order:
integrated_multimodal_description:
overall_soundscape:
non_diegetic_music:
""".strip()
            direct_contract = f"""
DIRECT PROMPT LIST OUTPUT MODE:
- Direct the same production with the selected mode, references, genre, motion,
  visual look, dialogue, audio, mood assets and user instructions.
- Return exactly {int(scene_count)} complete MiniMax H3 generation prompts.
- Return only one JSON object with one key named scene_prompts. Its value must be
  an array containing exactly {int(scene_count)} complete prompt strings.
- Do not add commentary, a synopsis, story bible, scene numbers, durations, seeds
  or technical plan metadata inside those strings.
- Inside every prompt follow this required structure exactly:
  {required_prompt_structure}
- Make every prompt self-contained: establish currently visible subjects and exact
  <Picture N>/<Subject N> assignments, inherited mutable state, chronological visible
  action, camera, lighting, synchronized sound and permitted dialogue.
- A speaking referenced character must be written as `<Subject N> (Sx) says:
  <d>[Language] exact words</d>`. Keep the same speaker ID across scenes. Put only
  the language and literal spoken words inside <d>; never write `(Sx) says` without
  first identifying the visible subject.
- Put ambience, physical effects and non-verbal vocal sounds only in
  overall_soundscape. Put audience-only score only in non_diegetic_music; use N/A
  when no score is requested. Do not duplicate these layers in detailed_description.
- Do not repeat a heavy shared global block. Carry only state needed by that scene.
  Never introduce a future person, costume, prop or location early.
- Continuous Story prompts are consecutive partitions of one uninterrupted take:
  prompt N+1 begins at the exact physical and audiovisual state where N ends.
  Cinematic Cuts may change shot setup while preserving current narrative state.
- For Continuous Story scene 2 and later, treat the previous scene's final state as
  the opening state carried by masked_av. Use <Picture N> only to preserve immutable
  identity; never reset pose, framing, wardrobe, nudity, prop state, location, lighting
  phase or subject placement back to the original Picture. State current inherited
  mutable details directly and continue the unfinished motion without replaying it.
- Across a Continuous Story boundary, preserve screen direction, contact points,
  body orientation, camera velocity, ambient room tone, music phase and active speech.
  Do not announce a new shot, repeat an establishing description, restart music, or
  make subjects re-enter unless the user explicitly requests that event.
- For approximately {duration:g} seconds, permit {action_budget}. Prefer complete,
  observable motion over an overloaded plot summary. Finish on one concrete visible
  and audible state that the next scene can inherit.
- Populate scene_prompts in chronological order and return no keys beyond it.
""".strip()
            payload["messages"][0]["content"] = "\n\n".join((
                str(payload["messages"][0]["content"]).strip(), direct_contract,
            ))
            if isinstance(payload["messages"][1].get("content"), list):
                payload["messages"][1]["content"].append({
                    "type": "text",
                    "text": (
                        f"Final output reminder: populate scene_prompts with exactly "
                        f"{int(scene_count)} complete chronological prompt strings."
                    ),
                })
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "minimax_h3_direct_scene_prompts",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "scene_prompts": {
                                "type": "array",
                                "minItems": int(scene_count),
                                "maxItems": int(scene_count),
                                "items": {"type": "string", "minLength": 1},
                            },
                        },
                        "required": ["scene_prompts"],
                        "additionalProperties": False,
                    },
                },
            }
            payload["provider"] = {"require_parameters": True}
        result = (
            _external_llm_request(llm_model, payload)
            if uses_external_llm else
            _openrouter_request(api_key, payload, int(timeout_seconds))
        )
        try:
            content_text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("OpenRouter returned an unexpected response.") from error
        if bool(getattr(cls, "PROMPT_LIST_MODE", False)):
            try:
                direct_prompts = _parse_direct_prompt_response(
                    content_text, scene_count
                )
            except RuntimeError as first_error:
                print(
                    "[H3 Direct Prompt Director] Scene-block mismatch detected; "
                    "requesting one automatic format repair."
                )
                repair_payload = {
                    **payload,
                    "temperature": min(float(temperature), 0.2),
                    "messages": [
                        *payload["messages"],
                        {"role": "assistant", "content": content_text},
                        {
                            "role": "user",
                            "content": (
                                f"Return a COMPLETE replacement matching the required "
                                f"scene_prompts JSON schema with exactly {int(scene_count)} "
                                "complete chronological prompt strings. Preserve the "
                                "intended content and complete every missing scene."
                            ),
                        },
                    ],
                }
                repair_result = (
                    _external_llm_request(llm_model, repair_payload)
                    if uses_external_llm else
                    _openrouter_request(
                        api_key, repair_payload, int(timeout_seconds)
                    )
                )
                try:
                    repaired_text = repair_result["choices"][0]["message"]["content"]
                    direct_prompts = _parse_direct_prompt_response(
                        repaired_text, scene_count
                    )
                    result = repair_result
                    content_text = repaired_text
                except (KeyError, IndexError, TypeError, RuntimeError) as second_error:
                    raise RuntimeError(
                        "Direct Prompt Director could not obtain the requested "
                        f"{int(scene_count)} complete scene blocks after one automatic "
                        f"format repair. First response: {first_error} Repair response: "
                        f"{second_error}"
                    ) from second_error
            usage = result.get("usage") or {}
            usage_stats = (
                f"input: {usage.get('prompt_tokens', '?')} · "
                f"output: {usage.get('completion_tokens', '?')} · "
                f"total: {usage.get('total_tokens', '?')}"
                if usage else "External LLM · usage not reported"
            )
            credits = (
                "External LLM · credits not available" if uses_external_llm else
                _credits(api_key, min(30, int(timeout_seconds)))
            )
            validation = (
                f"Direct prompts ready · {len(direct_prompts)} scenes · "
                f"{director_mode} · profile {director_profile} · no JSON plan"
            )
            preview = "\n\n".join(
                f"SCENE {index:02d}\n{prompt}"
                for index, prompt in enumerate(direct_prompts, 1)
            )
            prompt_lines = _format_direct_chain_payload(
                direct_prompts, scene_duration_seconds, steps, seed,
                director_mode,
            )
            _set_held_director_plan(hold_cache_key, {
                "cache_kind": "direct_scene_prompts",
                "prompt_lines": prompt_lines,
                "first_prompt": direct_prompts[0],
                "prompt_count": len(direct_prompts),
                "validation": validation,
                "usage_stats": usage_stats,
                "credits_remaining": credits,
                "mood_analysis": mood_analysis,
                "preview": preview,
            })
            return io.NodeOutput(
                prompt_lines,
                direct_prompts[0], len(direct_prompts),
                validation, usage_stats, credits, mood_analysis,
                ui=ui.PreviewText(preview),
            )
        raw_story = _parse_json_response(content_text)
        if bool(power_prompt_rules):
            power_issues = _power_plan_issues(
                raw_story, scene_count, director_mode, is_still_mode, story_idea
            )
            if power_issues:
                concise_issues = "; ".join(power_issues[:16])
                print(
                    "[H3 Story Director] Power validation found an inconsistent production "
                    "plan; requesting one automatic full repair: " + concise_issues
                )
                correction_payload = {
                    **payload,
                    "temperature": min(float(temperature), 0.25),
                    "messages": [
                        *payload["messages"],
                        {"role": "assistant", "content": content_text},
                        {
                            "role": "user",
                            "content": (
                                "Return a COMPLETE replacement JSON plan from the beginning; do "
                                "not patch, explain, abbreviate or continue the previous response. "
                                "The Power production audit rejected it for these exact reasons: "
                                + concise_issues + ". Preserve every explicit user request, selected "
                                "mode, genre, motion, look, dialogue setting, reference assignment, "
                                "scene count and duration. Rebuild power_blueprint first, then every "
                                "scene. In Continuous Story, copy each preceding exit_state verbatim "
                                "as the next entry_state. Make timeline beats physically achievable, "
                                "make the rendered prompt chronological and production-ready, scope "
                                "references only where active, and finish with a concrete exit state."
                            ),
                        },
                    ],
                }
                corrected_result = (
                    _external_llm_request(llm_model, correction_payload)
                    if uses_external_llm else
                    _openrouter_request(api_key, correction_payload, int(timeout_seconds))
                )
                try:
                    corrected_text = corrected_result["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as error:
                    raise RuntimeError(
                        "Power Director received an unexpected response during automatic repair."
                    ) from error
                corrected_story = _parse_json_response(corrected_text)
                remaining = _power_plan_issues(
                    corrected_story, scene_count, director_mode, is_still_mode,
                    story_idea,
                )
                if remaining:
                    raise RuntimeError(
                        "Power Director rejected two inconsistent production plans to avoid "
                        "wasting a video generation. Remaining problems: "
                        + "; ".join(remaining[:16])
                    )
                raw_story = corrected_story
                result = corrected_result
        if director_profile == "Gemma" and not is_still_mode and not is_i2v_mode:
            worksheet_issues = _gemma_scene_issues(raw_story, scene_count)
            if worksheet_issues:
                concise_issues = "; ".join(worksheet_issues[:12])
                print(
                    "[H3 Story Director] Gemma returned an incomplete scene plan; "
                    "requesting one complete automatic correction: " + concise_issues
                )
                correction_payload = {
                    **payload,
                    "temperature": min(float(temperature), 0.25),
                    "messages": [
                        *payload["messages"],
                        {"role": "assistant", "content": content_text},
                        {
                            "role": "user",
                            "content": (
                                "Rewrite the COMPLETE JSON plan from the beginning. "
                                "Do not patch or continue the previous text. The previous "
                                "plan is invalid because: " + concise_issues + ". "
                                "For every scene, return one complete `prompt` written as "
                                "natural production prose with at least three complete "
                                "sentences covering visible action, physical interaction, "
                                "camera/framing, environment/lighting, sound, and the final "
                                "visible state. Never place JSON keys, schema field names, "
                                "underscored placeholders, dangling quotes, translations, "
                                "or partial fragments inside a prompt. Visual characters "
                                "must use natural role phrases and the correct <Picture N> "
                                "when referenced; reserve (S1), (S2), etc. only for spoken "
                                "dialogue. Preserve the user's exact requested actions, cast, "
                                "genre, language, and scene count."
                            ),
                        },
                    ],
                }
                corrected_result = (
                    _external_llm_request(llm_model, correction_payload)
                    if uses_external_llm else
                    _openrouter_request(
                        api_key, correction_payload, int(timeout_seconds)
                    )
                )
                try:
                    corrected_text = corrected_result["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as error:
                    raise RuntimeError(
                        "Gemma returned an unexpected response while correcting its plan."
                    ) from error
                corrected_story = _parse_json_response(corrected_text)
                second_issues = _gemma_scene_issues(corrected_story, scene_count)
                if second_issues:
                    raise RuntimeError(
                        "Gemma produced incomplete scene plans twice; video generation "
                        "was stopped to avoid wasting time. Remaining problems: "
                        + "; ".join(second_issues[:12])
                    )
                raw_story = corrected_story
                result = corrected_result
        if bool(power_prompt_rules):
            _apply_power_target_design(raw_story, story_idea)
        mood_analysis = str(raw_story.get("mood_analysis") or "").strip()
        if mood_assets_connected:
            generation_mood_text = "\n".join((
                str(raw_story.get("prompt_prefix") or ""),
                str(raw_story.get("story_bible") or ""),
                str(raw_story.get("synopsis") or ""),
                *("\n".join((
                    str(shot.get("scene_global") or ""),
                    str(shot.get("prompt") or ""),
                )) for shot in (raw_story.get("shots") or [])
                  if isinstance(shot, dict)),
            ))
            leaked = re.search(
                r"(?:@mood\s*(?:video|[1-4])|<Mood\s+(?:Video|[1-4])>)",
                generation_mood_text, flags=re.IGNORECASE,
            )
            if leaked:
                raise RuntimeError(
                    "The Director left a private mood-board tag in generation text "
                    f"({leaked.group(0)}). Retry so it resolves the reference into prose."
                )
        if style_contract and bool(split_global):
            # The deterministic user controls remain authoritative, but in
            # Split Global they belong to each scene rather than one dominant
            # shared prefix.
            for shot in raw_story.get("shots") or []:
                if not isinstance(shot, dict):
                    continue
                current = str(shot.get("scene_global") or "").strip()
                shot["scene_global"] = "\n".join(
                    part for part in (style_contract, current) if part
                )
            raw_story["prompt_prefix"] = ""
        elif style_contract:
            # Keep the selected format authoritative for MiniMax as well as for
            # either planning backend. This deterministic prefix prevents model
            # preferences from silently replacing the user's controls.
            existing_prefix = str(raw_story.get("prompt_prefix") or "").strip()
            raw_story["prompt_prefix"] = "\n\n".join(
                part for part in (style_contract, existing_prefix) if part
            )
        if is_video_edit:
            analysis = str(raw_story.get("source_video_analysis") or "").strip()
            if len(analysis) < 500:
                raise RuntimeError(
                    "The source-video analysis was too vague or incomplete. "
                    "The plan was rejected before generation; retry with more "
                    "video samples or a stronger vision model."
                )
            # Some otherwise valid local/API models describe the source correctly
            # but omit the literal MiniMax reference tag from the generation text.
            # Keep this protocol detail deterministic instead of discarding an
            # expensive vision pass and asking the model to regenerate the plan.
            prompt_prefix = str(raw_story.get("prompt_prefix") or "").strip()
            shots = raw_story.get("shots") or []
            generation_text = "\n".join((
                prompt_prefix,
                *(str(shot.get("prompt") or "") for shot in shots
                  if isinstance(shot, dict)),
            ))
            if "<Video 1>" not in generation_text:
                source_rule = (
                    "Use <Video 1> as the locked source plate and temporal blueprint; "
                    "apply only the requested edit while preserving all unrequested "
                    "timing, composition, motion, identity, environment and camera details."
                    if director_mode == "Edit" else
                    "Use <Video 1> as the exact source motion and camera blueprint for "
                    "the requested reference edit while preserving the requested identities "
                    "and transformations consistently."
                )
                raw_story["prompt_prefix"] = "\n\n".join(
                    part for part in (source_rule, prompt_prefix) if part
                )
                print(
                    "[H3 Story Director] Restored the required <Video 1> tag "
                    "deterministically before compiling the plan."
                )
        if bool(generic_mode):
            raw_story["persistent_visual_overrides"] = ""
            generic_style_contract = style_contract.replace(
                "request and references", "user request"
            ).replace("request, and references", "user request")
            if director_mode == "Image to Video":
                reference_contract = (
                    "The supplied first frame is replaceable and exclusively defines all "
                    "identity, anatomy, appearance, clothes, accessories, props, location, "
                    "lighting, composition and background. Do not describe any of those "
                    "visual facts unless the user explicitly wrote them."
                )
            else:
                reference_contract = " ".join(
                    f"<Picture {index}> is a replaceable visual reference slot; it exclusively "
                    "defines its subject's identity, appearance, clothes and visual context."
                    for index in range(1, len(pictures) + 1)
                )
            # Never trust a vision model's shared paragraph in Generic Mode:
            # construct it from user controls and neutral slot contracts only.
            if bool(split_global):
                raw_story["prompt_prefix"] = ""
            else:
                raw_story["prompt_prefix"] = "\n\n".join(
                    part for part in (
                        generic_style_contract,
                        reference_contract,
                        adult_direction.strip(),
                    ) if part
                )
            for shot in raw_story.get("shots") or []:
                if isinstance(shot, dict):
                    shot["prompt"] = _sanitize_generic_reference_observations(
                        shot.get("prompt")
                    )
                    if bool(split_global):
                        shot["scene_global"] = "\n".join(
                            part for part in (
                                reference_contract,
                                str(shot.get("scene_global") or "").strip(),
                                adult_direction.strip(),
                            ) if part
                        )

        if director_mode == "Image to Video":
            def i2v_text(value):
                text = str(value or "")
                text = re.sub(
                    r"<Subject\s+\d+>",
                    "the corresponding subject in the supplied first frame",
                    text,
                    flags=re.IGNORECASE,
                )
                text = re.sub(
                    r"<Picture\s+1>", "the supplied first frame", text,
                    flags=re.IGNORECASE,
                )
                text = re.sub(
                    r"<Picture\s+2>", "the supplied last frame", text,
                    flags=re.IGNORECASE,
                )
                return re.sub(r"\s+", " ", text).strip()

            raw_story["prompt_prefix"] = i2v_text(raw_story.get("prompt_prefix"))
            for shot in raw_story.get("shots") or []:
                if isinstance(shot, dict):
                    shot["prompt"] = i2v_text(shot.get("prompt"))
                    if "scene_global" in shot:
                        shot["scene_global"] = i2v_text(
                            shot.get("scene_global")
                        )

        plan_json, story_bible, synopsis, validation, source_video_analysis = _compile_story(
            raw_story,
            scene_count,
            float(scene_duration_seconds),
            int(steps),
            0 if director_mode == "Image to Video" else len(pictures),
            director_mode,
            director_profile,
            bool(toolkit_prompt_rules),
            language,
            bool(generic_mode),
            bool(power_prompt_rules),
            bool(split_global),
        )
        validation += f" · profile {director_profile}"
        if bool(compact_mode):
            validation += " · Compact"
        if bool(split_global):
            validation += " · Split Global"
        if bool(generic_mode):
            validation += " · generic reusable references"
        compiled_plan = json.loads(plan_json)
        compiled_plan["direction_contract"] = {
            "user_intent": str(story_idea or "").strip(),
            "synopsis": str(synopsis or "").strip(),
            "story_bible": str(story_bible or "").strip(),
            "director_mode": str(director_mode or "").strip(),
            "genre": str(genre or "Auto").strip(),
            "secondary_genre": str(secondary_genre or "None").strip(),
            "language": str(language or "Auto").strip(),
            "motion_style": str(motion_style or "Auto").strip(),
            "secondary_motion_style": str(
                secondary_motion_style or "Auto"
            ).strip(),
            "visual_look": str(visual_look or "Auto").strip(),
            "secondary_visual_look": str(
                secondary_visual_look or "None"
            ).strip(),
            "audio_content": str(audio_content or "Auto").strip(),
            "additional_direction": str(additional_direction or "").strip(),
        }
        plan_json = json.dumps(compiled_plan, ensure_ascii=False, indent=2)
        scene_prompt = "\n\n".join(
            part for part in (
                str(compiled_plan["prompt_prefix"]).strip(),
                str(compiled_plan["shots"][0]["prompt"]).strip(),
            ) if part
        )
        if is_video_edit and "<Video 1>" not in scene_prompt:
            raise RuntimeError(
                "Internal Video Edit normalization failed to preserve <Video 1>."
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
        cache_record = {
            "plan_json": plan_json,
            "story_bible": story_bible,
            "synopsis": synopsis,
            "validation": validation,
            "usage_stats": usage_stats,
            "credits_remaining": credits,
            "scene_prompt": scene_prompt,
            "mode_prompt": scene_prompt,
            "source_video_analysis": source_video_analysis,
            "mood_analysis": mood_analysis,
        }
        _set_held_director_plan(hold_cache_key, cache_record)
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
            mood_analysis,
            ui=ui.PreviewText(preview),
        )


class H3DirectPromptDirector(H3StoryDirector):
    """Schema-free Director that returns one tagged prompt block per scene."""

    PROMPT_LIST_MODE = True

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3DirectPromptDirector"
        schema.display_name = "H3 Director — Direct Scene Prompts (Experimental)"
        schema.search_aliases = [
            "h3 prompt list director", "direct scene prompts", "no json director",
        ]
        schema.description = (
            "Experimental multimodal Director that keeps the normal creative and "
            "continuity reasoning but returns one tagged MiniMax H3 prompt block per scene, "
            "without building or exposing a JSON chain plan."
        )
        schema.inputs = [
            item for item in schema.inputs
            if item.id not in {"draft_only", "bypass_director"}
        ]
        schema.outputs = [
            io.String.Output("prompt_lines", tooltip="Connect to Simple H3 Prompt Lines to List."),
            io.String.Output("first_prompt"),
            io.Int.Output("prompt_count"),
            io.String.Output("validation"),
            io.String.Output("usage_stats"),
            io.String.Output("credits_remaining"),
            io.String.Output("mood_analysis"),
        ]
        return schema

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        kwargs.update(draft_only=False, bypass_director=False)
        return H3StoryDirector.execute.__func__(cls, **kwargs)


class LTX25EditAnythingDirector(io.ComfyNode):
    """Multimodal OpenRouter prompt writer for the Edit Anything IC-LoRA."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25EditAnythingDirector",
            display_name="LTX 2.5 Director — Edit Anything",
            category="text/ltx25",
            search_aliases=[
                "edit anything director", "ltx edit prompt", "openrouter video edit",
            ],
            description=(
                "Inspects three representative input-video frames and an optional target "
                "reference, then writes one training-aligned Edit Anything prompt through "
                "OpenRouter. Hold reuses the last successful prompt without spending credits."
            ),
            inputs=[
                io.String.Input(
                    "api_key", default="", placeholder="sk-or-v1-...",
                    extra_dict={"password": True},
                ),
                io.String.Input("model", default=DEFAULT_MODEL),
                io.Image.Input(
                    "video_frames",
                    tooltip="Batch containing the first, middle and last sampled video frames.",
                ),
                io.String.Input(
                    "edit_request", multiline=True, dynamic_prompts=False,
                    default="Replace the background with the environment shown in the reference image.",
                ),
                io.Image.Input(
                    "reference_image", optional=True,
                    tooltip=(
                        "Optional visual target. The Director describes it textually and the "
                        "workflow also supplies it to LTX as identity/reference conditioning."
                    ),
                ),
                io.Int.Input("max_tokens", default=768, min=256, max=2048),
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
                    tooltip="Reuse the last successful prompt. No OpenRouter call is made.",
                ),
            ],
            outputs=[
                io.String.Output("edit_prompt"),
                io.String.Output("validation"),
                io.String.Output("usage_stats"),
                io.String.Output("credits_remaining"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls, api_key: str, model: str, video_frames, edit_request: str,
        max_tokens: int, temperature: float, reasoning: bool, seed: int,
        image_max_dimension: int, timeout_seconds: int,
        hold_prompt: bool = False, reference_image=None, unique_id=None,
    ) -> io.NodeOutput:
        request_text = str(edit_request or "").strip()
        if not request_text:
            raise ValueError("Edit Anything Director requires an edit_request.")
        if video_frames is None or not hasattr(video_frames, "shape"):
            raise ValueError("Edit Anything Director requires sampled input-video frames.")
        frame_count = int(video_frames.shape[0])
        if frame_count < 1:
            raise ValueError("The sampled input-video frame batch is empty.")

        cache_key = f"{cls.__name__}:{str(unique_id or 'default')}"
        if bool(hold_prompt):
            held = _get_held_director_plan(cache_key)
            if held and held.get("cache_kind") == "ltx25_edit_anything_prompt":
                return io.NodeOutput(
                    held["edit_prompt"], held["validation"] + " · HOLD",
                    "Held prompt · OpenRouter not called", "Credits unchanged",
                    ui=ui.PreviewText(held["edit_prompt"]),
                )

        key = str(api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "An OpenRouter API key is required in the node or the "
                "OPENROUTER_API_KEY environment variable."
            )

        has_reference = (
            reference_image is not None
            and hasattr(reference_image, "shape")
            and int(reference_image.shape[0]) > 0
        )
        system = """
You are a precise multimodal video-edit prompt director for the LTX 2.5 Edit Anything IC-LoRA.
Inspect every supplied SOURCE VIDEO frame before writing. If supplied, inspect TARGET REFERENCE
IMAGE as visual evidence for the requested replacement, but never invent an additional edit.

Return English only. Return one JSON object with one key named edit_prompt and no commentary.
The edit_prompt must begin exactly once with `edit_anything: `.

Write exactly one instruction for each edit explicitly requested by the user. Never add an
unrequested change to lighting, color, background, wardrobe, subjects, camera or style.
Each instruction must be a single sentence of 10 to 20 words and begin with exactly one of:
Add, Replace, Remove, Restyle.

Use Add for one new object and anchor it to a visible person, object or region.
Use Replace for a direct substitution; every Replace sentence must contain the word `with`.
Use Remove to identify one visible target precisely.
Use Restyle for a named visible region, not vaguely for the entire video.
Prefer one Replace over Remove plus Add for ordinary substitutions.

Ground the source target using stable details visible across the sampled video frames. Preserve
all non-requested content, subject identities, motion, geometry, framing and temporal consistency.
For a requested background replacement, identify the existing background from the source frames,
describe the target reference's concrete environment, lighting and spatial features, and explicitly
preserve the foreground subjects unchanged. Do not literally mention frames, images, references,
Picture tags, analysis, the model, the LoRA or these rules in edit_prompt.
""".strip()
        content = [{
            "type": "text",
            "text": (
                "USER EDIT REQUEST:\n" + request_text +
                "\n\nThe next images labeled SOURCE VIDEO are chronological samples "
                "from the same clip. Apply only the requested edit consistently through time."
            ),
        }]
        # The upstream sampler normally supplies first/middle/last. Bound this to
        # three images so an accidental large batch cannot inflate API cost.
        sample_indices = list(range(min(frame_count, 3)))
        for index in sample_indices:
            content.append({"type": "text", "text": f"SOURCE VIDEO FRAME {index + 1}"})
            content.append({"type": "image_url", "image_url": {
                "url": _image_data_url(video_frames[index:index + 1], int(image_max_dimension))
            }})
        if has_reference:
            content.append({
                "type": "text",
                "text": (
                    "TARGET REFERENCE IMAGE. Use its visible appearance only when the "
                    "user explicitly asks for it; it is not an additional source-video frame."
                ),
            })
            content.append({"type": "image_url", "image_url": {
                "url": _image_data_url(reference_image[:1], int(image_max_dimension))
            }})

        payload = {
            "model": str(model or DEFAULT_MODEL).strip(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "reasoning": {"enabled": bool(reasoning)},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "edit_anything_prompt",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "edit_prompt": {"type": "string", "minLength": 20}
                        },
                        "required": ["edit_prompt"],
                        "additionalProperties": False,
                    },
                },
            },
            "provider": {"require_parameters": True},
        }
        result = _openrouter_request(key, payload, int(timeout_seconds))
        try:
            raw = result["choices"][0]["message"]["content"]
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            edit_prompt = str(parsed["edit_prompt"]).strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Edit Anything Director returned an invalid response.") from error
        if not edit_prompt.startswith("edit_anything: "):
            raise RuntimeError(
                "Director response did not begin with the required `edit_anything: ` trigger."
            )
        body = edit_prompt[len("edit_anything: "):].strip()
        instructions = [part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]
        warnings = []
        for index, instruction in enumerate(instructions, 1):
            if not re.match(r"^(Add|Replace|Remove|Restyle)\b", instruction):
                warnings.append(f"instruction {index} has an invalid opening verb")
            words = re.findall(r"\b[\w'-]+\b", instruction)
            if not 10 <= len(words) <= 20:
                warnings.append(f"instruction {index} has {len(words)} words")
            if instruction.startswith("Replace") and not re.search(r"\bwith\b", instruction):
                warnings.append(f"instruction {index} Replace lacks `with`")
        validation = (
            f"Edit Anything prompt ready · {len(sample_indices)} video samples · "
            f"target reference {'included' if has_reference else 'not supplied'} · "
            + ("format valid" if not warnings else "warnings: " + "; ".join(warnings))
        )
        usage = result.get("usage") or {}
        usage_stats = (
            f"input: {usage.get('prompt_tokens', '?')} · "
            f"output: {usage.get('completion_tokens', '?')} · "
            f"total: {usage.get('total_tokens', '?')}"
        )
        credits = _credits(key, min(30, int(timeout_seconds)))
        _set_held_director_plan(cache_key, {
            "cache_kind": "ltx25_edit_anything_prompt",
            "edit_prompt": edit_prompt,
            "validation": validation,
        })
        return io.NodeOutput(
            edit_prompt, validation, usage_stats, credits,
            ui=ui.PreviewText(edit_prompt),
        )


class H3DirectVideoEditDirector(io.ComfyNode):
    """Reusable multimodal OpenRouter prompt writer for direct H3 edits."""

    DEFAULT_SYSTEM_PROMPT = """
You are a precise multimodal prompt director for one-pass native MiniMax H3 video editing.
Inspect every connected VIDEO SOURCE and PICTURE REFERENCE before writing.
Return English only as one JSON object with one key named edit_prompt and no commentary.

The prompt must begin exactly with `[video editing]` when no target reference is supplied, or
`[video editing + reference generation]` when a target reference is supplied.

Use <Video 1> as the locked source video and temporal blueprint unless the user explicitly
assigns a different role. <Video 2>, when connected, is a secondary visual or temporal source:
use it only for the role explicitly assigned by the user. Connected <Picture 1> through
<Picture 4> are independent visual sources. Use each one only for the exact person, object,
place, material, wardrobe or style assigned to it by the user request.

<Mood Image 1> and <Mood Video 1>, when connected, are non-literal art-direction sources.
Use them to infer only atmosphere, palette, lighting character, texture language, environmental
density and visual energy requested by the user. Do not copy their people, identities, exact
objects, actions, geography or composition. When asked to invent a new background from mood,
design a genuinely new coherent environment that expresses those mood traits while matching
<Video 1>'s perspective, camera motion, occlusions and physical lighting.

First resolve the user's explicit source-to-target mapping. Never blend connected references
together merely because they are available. Never import an unassigned subject, background,
wardrobe, action, composition or lighting property from any reference. When the request is
ambiguous, choose the narrowest edit that preserves the maximum amount of <Video 1>.
Write one compact self-contained edit prompt, normally three or four short paragraphs:
1. Declare the role of every connected source that is actually used.
2. State only the requested replacement or modification.
3. State the specific reference traits that must transfer consistently.
4. Preserve every non-requested property of <Video 1> unchanged.

Never infer or add production genre, motion style, visual look, new camera direction, shot
description, opening state, closing state, story bible, subject definitions, extra action,
dialogue, sound design or negative-prompt list. Do not narrate what already happens in the source.
Do not repeat the same instruction. Do not invent edits.

For identity replacement, maintain exactly one physical body for the edited person and transfer
only the requested identity/appearance traits through every source pose, angle and occlusion.
For background replacement, preserve all foreground content and reconstruct only the referenced
environment with coherent perspective, scale, parallax, depth, occlusion, lighting, reflections
and contact shadows. The original audio is handled externally and need not be described.
""".strip()

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3DirectVideoEditDirector",
            display_name="H3 Director — Direct Video Edit (OpenRouter)",
            category="text/minimax_h3",
            search_aliases=[
                "h3 edit anything director", "direct h3 video edit",
                "openrouter h3 edit prompt",
            ],
            description=(
                "Multisource Edit-Anything-style Director for native H3 edits. "
                "It inspects samples from as many as two videos and four target "
                "pictures, resolves their explicit roles, and returns one direct "
                "prompt without a story plan."
            ),
            inputs=[
                io.String.Input(
                    "api_key", default="", placeholder="sk-or-v1-...",
                    extra_dict={"password": True},
                ),
                io.String.Input("model", default=DEFAULT_MODEL),
                io.Image.Input(
                    "source_video",
                    tooltip="Complete IMAGE frame batch from VHS Load Video.",
                ),
                io.String.Input(
                    "edit_request", multiline=True, dynamic_prompts=False,
                    default=(
                        "Replace only the original background of <Video 1> with the "
                        "environment from <Picture 1>, preserving everything else exactly."
                    ),
                ),
                io.Image.Input(
                    "reference_image", display_name="reference_image_1", optional=True,
                    tooltip="Optional <Picture 1>. Existing workflows remain compatible.",
                ),
                io.Image.Input("reference_image_2", optional=True),
                io.Image.Input("reference_image_3", optional=True),
                io.Image.Input("reference_image_4", optional=True),
                io.Image.Input(
                    "source_video_2", optional=True,
                    tooltip="Optional complete IMAGE frame batch used as <Video 2>.",
                ),
                io.Image.Input(
                    "mood_image", optional=True,
                    tooltip=(
                        "Optional <Mood Image 1>: non-literal atmosphere, palette, "
                        "lighting, texture and visual-energy reference."
                    ),
                ),
                io.Image.Input(
                    "mood_video", optional=True,
                    tooltip=(
                        "Optional <Mood Video 1> frame batch. It supplies non-literal "
                        "moving art direction, not subjects or exact scenery."
                    ),
                ),
                io.Combo.Input(
                    "video_samples", options=["3", "5", "10"], default="3",
                    tooltip=(
                        "Number of uniformly distributed frames inspected from each "
                        "connected video. More samples improve cut awareness but cost more."
                    ),
                ),
                io.String.Input(
                    "system_prompt", multiline=True, dynamic_prompts=False,
                    default=cls.DEFAULT_SYSTEM_PROMPT,
                    tooltip="Editable H3 direct-edit contract sent to OpenRouter.",
                ),
                io.Int.Input("max_tokens", default=1024, min=256, max=3072),
                io.Float.Input("temperature", default=0.15, min=0.0, max=1.0, step=0.05),
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
                    tooltip="Reuse the last successful prompt without calling OpenRouter.",
                ),
            ],
            outputs=[
                io.String.Output("edit_prompt"),
                io.String.Output("validation"),
                io.String.Output("usage_stats"),
                io.String.Output("credits_remaining"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls, api_key: str, model: str, source_video, edit_request: str,
        video_samples: str, system_prompt: str, max_tokens: int, temperature: float,
        reasoning: bool, seed: int, image_max_dimension: int,
        timeout_seconds: int, hold_prompt: bool = False,
        reference_image=None, reference_image_2=None, reference_image_3=None,
        reference_image_4=None, source_video_2=None, mood_image=None,
        mood_video=None, unique_id=None,
    ) -> io.NodeOutput:
        request_text = str(edit_request or "").strip()
        if not request_text:
            raise ValueError("H3 Direct Video Edit Director requires an edit_request.")
        if source_video is None or not hasattr(source_video, "shape"):
            raise ValueError("H3 Direct Video Edit Director requires source_video frames.")
        frame_count = int(source_video.shape[0])
        if frame_count < 1:
            raise ValueError("The source_video frame batch is empty.")
        reference_candidates = (
            reference_image, reference_image_2, reference_image_3, reference_image_4,
        )
        references = [
            (index, image)
            for index, image in enumerate(reference_candidates, 1)
            if image is not None and hasattr(image, "shape") and int(image.shape[0]) > 0
        ]
        has_reference = bool(references)
        has_video_2 = (
            source_video_2 is not None
            and hasattr(source_video_2, "shape")
            and int(source_video_2.shape[0]) > 0
        )
        has_mood_image = (
            mood_image is not None
            and hasattr(mood_image, "shape")
            and int(mood_image.shape[0]) > 0
        )
        has_mood_video = (
            mood_video is not None
            and hasattr(mood_video, "shape")
            and int(mood_video.shape[0]) > 0
        )
        cache_key = f"{cls.__name__}:{str(unique_id or 'default')}"
        if bool(hold_prompt):
            held = _get_held_director_plan(cache_key)
            if held and held.get("cache_kind") == "h3_direct_video_edit_prompt":
                return io.NodeOutput(
                    held["edit_prompt"], held["validation"] + " · HOLD",
                    "Held prompt · OpenRouter not called", "Credits unchanged",
                    ui=ui.PreviewText(held["edit_prompt"]),
                )

        key = str(api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "An OpenRouter API key is required in the node or the "
                "OPENROUTER_API_KEY environment variable."
            )
        system = str(system_prompt or cls.DEFAULT_SYSTEM_PROMPT).strip()
        requested_samples = int(str(video_samples or "3"))

        def uniform_indices(count: int) -> list[int]:
            if count <= 1:
                return [0]
            sample_count = min(requested_samples, count)
            return sorted(set(
                round(index * (count - 1) / (sample_count - 1))
                for index in range(sample_count)
            ))

        indices = uniform_indices(frame_count)
        content = [{
            "type": "text",
            "text": (
                "USER EDIT REQUEST:\n" + request_text +
                "\n\nThe following SOURCE VIDEO frames are chronological samples "
                "from the same <Video 1>."
            ),
        }]
        for order, index in enumerate(indices, 1):
            content.append({"type": "text", "text": f"SOURCE VIDEO SAMPLE {order}"})
            content.append({"type": "image_url", "image_url": {
                "url": _image_data_url(
                    source_video[index:index + 1], int(image_max_dimension)
                )
            }})
        video_sample_counts = [len(indices)]
        if has_video_2:
            video_2_count = int(source_video_2.shape[0])
            indices_2 = uniform_indices(video_2_count)
            content.append({
                "type": "text",
                "text": (
                    "SECONDARY VIDEO SOURCE <Video 2>. Use it only when and exactly "
                    "as assigned by the user edit request."
                ),
            })
            for order, index in enumerate(indices_2, 1):
                content.append({"type": "text", "text": f"<Video 2> SAMPLE {order}"})
                content.append({"type": "image_url", "image_url": {
                    "url": _image_data_url(
                        source_video_2[index:index + 1], int(image_max_dimension)
                    )
                }})
            video_sample_counts.append(len(indices_2))
        if has_mood_video:
            mood_video_count = int(mood_video.shape[0])
            mood_indices = uniform_indices(mood_video_count)
            content.append({
                "type": "text",
                "text": (
                    "NON-LITERAL ART-DIRECTION SOURCE <Mood Video 1>. Extract only "
                    "mood, palette, lighting character, texture language, environmental "
                    "density and visual energy. Never copy its people, identities, exact "
                    "objects, actions, geography or composition."
                ),
            })
            for order, index in enumerate(mood_indices, 1):
                content.append({
                    "type": "text", "text": f"<Mood Video 1> SAMPLE {order}"
                })
                content.append({"type": "image_url", "image_url": {
                    "url": _image_data_url(
                        mood_video[index:index + 1], int(image_max_dimension)
                    )
                }})
            video_sample_counts.append(len(mood_indices))
        for picture_index, image in references:
            content.append({
                "type": "text",
                "text": (
                    f"TARGET REFERENCE <Picture {picture_index}>. Use it only for the exact visual "
                    "assignment made in the user request."
                ),
            })
            content.append({"type": "image_url", "image_url": {
                "url": _image_data_url(image[:1], int(image_max_dimension))
            }})
        if has_mood_image:
            content.append({
                "type": "text",
                "text": (
                    "NON-LITERAL ART-DIRECTION SOURCE <Mood Image 1>. Extract only "
                    "atmosphere, palette, lighting character, texture language, "
                    "environmental density and visual energy. Invent new content rather "
                    "than copying its people, exact objects, geography or composition."
                ),
            })
            content.append({"type": "image_url", "image_url": {
                "url": _image_data_url(mood_image[:1], int(image_max_dimension))
            }})
        payload = {
            "model": str(model or DEFAULT_MODEL).strip(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "reasoning": {"enabled": bool(reasoning)},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "h3_direct_video_edit_prompt",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "edit_prompt": {"type": "string", "minLength": 30}
                        },
                        "required": ["edit_prompt"],
                        "additionalProperties": False,
                    },
                },
            },
            "provider": {"require_parameters": True},
        }
        result = _openrouter_request(key, payload, int(timeout_seconds))
        try:
            raw = result["choices"][0]["message"]["content"]
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            edit_prompt = str(parsed["edit_prompt"]).strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("H3 Direct Video Edit Director returned an invalid response.") from error
        required_header = (
            "[video editing + reference generation]"
            if has_reference else "[video editing]"
        )
        if not edit_prompt.startswith(required_header):
            edit_prompt = re.sub(
                r"^\s*\[(?:video\s+editing(?:\s*\+\s*reference\s+generation)?|reference\s+generation|image\s+editing(?:\s*\+\s*reference\s+generation)?)\]\s*",
                "",
                edit_prompt,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            edit_prompt = f"{required_header}\n\n{edit_prompt}"
        if "<Video 1>" not in edit_prompt:
            raise RuntimeError("Director response lost the required <Video 1> tag.")
        request_lower = request_text.lower()
        if has_video_2 and "<video 2>" in request_lower and "<Video 2>" not in edit_prompt:
            raise RuntimeError("Director response lost explicitly requested <Video 2>.")
        for picture_index, _image in references:
            tag = f"<Picture {picture_index}>"
            if tag.lower() in request_lower and tag not in edit_prompt:
                raise RuntimeError(f"Director response lost explicitly requested {tag}.")
        if has_mood_image and "mood" in request_lower and "<Mood Image 1>" not in edit_prompt:
            raise RuntimeError("Director response lost explicitly requested <Mood Image 1>.")
        if has_mood_video and "mood" in request_lower and "<Mood Video 1>" not in edit_prompt:
            raise RuntimeError("Director response lost explicitly requested <Mood Video 1>.")
        forbidden = []
        for label, pattern in (
            ("production-style inference", r"\bProduction style\s*:"),
            ("motion inference", r"\bMotion\s*:\s*infer"),
            ("visual-look inference", r"\bVisual look\s*:\s*infer"),
            ("story bible", r"\bstory bible\b"),
            ("subject-definition scaffold", r"\bsubject_definitions\b"),
        ):
            if re.search(pattern, edit_prompt, flags=re.IGNORECASE):
                forbidden.append(label)
        validation = (
            f"Direct H3 edit prompt ready · {sum(video_sample_counts)} video samples "
            f"from {len(video_sample_counts)} source(s) · {len(references)} picture reference(s) · "
            f"mood image {'yes' if has_mood_image else 'no'} · "
            f"mood video {'yes' if has_mood_video else 'no'} · "
            + ("compact contract clean" if not forbidden else "warnings: " + ", ".join(forbidden))
        )
        usage = result.get("usage") or {}
        usage_stats = (
            f"input: {usage.get('prompt_tokens', '?')} · "
            f"output: {usage.get('completion_tokens', '?')} · "
            f"total: {usage.get('total_tokens', '?')}"
        )
        credits = _credits(key, min(30, int(timeout_seconds)))
        _set_held_director_plan(cache_key, {
            "cache_kind": "h3_direct_video_edit_prompt",
            "edit_prompt": edit_prompt,
            "validation": validation,
        })
        return io.NodeOutput(
            edit_prompt, validation, usage_stats, credits,
            ui=ui.PreviewText(edit_prompt),
        )


class LTX25FirstFrameDirector(io.ComfyNode):
    """Literal, compact first-frame description for LTX 2.5 image-to-video."""

    SYSTEM_PROMPT = """
You are a literal visual prompt writer for LTX 2.5 image-to-video. Inspect the supplied FIRST FRAME
and return English only as one JSON object with exactly one key named `prompt`.

Write one compact paragraph of 60 to 130 words describing only what is visibly established in the
image. Describe the main visual subject, exact subject count when relevant, environment, foreground
and background depth planes, composition, framing, viewpoint, lens/depth-of-field character,
materials, lighting direction, color palette and atmosphere. Preserve the image's spatial layout.

Do not write an edit instruction. Do not mention an input image, first frame, reference, prompt,
model or analysis. Do not invent unseen people, objects, wardrobe, actions, dialogue, plot, camera
cuts or a new location. Never turn ambiguous shapes into people. If no person is visibly present,
describe the shot positively as an unoccupied environmental or architectural shot and make the
environment itself the only visual subject. Add only restrained, physically plausible ambient
motion already supported by visible elements, such as subtle foliage, haze, reflections or light;
otherwise keep the composition and camera stable. Do not append a negative prompt or headers.
""".strip()

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25FirstFrameDirector",
            display_name="LTX 2.5 Director — Compact First Frame",
            category="text/ltx25",
            search_aliases=["ltx first frame prompt", "ltx image description"],
            description=(
                "Inspects one first frame and writes a literal compact LTX 2.5 I2V prompt. "
                "It does not perform edits or invent additional shots."
            ),
            inputs=[
                io.Image.Input("first_frame"),
                io.Custom("LLMMODEL").Input("llm"),
                io.Int.Input("max_tokens", default=420, min=192, max=1024),
                io.Float.Input("temperature", default=0.1, min=0.0, max=0.6, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF,
                             control_after_generate=True),
                io.Int.Input("image_max_dimension", default=1024, min=512, max=2048,
                             step=64, advanced=True),
            ],
            outputs=[
                io.String.Output("prompt"),
                io.String.Output("validation"),
                io.String.Output("usage_stats"),
                io.String.Output("raw_response"),
            ],
        )

    @classmethod
    def execute(cls, first_frame, llm, max_tokens=420, temperature=0.1,
                seed=0, image_max_dimension=1024):
        if first_frame is None or not hasattr(first_frame, "shape") or int(first_frame.shape[0]) < 1:
            raise ValueError("LTX 2.5 First Frame Director requires one image.")
        if llm is None:
            raise ValueError("Connect an OpenRouter, Ollama or compatible LLMMODEL provider.")

        payload = {
            "model": str(getattr(llm, "model", "") or ""),
            "messages": [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "FIRST FRAME — describe its visible shot literally."},
                    {"type": "image_url", "image_url": {
                        "url": _image_data_url(first_frame[:1], int(image_max_dimension))
                    }},
                ]},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "ltx25_first_frame_prompt",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string", "minLength": 40}},
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            }},
            "provider": {"require_parameters": True},
        }
        result = _external_llm_request(llm, payload)
        try:
            raw = result["choices"][0]["message"]["content"]
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            prompt = str(parsed["prompt"]).strip()
            if len(prompt) < 40:
                raise ValueError("description too short")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("LTX 2.5 First Frame Director returned an invalid response.") from error

        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        usage_stats = (
            f"input: {usage.get('prompt_tokens', '?')} · "
            f"output: {usage.get('completion_tokens', '?')}"
        )
        raw_response = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        validation = "OK · literal first-frame description · LTX 2.5 I2V"
        return io.NodeOutput(prompt, validation, usage_stats, raw_response,
                             ui=ui.PreviewText(prompt))


class H3CompactDirectionControls(io.ComfyNode):
    """Optional, local-only direction hints for the compact director."""

    @classmethod
    def define_schema(cls):
        def choices(values):
            return ["none", *[value for value in values
                             if str(value).lower() != "none"]]

        return io.Schema(
            node_id="H3CompactDirectionControls",
            display_name="H3 Compact Director — Direction Controls",
            category="text/minimax_h3",
            inputs=[
                io.Combo.Input("genre_1", options=choices(GENRES), default="none"),
                io.Combo.Input("genre_2", options=choices(GENRES), default="none"),
                io.Combo.Input("motion_style_1", options=choices(MOTION_STYLES), default="none"),
                io.Combo.Input("motion_style_2", options=choices(MOTION_STYLES), default="none"),
                io.Combo.Input("look_1", options=choices(VISUAL_LOOKS), default="none"),
                io.Combo.Input("look_2", options=choices(VISUAL_LOOKS), default="none"),
                io.Combo.Input(
                    "dialogue", options=choices(DIALOGUE_OPTIONS), default="none",
                    tooltip="none leaves dialogue unchanged; No dialogue explicitly requests silence from speakers.",
                ),
            ],
            outputs=[io.String.Output("direction_context")],
        )

    @classmethod
    def execute(cls, genre_1="none", genre_2="none", motion_style_1="none",
                motion_style_2="none", look_1="none", look_2="none",
                dialogue="none"):
        lines = []
        for label, values in (
            ("Genre", (genre_1, genre_2)),
            ("Motion style", (motion_style_1, motion_style_2)),
            ("Visual look", (look_1, look_2)),
            ("Dialogue", (dialogue,)),
        ):
            active = list(dict.fromkeys(
                str(value).strip() for value in values
                if str(value or "").strip().lower() not in ("", "none")
            ))
            if active:
                lines.append(label + ": " + " + ".join(active))
        return io.NodeOutput("\n".join(lines))


class H3CompactMultimodalEditDirector(io.ComfyNode):
    """Compact, source-optional multimodal director for precise H3 edits."""

    DEFAULT_SYSTEM_PROMPT = """
You are a precise multimodal edit director for MiniMax H3. Inspect the actual pixels of every
connected source before writing. Return English only as one JSON object with exactly four keys:
`edit_type`, `source_roles`, `visual_evidence`, and `edit_prompt`. `edit_type` classifies the
requested operation. `source_roles` briefly binds each used tag to one role. `visual_evidence`
lists only concrete visible traits needed for the edit. `edit_prompt` integrates that evidence
into the final compact instruction. Do not include a bracketed mode header; the node adds it.

Resolve the user's source-to-target mapping first. Use only connected sources and only for their
assigned roles. <Picture 1> through <Picture 4> are literal visual references. <Video 1> and
<Video 2> may be source plates or temporal references. <Mood Image 1> and <Mood Video 1> are
flexible direction references. Never blend sources merely because they are connected and never
transfer an unrequested person, object, setting, wardrobe, action or style.

Write one compact, self-contained edit instruction. Name every source actually used, state the
exact requested change, describe the concrete visible traits that must transfer, and preserve all
unrequested content. Prefer specific visual language over generic phrases such as `same person`,
`complete environment`, `matching outfit`, or `use the reference`. If the request conflicts with
visible source evidence, obey the explicit request while preserving everything it does not change.

Apply the relevant rules:
- Background or environment: preserve foreground subjects and performance; match perspective,
  scale, parallax, depth, occlusion, reflections, contact shadows, practical-light direction,
  color temperature and depth of field.
- Identity or person: transfer observed face, hair and distinctive assigned traits onto exactly
  one existing body; preserve pose, anatomy, action, timing, gaze, expression and occlusions unless
  the user requests otherwise.
- Wardrobe: state garment type, cut, fit, material, color, pattern and accessories; preserve the
  wearer, body mechanics and temporal consistency.
- Object replacement: preserve placement, grip/contact, scale, perspective, material response,
  shadows, reflections and occlusion order.
- Pose, action or motion: preserve identity and scene while matching the assigned body configuration,
  direction, weight, rhythm and camera-relative movement without duplicating limbs or subjects.
- Style, mood or relighting: transfer only the requested palette, contrast, texture, lighting,
  lens or motion qualities; do not import unrelated content.

For video, treat the assigned source plate as the temporal blueprint and preserve chronology,
camera path, cuts, timing, speed, audio-visible synchronization and continuity unless explicitly
changed. For image-only editing, preserve composition, viewpoint and geometry unless explicitly
changed. Do not invent extra shots, actions, story, dialogue, negative prompts or production notes.
""".strip()

    DEEP_EDIT_RULES = """

DEEP EDIT MODE — H3 CONTEXT-IR STYLE
Perform a private, evidence-led edit analysis before writing. The JSON fields are the audit trail;
the final edit_prompt must still be economical, executable prose rather than an explanation.

REQUEST PRECEDENCE IS ABSOLUTE:
- Treat every explicit requested modification as part of one cumulative edit set, including later
  clauses, corrections, additions, slang and misspellings. A phrase such as `replace only the
  background` limits that particular background operation; it does not cancel a later explicit
  request to also change the character, body, wardrobe, state, action, camera or look.
- Preservation applies only after subtracting the cumulative edit set. Preserve every attribute
  the user did not request to change, but never preserve, restate as locked, or protect an attribute
  that the user explicitly changes. For example, a request to remove or replace clothing means the
  original clothing must not appear in the preservation list, even when the same request also says
  to preserve the person or foreground.
- Specific instructions override broad invariants. `Preserve the woman` normally means preserve her
  identity, performance, pose and timing; it does not mean preserve her clothing, hairstyle, body,
  or appearance when one of those is separately requested to change.
- Before returning, verify that every requested operation appears positively and unambiguously in
  edit_prompt. Remove any sentence such as `do not alter the subject`, `preserve all foreground`, or
  `change nothing else` if its scope would contradict even one requested change. Rewrite it with
  explicit exceptions instead: preserve the listed unaffected properties while applying all named
  changes.

1. Build an explicit source inventory. Bind each connected tag to exactly one requested function:
locked source plate, identity/character, body or proportions, wardrobe, object, environment,
lighting/look, pose/action, camera language, interaction choreography, or temporal reference.
Unused sources stay unused. A mood source is not automatically a background.
2. Convert the request into ordered change-and-preserve pairs. For every change identify the target
instance, replacement source, spatial/temporal scope, traits to transfer, traits to retain, and the
physical integration needed. Resolve pronouns and ambiguous targets from visible evidence.
3. Decide whether this is a conservative edit or an intentionally generative restaging. Lock the
source plate by default. Unlock composition, viewpoint, camera path, action, or timing only when the
user explicitly requests new shots, poses, actions, interactions, or camera work.
4. Design one coherent result. Do not average references. When different references control face,
hair, physique, wardrobe, motion, environment, or look, assign those attributes separately and
recombine them on one consistent subject/body and one consistent world.
5. Express visible evidence concretely and selectively. Describe identity through observed facial
geometry, hair, skin and distinctive stable traits; wardrobe through silhouette, construction,
fabric and fit; environments through layout, depth planes, materials and practical lights; look
through capture medium, lens behavior, contrast, palette, grain and lighting—not vague adjectives.

Operation contracts:
- Background transfer: reconstruct the target environment as a three-dimensional moving plate,
not a flat cutout. Preserve foreground matte edges, hair detail and transparent/reflective objects;
solve horizon, vanishing lines, scale, camera parallax, occlusion order, depth of field, spill,
contact shadows, reflections, atmospheric depth, light direction, intensity and color temperature.
- Character replacement: replace the intended person, not merely clothing. Transfer face, head,
hair, skin, physique and explicitly assigned distinctive traits to exactly one body across every
angle and occlusion. Preserve source performance, pose sequence, gaze, expression, contacts,
wardrobe and timing unless the request assigns any of those to another reference.
- Wardrobe: reconstruct the garment on the moving body with stable cut, layers, closures, fabric,
fit, wrinkles and accessories; preserve anatomy and correct cloth-body/hand occlusions.
- Add/remove/object edit: specify count, placement and ownership. Maintain scale, support, grip,
collision, reflections, shadows and reveal/occlusion behavior; never duplicate a subject or prop.
- Pose/action/interaction: state participants, initiator, recipient, contact points, direction,
weight transfer, reaction and temporal order. Maintain anatomy, screen direction and causal motion.
- Camera/shot change: when requested, name framing, height, angle, lens character, movement path,
subject blocking and transition timing. Otherwise camera, framing and cuts are locked.
- Look/relighting: transfer only assigned capture and lighting attributes. Relight foreground and
environment coherently while retaining identity, geometry and material identity.

For audiovisual edits, describe events chronologically when they change over time. Preserve source
audio, lip synchronization, musical/performance timing and causal contacts unless explicitly edited.
Use positive construction language. Include only a short final failure-control clause for the most
likely task-specific failures; never append a generic negative-prompt dump. Keep edit_prompt usually
between 120 and 320 English words, expanding only for genuinely multi-operation edits. Every source
tag used in the request must appear canonically as <Picture N>, <Video N>, <Mood Image 1>, or
<Mood Video 1>.

When an actual source video is being edited, reason using H3's full-reference concepts: stable
subject definitions, task summary, retention relationships and chronological shot description.
For a simple direct edit, compress that reasoning into one precise paragraph like the user's
existing compact edit contract. Use the full labeled sections `subject_definitions:`, `summary:`,
`retention_analysis:`, `detailed_description:`, `overall_soundscape:`, and
`non_diegetic_music:` only when multiple subjects, timed changes, interactions or requested shot
changes genuinely need them. Define reusable people, environments, wardrobe, objects, actions or
styles as stable <Subject N> units when the structured form is used. In retention_analysis use only
`fully_preserved`, `partially_preserved`, `attribute_transfer`, or `weak_reference` for visible
content. Use [Shot 1] and timed later shots only when the source has cuts or the user requests new
coverage; do not invent cuts. Keep copied source audio explicit in prose without inventing an
<Audio N> tag when no separate audio reference is connected. The node supplies the bracketed task
header outside edit_prompt.
For image-only edits, do not force this six-section video structure: return one direct, spatially
precise edit instruction with the same source-role and preservation discipline.
""".strip()

    ELABORATE_RULES = """
ELABORATE MODE — RICH SINGLE-PROMPT DIRECTION
Produce one substantially developed, production-ready prompt rather than a compact summary.
Preserve the same source-role discipline and cumulative change/preserve contract, but give the
requested result enough concrete visual and temporal information for MiniMax H3 to stage it
coherently. The final edit_prompt should normally contain 380–650 English words and may reach 800
only for a genuinely complex multi-subject or multi-operation request. Length is not a quota:
every detail must guide a visible or audible property of the result.

CREATIVE DIRECTOR MANDATE:
The user's text may be only a seed. Elevate it into a distinctive, fully conceived audiovisual
moment rather than paraphrasing or padding it. Creative completion is explicitly authorized for
every dimension the user leaves unspecified: precise location and time, production design,
foreground activity, atmosphere, weather, practical props, performance intention, staging,
choreography, camera grammar, lens and focus behavior, lighting progression, palette, environmental
motion, soundscape, rhythm and final visual payoff. Make decisive compatible choices; never respond
with alternatives, `could`, `may`, generic filler or a neutral default. The result should reveal a
directorial point of view and feel designed for this exact premise rather than reusable stock prose.

Before writing, privately solve the following compact creative blueprint. Do not output its labels,
analysis or JSON; express its decisions only through the finished edit_prompt:
- Creative objective: identify what the audience should feel, notice and remember at the end.
- Action spine: preserve every user-requested event in order, then add only small connective beats
  that make cause, physical movement, reaction and completion readable.
- Reference ownership: decide exactly which identity, wardrobe, object, environment, pose, style or
  motion traits each source controls and prevent unrelated traits from leaking across sources.
- World design: choose a specific geography with navigable depth, architecture or natural forms,
  materials, set dressing, practical light sources, atmosphere and a coherent sound perspective.
- Subject direction: give each visible subject an intention expressed through gaze, posture,
  breathing, gesture, timing and interaction with space; appearance alone is not performance.
- Cinematic strategy: choose framing, axis, camera height, lens behavior, focus, movement and
  reveal structure because they strengthen the premise, emotion, power relationship or product.
- Sensory arc: shape light, color, texture, environmental motion and sound across the action instead
  of describing one static mood repeatedly.
- Payoff: end on a concrete changed composition, completed action, reveal, reaction or resonant held
  beat that fulfills the user's idea rather than merely stopping.

CREATIVE FREEDOM BOUNDARY:
Explicit requests and assigned source traits remain authoritative. A source video used as a locked
plate still protects its timing, performance, camera and audio unless the user changes them. Do not
replace identities, assigned wardrobe, named locations, required actions or outcomes. Do not add a
new principal character, unrelated subplot, dialogue, major stunt, violence, intimacy, supernatural
event or location change merely for spectacle. Within those boundaries, enrich empty space boldly
with compatible environmental detail, motivated secondary action, expressive reactions, camera
design, atmosphere and sound. `Preserve unrequested content` is not a command to be unimaginative:
it protects established evidence while permitting invention wherever the request and sources are
silent.

Develop the prompt as cohesive prose in a useful cinematic order:
1. Establish the current subjects, their stable identity traits, relevant wardrobe and exact
   positions relative to one another and to the environment.
2. Build the setting through specific layout, foreground/midground/background depth, materials,
   practical elements, atmosphere, weather when applicable, palette, light sources and shadow
   behavior. Make the selected genre or mood tangible without importing unrelated story content.
3. Describe requested actions chronologically with initiator, direction, pace, body mechanics,
   contact, reactions and resulting state. Keep causality readable and do not repeat completed beats.
4. Specify shot size, camera height and angle, lens/depth-of-field character and camera movement.
   Preserve source framing and motion unless the request or Direction Controls authorizes a change.
5. Integrate physically plausible ambience, Foley, dialogue and music only when applicable. Preserve
   source audio and synchronization whenever the source video is the temporal plate.
6. Close with a short task-specific continuity constraint covering only the most likely failures,
   such as identity drift, duplicated subjects, unstable wardrobe, broken contact, incorrect
   occlusion or environment flicker. Do not append a generic negative-prompt list.

Use precise sensory and spatial details instead of adjective stacks. Maintain one coherent world,
consistent screen direction, scale, perspective, parallax, reflections, occlusion, contact shadows,
light spill and temporal continuity. Explicit user requests remain absolute; elaboration must never
invent extra characters, actions, dialogue, cuts, plot escalation or changes to protected content.
Return the same four-key JSON schema required by the base director. `edit_prompt` contains the rich
finished instruction; `visual_evidence` remains a concise audit rather than duplicating the prompt.

DIRECT VISUAL PROSE IS MANDATORY:
- Write edit_prompt as a description of finished footage visibly unfolding, never as a request to
  another model. Do not begin with or use meta-directive phrases such as `Create a scene`,
  `Generate`, `Show`, `Depict`, `Make`, `Use`, `Maintain`, `Preserve`, `Ensure`, `Keep`, `Avoid`,
  `Do not`, `The scene should`, `The model should`, or `focus on`.
- Begin with the active canonical subjects or visible setting: `<Picture 1> stands...`,
  `<Video 1> continues...`, or `Inside the narrow workshop...`. State preservation as an observable
  fact: `Her facial structure, white hair and purple beanie remain consistent throughout`, never
  `Maintain her identity`.
- Replace emotional or stylistic adjective stacks with observable evidence. Words such as intense,
  passionate, desperate, visceral, raw, frantic or animalistic cannot stand in for choreography.
  Express their visible meaning through exact distance, posture, gaze, hand placement, direction,
  pace, contact, weight transfer, breathing and reaction.
- Write actions as chronological micro-beats with a readable beginning, development and resulting
  state. Do not summarize the action as `they engage in` or `are locked in`; describe what each
  participant actually does and how the other visibly responds.
- The final continuity sentence must remain descriptive, such as `Their faces, clothing and anatomy
  remain stable through every occlusion`, with no command verbs or production commentary.
""".strip()

    CONTINUOUS_RULES = """
ENHANCE — CHRONOLOGICAL GENERATION
Turn the user's idea into exactly the requested number of standalone scene prompts.
Priority: explicit user instructions, assigned reference roles, selected direction, then creative completion.
A character reference supplies identity and visible wardrobe unless the user changes them;
it does not impose its background, pose, expression or camera. Use a reference environment,
composition or starting pose only when assigned that role. Keep the user's named location.
Develop unspecified surroundings and performance naturally within the requested idea.

Divide the action into distinct chronological beats. Each scene advances the preceding state
without replaying completed actions, anticipating later beats or inventing additional plot.
Establish a readable starting position, action and resulting state; reach the requested outcome
in the final scene. Preserve established identities, wardrobe and spatial continuity unless changed.
Use present-tense English prose, normally 70–180 words per scene, up to 260 when necessary,
not as a quota. Return only the requested JSON, not analysis, timestamps or a separate plan.

SOURCE-TAG CONTRACT
Each scene explicitly binds every active referenced element to its canonical tag:
<Picture N>, <Video N>, <Mood Image 1> or <Mood Video 1>.
Keep angle brackets; never substitute an untagged "same person" or invent <Subject N>.
Reference only connected sources and only for their assigned roles.
""".strip()

    SPECIFICITY_RULES = """
CONCISE VISUAL SPECIFICITY
Make each prompt executable: current tagged cast, location with two or three distinguishing
spatial/material/light details, brief relevant wardrobe, focused action, framing and relevant sound.
Use reference details only for assigned roles; do not import an unrelated reference background.
State where subjects are relative to one another and the environment. Choose a concrete camera
position or path appropriate to the beat; no particular movement is required without a selection.
For editing, describe the requested changes and briefly preserve unaffected elements.
Locked camera, performance and audio take priority over creative additions.
Every sentence adds useful information; avoid repeated descriptions and preservation boilerplate.
""".strip()

    CONTEXTUAL_REACTION_RULES = """
PERFORMANCE AND COMPLETION
Give important interactions one or two observable, causally connected reactions: expression,
gesture or posture consistent with the event and the character's intent. Identity is not a fixed
expression. Do not copy a reference smile into a changed emotional situation or use arbitrary
gestures for drama. Hostility implies distress or self-protection unless the user specifies otherwise;
appearance and genre do not imply consent or enjoyment. Do not invent unrequested escalation.
In source edits, change performance only when authorized; preserve locked timing and audio.

When speech is appropriate, write brief actual lines responding to the current action and intent,
not exposition or a description of a conversation. Bind each speaker to a tagged subject; use
stable speaker numbers and (S1) says: <d>[Spanish] exact words</d>, substituting the actual language.
Directions stay English; dialogue uses the requested language. Keep supplied lines verbatim.
Dialogue none means no override, not mandatory speech or silence; No dialogue prohibits speech.
Do not replace preserved source speech or lyrics. Sound must have a plausible physical source.

Check the sequence silently: correct cast, reference roles, named location, stable wardrobe,
readable positions, distinct beats, motivated reactions and the exact requested final outcome.
Do not substitute an attempt for success or unconsciousness for death, or add graphic injury detail.
Make a selected genre recognizable through a coherent few environment, light, staging or sound
choices, maintained across scenes without changing protected elements or adding plot.
When controls are none, make context-appropriate creative choices rather than using a fixed template.
Return only the finished prompts.
""".strip()

    CONTINUOUS_EDIT_RULES = """
EDIT CONTINUO — EDIT CONTRACT PER SCENE
Use the same precise change/preserve discipline as edit mode, but distribute the requested
progression into exactly the selected number of chronological scene prompts.
The output schema for this mode overrides single-edit output instructions: return only
{"scene_prompts": ["...", "..."]}. Do not return edit_type, source_roles or visual_evidence;
use those concepts internally to write each finished prompt.
Each scene prompt is a standalone, compact edit instruction, normally 80–200 English words,
with a maximum of 320 only when needed for a complex requested edit, never as a quota:
identify the affected subject/region and canonical source tags, specify the current requested
change concretely, then briefly preserve the relevant unaffected attributes.
Repeat persistent edits in later scenes so they do not revert. Advance only changes/actions
the user requested, maintaining the established result of prior scenes. Never replay completed
changes, anticipate later beats, invent plot events or introduce camera cuts merely to fill
the scene count. A single static edit remains consistent in every scene.
When a source video is connected, preserve its performance, timing, camera and audio unless
explicitly changed. Samples are observations, not separate scenes; never claim knowledge of
unseen moments. With only images, preserve the assigned identities/designs while developing
only the requested sequence.
Keep every applicable <Picture N>, <Video N> and mood reference explicit per scene. Do not
use "same person" as a replacement for a source tag. Return final instructions, not analysis,
and do not embed your own task headers: the node supplies those.
""".strip()

    CONTINUOUS_ELABORATE_RULES = """
CONTINUOUS ELABORATE — RICH EDIT CONTRACT PER SCENE
This mode keeps the exact sequencing and JSON contract of CONTINUOUS EDIT, but deliberately
expands every scene into richer production direction. Return exactly the selected number of
standalone scene_prompts. Each scene should normally contain 260–480 English words and may reach
650 only when the requested beat genuinely needs multiple subjects, interactions or simultaneous
edits. These length targets override the compact word targets in CONTINUOUS EDIT; detail is useful
only when it controls a visible, temporal or audible property.

CREATIVE SEQUENCE DIRECTOR MANDATE:
Treat the user's text as the action seed for one designed audiovisual progression, not as prose to
paraphrase once per scene. Creative completion is explicitly authorized wherever the request and
assigned sources are silent: choose a precise setting and time, production design, atmosphere,
performance intention, connective physical behavior, visual motifs, environmental activity,
camera strategy, lighting and color progression, sound perspective, rhythm, reveals and a final
payoff. Make decisive compatible choices with a recognizable directorial point of view. Never pad
the sequence by restating the premise, multiplying adjectives or repeating the same action with
greater claimed intensity.

Before writing, privately solve one sequence blueprint. Do not expose its labels, analysis or JSON:
- Creative objective: the audience-facing feeling, idea, spectacle or product value that governs
  the complete sequence and the memorable image or state that delivers it.
- User action spine: every requested action, transformation, interaction, line and final outcome in
  its original causal order, with no omission or unauthorized substitution.
- Reference ownership: the exact identity, appearance, wardrobe, object, environment, pose, style,
  motion or temporal role controlled by every connected source.
- Continuity ledger: immutable identities and designs plus evolving geography, body configuration,
  hand/object contacts, gaze, momentum, camera axis/path, lighting phase and audio phase.
- World design: one coherent location with traversable depth, materials, set dressing, practical
  lights, atmosphere, environmental behavior and consistent acoustic character.
- Cinematic arc: a motivated evolution of framing, camera relationship, lens/focus behavior,
  staging, light, palette and sound that supports the action rather than randomly decorating it.
- Scene functions: assign each block a distinct dramatic or visual purpose—initiation, development,
  complication, reveal, reaction, transformation, culmination or settle—as appropriate to the
  user's actual request. Never invent conflict merely to fill these functions.
- Boundary handoffs: define the exact unfinished physical and audiovisual state passed between
  adjacent scenes so motion, contact, camera travel, light and sound do not reset.
- Final payoff: reserve the requested completion or strongest resolved image for the final scene,
  then allow a brief natural settling beat.

CREATIVE FREEDOM BOUNDARY:
Explicit requests and source assignments remain authoritative. Do not change identity, assigned
wardrobe, named location, required action, consent, relationship or outcome. Do not introduce a new
principal character, unrelated subplot, dialogue, violence, intimacy, supernatural event, stunt,
cut or location change simply to make the sequence seem dramatic. A locked source video retains its
timing, camera, performance and audio outside requested edits. Within these boundaries, invent
compatible context and production detail boldly enough that every scene feels authored and specific.

Within every scene, establish the currently visible tagged cast and stable appearance, relevant
wardrobe and objects, spatial relationships and the persistent edited state inherited from earlier
scenes. Develop the location through concrete layout, foreground/midground/background depth,
materials, practical elements, atmosphere, palette, motivated light sources, shadows, reflections
and depth of field. Make selected genre, look and mood tangible through a coherent set of details,
without importing unrelated story content.

Describe the current beat chronologically: starting state, initiator, direction and pace of motion,
body mechanics or object behavior, contact and occlusion, motivated reactions, and the resulting
state that the following scene must inherit. Advance the user's requested progression without
replaying completed actions or prematurely performing later beats. Keep identity, wardrobe,
environment geography, screen direction, scale, perspective and temporal state stable across all
prompts unless the request explicitly changes them.

Specify shot size, camera height and angle, lens character, focus behavior and any selected camera
path with physically coherent parallax. When editing a source video, its framing, camera, timing,
performance and audio remain locked unless explicitly changed. Integrate relevant ambience, Foley,
dialogue and music in the same chronology. End each prompt with only a brief task-specific
continuity safeguard; never append a generic negative-prompt dump.

Every applicable connected canonical source tag must appear verbatim in every scene where its
assigned element remains active. Return only {"scene_prompts": ["...", "..."]}; do not include
analysis, headings, timestamps or the four-key single-edit schema. The node adds task headers.

DIRECT VISUAL PROSE IS MANDATORY IN EVERY SCENE:
- Describe finished footage unfolding in present tense. Never address the generator or use
  `Create`, `Generate`, `Show`, `Depict`, `Make`, `Use`, `Maintain`, `Preserve`, `Ensure`, `Keep`,
  `Avoid`, `Do not`, `should`, `focus on`, or comparable production commands.
- Begin each prompt with the active tagged subject or visible setting and immediately establish its
  starting physical state. State persistent identity, wardrobe, environment and continuity as
  visible facts rather than instructions.
- Convert mood words into exact performance and staging. Describe positions, gaze, gestures, hand
  placement, direction, speed, contact, weight, breathing, reaction and the resulting state instead
  of repeating abstract adjectives.
- Give each block chronological micro-beats while inheriting the preceding state. The last sentence
  remains descriptive and cannot become a list of prohibitions or generation safeguards.
""".strip()

    I2V_RULES = """
IMAGE-TO-VIDEO MODE — PICTURE 1 IS FRAME ZERO
The generation is image-to-video. <Picture 1> is not a loose character, style or composition
reference: it is the exact visible opening frame of the generated video. Inspect its pixels and
begin from its established subject count, identities, wardrobe, objects, poses, hand placement,
facial expressions, environment, framing, viewpoint, scale, depth, lighting and color. The first
described movement must emerge naturally from that precise physical state.

Write direct finished-footage prose, not an edit request. Do not say create, generate, use the
image, match the reference, transform the picture, animate the image, maintain, preserve, ensure,
avoid or do not. State stable properties as visible facts. Mention <Picture 1> canonically so H3
binds the opening visual source, but never call it an input image or reference inside the prompt.

The user's requested motion, action, camera behavior, atmosphere and outcome determine what happens
after frame zero. Do not reset poses, relocate subjects, change wardrobe, introduce a new opening
composition or invent a transition into the visible starting state. When the requested action
requires repositioning, describe the movement from the observed pose to the new pose chronologically.
Camera movement begins from the observed framing with coherent parallax and occlusion; a locked
camera remains locked. Natural secondary motion may develop only from visible materials and forces.

For sequence modes, Scene 1 starts exactly at <Picture 1>. Later scenes inherit the preceding
scene's final state and use <Picture 1> only as the persistent identity/design anchor; they must not
snap back to its original pose or composition. Every prompt still includes <Picture 1> wherever its
assigned subjects or design remain active.
""".strip()

    ELABORATE_DIRECTION_RULES = """
ELABORATE DIRECTION CONTROLS — VERIFIABLE EXECUTION
Every selected non-none Direction Control is a binding visual or audible requirement. Do not merely
name, paraphrase or acknowledge a selection. Translate it into observable construction inside the
finished footage. A prompt fails this mode if deleting the control's label would leave no concrete
evidence that the selection changed the scene.

- Genre: realize it through at least three mutually coherent choices among environment/production
  design, lighting and palette, performance/staging, atmosphere and sound. Never output the genre
  label as a substitute for those choices, relocate the requested setting without permission or
  invent plot events associated with the genre.
- Motion style: specify the mechanics that distinguish the selected motion: who or what moves,
  trajectory, direction, speed or cadence, acceleration/deceleration, depth change, secondary
  motion, camera response and resulting position. Mood adjectives do not satisfy motion controls.
- Camera style: define the starting shot size and viewpoint, camera height, physical travel path,
  direction, relationship to a clear pivot or tracked subject, speed, stabilization character,
  changing foreground/background relationships and final composition. A named camera move cannot
  be replaced by a generic pan, zoom, reframing or unspecified `dynamic camera`.
- Visual look: express capture format, lens behavior, depth of field, contrast, highlight/shadow
  response, palette, texture and motivated light behavior as applicable. `Cinematic`, `dramatic
  lighting`, `beautiful` or the selected label alone is insufficient.
- Dialogue: `No dialogue` means no spoken words or vocalizations presented as speech. A selected
  language applies only to dialogue permitted by the request and uses exact H3 dialogue syntax;
  never invent speech solely because a language is selected.

When two controls of one category are selected, the first defines the dominant execution and the
second supplies compatible secondary traits. State one coherent result rather than two competing
alternatives. In Continuous Elaborate, carry the chosen grammar across every scene, but continue
camera paths, actions and temporal states from the prior endpoint instead of restarting them.
""".strip()

    ELABORATE_ORBIT_RULES = """
ORBIT CAMERA — REQUIRED GEOMETRY
The prompt must describe a genuine translating camera orbit, not just contain the words orbit,
circle or parallax. Identify: (1) the initial viewpoint relative to the pivot subject(s), (2) a
clockwise or counterclockwise direction, (3) an approximate arc in degrees, (4) camera height,
(5) a smooth circular travel path with a substantially constant radius, (6) the pivot/center held
in composition, (7) a clear final viewpoint, and (8) at least two depth layers whose different
screen displacement demonstrates parallax. Describe how subject overlap, profile or background
reveal changes during the move. Keep focal length and subject scale substantially stable; describe
the viewpoint physically travelling through space rather than remaining planted for a pan or using
an optical zoom. Express all of this as direct finished-footage prose, not camera instructions.

For Continuous Elaborate, divide one coherent orbit into consecutive arc segments. Each scene begins
at the exact viewpoint reached by the prior scene and proceeds in the same direction, height and
radius unless the user explicitly requests a change. Never reset to the original angle at a scene
boundary and never repeat the same arc description in every scene.
""".strip()

    @staticmethod
    def _debug_request(llm, payload):
        builder = getattr(llm, "build_request_payload", None)
        exact = callable(builder)
        body = builder(payload) if exact else payload
        image_count = 0
        secret = str(getattr(llm, "api_key", "") or "")

        def clean(value, field=""):
            nonlocal image_count
            if field.lower() in ("api_key", "authorization", "password", "access_token"):
                return "[REDACTED]"
            if field == "images" and isinstance(value, list):
                markers = []
                for _ in value:
                    image_count += 1
                    markers.append(f"[IMAGE {image_count}: base64 omitted]")
                return markers
            if isinstance(value, dict):
                return {key: clean(item, key) for key, item in value.items()}
            if isinstance(value, list):
                return [clean(item, field) for item in value]
            if isinstance(value, str):
                if value.startswith("data:image/"):
                    image_count += 1
                    return f"[IMAGE {image_count}: embedded data omitted]"
                if secret:
                    value = value.replace(secret, "[REDACTED]")
                return re.sub(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
            return value

        safe_body = clean(body)
        return json.dumps({
            "provider": type(llm).__name__,
            "request_kind": "provider request body" if exact else
                "director payload before provider adaptation",
            "images_omitted": image_count,
            "body": safe_body,
        }, ensure_ascii=False, indent=2)

    @staticmethod
    def _canonicalize_source_tags(text: str) -> str:
        canonical = str(text or "")
        for label in (
            "Mood Image 1", "Mood Video 1",
            "Picture 1", "Picture 2", "Picture 3", "Picture 4",
            "Video 1", "Video 2",
        ):
            canonical = re.sub(
                rf"(?<!<)\b{re.escape(label)}\b(?!>)",
                f"<{label}>",
                canonical,
                flags=re.IGNORECASE,
            )
        return canonical

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3CompactMultimodalEditDirector",
            display_name="H3 Director — Compact Multimodal Edit",
            category="text/minimax_h3",
            search_aliases=[
                "h3 compact edit director", "h3 multimodal edit", "h3 image video edit",
            ],
            description=(
                "Compact optional-source director for background, identity, wardrobe, object, "
                "pose, action, style and relighting edits. Requires an external LLM provider."
            ),
            inputs=[
                io.String.Input(
                    "edit_request", multiline=True, dynamic_prompts=False,
                    default="Describe the exact edit and assign each connected source its role.",
                ),
                io.Combo.Input(
                    "edit_mode", options=["compact", "edit", "Elaborate", "Enhance", "Continuous Edit", "Continuous Elaborate"], default="compact",
                    tooltip=(
                        "compact keeps the established short director. edit performs a "
                        "strong Context-IR-style source ledger and change/preserve analysis, "
                        "then returns one compact production prompt. Elaborate returns one richer "
                        "cinematic prompt with expanded setting, action, mood, camera and sound. "
                        "Enhance writes a specific "
                        "compact prompt for each requested scene. Continuous Edit applies edit "
                        "rules with one progressive edit prompt per scene. Continuous Elaborate "
                        "uses that same sequence structure with richer direction in every scene."
                    ),
                ),
                io.Image.Input("reference_image_1", optional=True),
                io.Image.Input("reference_image_2", optional=True),
                io.Image.Input("reference_image_3", optional=True),
                io.Image.Input("reference_image_4", optional=True),
                io.Image.Input("source_video_1", optional=True),
                io.Image.Input("source_video_2", optional=True),
                io.Image.Input("mood_image", optional=True),
                io.Image.Input("mood_video", optional=True),
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
                    "continuous_scene_count", display_name="Scenes — Enhance / Continuous Edit",
                    default=3, min=0, max=12, step=1,
                    tooltip=(
                        "Number of contiguous generated scenes. Used by the local compact "
                        "continuous plan and does not increase OpenRouter usage. Legacy value "
                        "0 is accepted and normalized internally to 1."
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
    def execute(
        cls, edit_request: str, edit_mode: str,
        video_samples: str,
        system_prompt: str, max_tokens: int, temperature: float, reasoning: bool,
        seed: int, image_max_dimension: int, timeout_seconds: int,
        hold_prompt: bool = False, reference_image_1=None, reference_image_2=None,
        reference_image_3=None, reference_image_4=None, source_video_1=None,
        source_video_2=None, mood_image=None, mood_video=None,
        continuous_scene_count: int = 3, unique_id=None, direction_context=None,
        llm=None, bypass: bool = False, debug_request: bool = False,
        i2v_mode: bool = False,
    ) -> io.NodeOutput:
        if bool(bypass):
            direct_prompt = str(edit_request if edit_request is not None else "")
            return io.NodeOutput(
                direct_prompt,
                "BYPASS · user prompt passed through unchanged",
                "Bypass · no LLM called",
                max(1, min(12, int(continuous_scene_count))),
                "BYPASS: no request sent." if debug_request else "",
                "BYPASS: no model response." if debug_request else "",
                ui=ui.PreviewText(direct_prompt),
            )
        if llm is None:
            raise ValueError(
                "Connect an external provider to llm: H3 OpenRouter Model, "
                "H3 Ollama Model or H3 Qwen3-VL Model. No internal API is available."
            )
        request_text = str(edit_request or "").strip()
        direction_text = str(direction_context or "").strip()
        if direction_text.lower() == "none":
            direction_text = ""

        requested_mode = str(edit_mode or "compact")
        requested_mode = {
            "edit": "deep_edit", "Enhance": "continuous", "enhance": "continuous",
            "Elaborate": "elaborate", "elaborate": "elaborate",
            "Edit Continuo": "continuous_edit",
            "Continuous Edit": "continuous_edit",
            "Elaborate Continuo": "continuous_elaborate",
            "Continuous Elaborate": "continuous_elaborate",
        }.get(requested_mode, requested_mode)
        if requested_mode not in (
            "compact", "deep_edit", "elaborate", "continuous",
            "continuous_edit", "continuous_elaborate",
        ):
            requested_mode = "compact"
        creative_control_present = bool(re.search(
            r"(?im)^(?:Genre|Motion style|Visual look):\s*\S", direction_text
        ))
        controls_only_generation = False
        if not request_text:
            if (
                requested_mode in ("elaborate", "continuous_elaborate")
                and creative_control_present
            ):
                controls_only_generation = True
                request_text = (
                    "No written premise was supplied. Invent one original, coherent audiovisual "
                    "concept grounded in the connected visual sources and governed completely by "
                    "the selected Direction Controls."
                )
            else:
                raise ValueError(
                    "Compact H3 Edit Director requires an edit_request, except Elaborate modes "
                    "can generate from active Genre, Motion style or Visual look controls."
                )

        def valid_frames(value):
            return (
                value is not None and hasattr(value, "shape")
                and len(value.shape) >= 4 and int(value.shape[0]) > 0
            )

        pictures = [
            (index, image) for index, image in enumerate((
                reference_image_1, reference_image_2,
                reference_image_3, reference_image_4,
            ), 1) if valid_frames(image)
        ]
        videos = [
            (index, video) for index, video in enumerate((
                source_video_1, source_video_2,
            ), 1) if valid_frames(video)
        ]
        has_mood_image = valid_frames(mood_image)
        has_mood_video = valid_frames(mood_video)
        if not (pictures or videos or has_mood_image or has_mood_video):
            raise ValueError("Connect at least one image, video or mood source.")

        is_sequence = requested_mode in (
            "continuous", "continuous_edit", "continuous_elaborate",
        )
        requested_count = max(1, min(12, int(continuous_scene_count)))
        cache_key = f"{cls.__name__}:{str(unique_id or 'default')}"
        provider_identity = (
            json.dumps({
                "type": type(llm).__name__,
                "model": str(getattr(llm, "model", "")),
                "endpoint": str(getattr(llm, "server_url", getattr(llm, "base_url", ""))),
                "clip_type": str(getattr(llm, "clip_type", "")),
                "thinking": getattr(llm, "thinking", None),
            }, sort_keys=True)
            if llm is not None else ""
        )
        if bool(hold_prompt):
            held = _get_held_director_plan(cache_key)
            if held and held.get("cache_kind") == "h3_compact_multimodal_edit":
                if str(held.get("provider_identity") or "") != provider_identity:
                    raise RuntimeError(
                        "The LLM provider/model changed. Disable Hold once to compare "
                        "the new provider; no model or API was called."
                    )
                if str(held.get("direction_context") or "") != direction_text:
                    raise RuntimeError(
                        "Direction Controls changed since Hold Edit Prompt was saved. "
                        "Disable Hold once to apply the new direction; no API call was made."
                    )
                if held.get("resolved_mode", "compact") != requested_mode:
                    raise RuntimeError(
                        "Hold Edit Prompt belongs to a different director mode. "
                        "Disable Hold once to regenerate."
                    )
                if bool(held.get("i2v_mode", False)) != bool(i2v_mode):
                    raise RuntimeError(
                        "I2V Mode changed since Hold Edit Prompt was saved. "
                        "Disable Hold once to regenerate."
                    )
                if (
                    is_sequence
                    and int(held.get("continuous_scene_count", 0)) != requested_count
                ):
                    raise RuntimeError(
                        "Hold Edit Prompt contains a different number of continuous "
                        "scene prompts. Disable Hold once to regenerate."
                    )
                held_prompt = cls._canonicalize_source_tags(held["edit_prompt"])
                held_preview = str(held.get("preview_prompt") or held_prompt)
                return io.NodeOutput(
                    held_prompt, held["validation"] + " · HOLD",
                    "Held prompt · provider not called",
                    requested_count,
                    (
                        "HOLD: historical request; no new request sent.\n"
                        + str(held.get("debug_request") or
                              "Unavailable. Enable debug_request and disable Hold for the next generation.")
                    ) if debug_request else "",
                    str(held.get("raw_response", "Unavailable: regenerate with debug_request enabled.")) if debug_request else "",
                    ui=ui.PreviewText(held_preview),
                )

        requested_samples = int(str(video_samples or "3"))

        def sample_indices(count):
            amount = min(requested_samples, int(count))
            if amount <= 1:
                return [0]
            return sorted(set(
                round(position * (int(count) - 1) / (amount - 1))
                for position in range(amount)
            ))

        content = [{
            "type": "text",
            "text": "USER EDIT REQUEST:\n" + request_text,
        }]
        for picture_index, image in pictures:
            content.append({
                "type": "text",
                "text": (
                    f"LITERAL VISUAL REFERENCE <Picture {picture_index}>. Inspect its actual "
                    "visible traits and use it only for the role assigned by the user."
                ),
            })
            content.append({"type": "image_url", "image_url": {
                "url": _image_data_url(image[:1], int(image_max_dimension))
            }})

        video_frame_total = 0
        for video_index, video in videos:
            indices = sample_indices(int(video.shape[0]))
            content.append({
                "type": "text",
                "text": (
                    f"CHRONOLOGICAL VIDEO SOURCE <Video {video_index}>. Infer its role only "
                    "from the user request and inspect the following ordered samples."
                ),
            })
            for order, frame_index in enumerate(indices, 1):
                content.append({
                    "type": "text",
                    "text": f"<Video {video_index}> SAMPLE {order}/{len(indices)}",
                })
                content.append({"type": "image_url", "image_url": {
                    "url": _image_data_url(
                        video[frame_index:frame_index + 1], int(image_max_dimension)
                    )
                }})
            video_frame_total += len(indices)

        if has_mood_image:
            content.append({
                "type": "text",
                "text": (
                    "FLEXIBLE DIRECTION REFERENCE <Mood Image 1>. Transfer only the visual "
                    "attributes explicitly assigned to it by the user."
                ),
            })
            content.append({"type": "image_url", "image_url": {
                "url": _image_data_url(mood_image[:1], int(image_max_dimension))
            }})
        if has_mood_video:
            indices = sample_indices(int(mood_video.shape[0]))
            content.append({
                "type": "text",
                "text": (
                    "FLEXIBLE DIRECTION REFERENCE <Mood Video 1>. Transfer only the motion "
                    "or visual attributes explicitly assigned to it by the user."
                ),
            })
            for order, frame_index in enumerate(indices, 1):
                content.append({
                    "type": "text",
                    "text": f"<Mood Video 1> SAMPLE {order}/{len(indices)}",
                })
                content.append({"type": "image_url", "image_url": {
                    "url": _image_data_url(
                        mood_video[frame_index:frame_index + 1], int(image_max_dimension)
                    )
                }})
            video_frame_total += len(indices)

        canonical_request = cls._canonicalize_source_tags(request_text)
        connected_canonical_tags = [
            *(f"<Picture {index}>" for index, _image in pictures),
            *(f"<Video {index}>" for index, _video in videos),
            *(("<Mood Image 1>",) if has_mood_image else ()),
            *(("<Mood Video 1>",) if has_mood_video else ()),
        ]
        persistent_tags = [
            tag for tag in connected_canonical_tags
            if tag.lower() in canonical_request.lower()
        ]
        # A sole ordinary picture is the persistent visual source for an
        # image-started continuation even if the user calls it only "the image".
        if len(pictures) == 1 and not videos and not persistent_tags:
            persistent_tags = [f"<Picture {pictures[0][0]}>"]

        resolved_control_specs = []
        for control_line in direction_text.splitlines():
            if ":" not in control_line:
                continue
            control_label, control_values = control_line.split(":", 1)
            control_label = control_label.strip().lower()
            selections = [
                value.strip() for value in control_values.split(" + ") if value.strip()
            ]
            if control_label == "motion style":
                for selection in selections:
                    specification = MOTION_STYLES.get(selection)
                    if specification:
                        resolved_control_specs.append(
                            f"Motion style `{selection}` means: {specification}"
                        )
            elif control_label == "visual look":
                for selection in selections:
                    specification = VISUAL_LOOKS.get(selection)
                    if specification:
                        resolved_control_specs.append(
                            f"Visual look `{selection}` means: {specification}"
                        )
            elif control_label == "genre":
                for selection in selections:
                    resolved_control_specs.append(
                        f"Genre `{selection}` governs the premise, performance, production "
                        "design, atmosphere, lighting, palette, pacing and sound through "
                        "recognizable concrete choices rather than the label alone."
                    )

        resolved_mode = requested_mode
        resolved_system = str(system_prompt or cls.DEFAULT_SYSTEM_PROMPT).strip()
        if resolved_mode == "deep_edit":
            resolved_system = "\n\n".join((resolved_system, cls.DEEP_EDIT_RULES))
        elif resolved_mode == "elaborate":
            resolved_system = "\n\n".join((
                resolved_system, cls.DEEP_EDIT_RULES, cls.ELABORATE_RULES,
            ))
        elif resolved_mode == "continuous_edit":
            resolved_system = "\n\n".join((
                resolved_system, cls.DEEP_EDIT_RULES, cls.CONTINUOUS_EDIT_RULES,
                f"Return exactly {requested_count} chronological scene_prompts.",
                "Keep these source assignments explicit where applicable: "
                + ", ".join(persistent_tags),
            ))
        elif resolved_mode == "continuous_elaborate":
            resolved_system = "\n\n".join((
                resolved_system, cls.DEEP_EDIT_RULES, cls.CONTINUOUS_EDIT_RULES,
                cls.CONTINUOUS_ELABORATE_RULES,
                f"Return exactly {requested_count} chronological scene_prompts.",
                "Keep these source assignments explicit where applicable: "
                + ", ".join(persistent_tags),
            ))
        elif resolved_mode == "continuous":
            persistent_contract = (
                "The following canonical source tags remain active and MUST appear "
                "verbatim in every scene_prompt: " + ", ".join(persistent_tags) + "."
                if persistent_tags else
                "Repeat every applicable connected canonical source tag verbatim in "
                "each scene_prompt where its assigned element remains visible."
            )
            resolved_system = "\n\n".join((
                "You are a compact multimodal continuation director for MiniMax H3. "
                "Inspect the connected visual references and obey the user's chronology. "
                "Return the required JSON schema with English directions and dialogue in its requested language.",
                cls.CONTINUOUS_RULES,
                persistent_contract,
                f"Return exactly {max(1, min(12, int(continuous_scene_count)))} "
                "chronological strings in scene_prompts, one compact prompt per scene.",
            ))

        if bool(i2v_mode):
            if not any(index == 1 for index, _image in pictures):
                raise ValueError("I2V Mode requires reference_image_1 as the opening frame.")
            resolved_system += "\n\n" + cls.I2V_RULES

        resolved_system += "\n\n" + cls.SPECIFICITY_RULES
        if is_sequence:
            resolved_system += "\n\n" + cls.CONTEXTUAL_REACTION_RULES
            if "gothic horror" in direction_text.lower():
                resolved_system += (
                    "\n\nGOTHIC HORROR EXECUTION: Within the requested location and permitted "
                    "edits, choose a small coherent set of gothic details: for example aged "
                    "carved wood, heavy drapery, tall shadowed windows, candlelight contrasting "
                    "with cold window light, oppressive negative space or distant timber creaks. "
                    "These are alternatives, not a checklist. Maintain the chosen atmosphere "
                    "across scenes without relocating the story to a castle, inventing "
                    "supernatural events, changing assigned wardrobe or adding gore. Use only "
                    "details compatible with the user's source-preservation constraints."
                )
        if direction_text:
            resolved_system += (
                "\n\nSELECTED DIRECTION CONTRACT:\n" + direction_text
                + "\nSelected non-none controls are requirements, not optional suggestions. "
                "The user's explicit request, "
                "source-role assignments and preservation constraints always take priority. "
                "Otherwise realize every selection concretely in the final prompt. "
                "Blend the second selection as a compatible accent; do not invent a new plot "
                "or change locked subjects, camera, actions, wardrobe or source audio. "
                "Express applicable hints through concrete choices inside the existing compact "
                "prompt, never as appended labels, a checklist or extra boilerplate. "
                "For continuous mode realize the selected camera/motion in every applicable "
                "scene, not just the first. If two selections conflict, the first is primary; "
                "use the second only in a compatible way rather than canceling the first. "
                "Before returning, check each prompt for explicit selected camera behavior, "
                "environment and wardrobe details, and source tags. "
                "Dialogue none or omitted means no override, not a ban on requested speech. "
                "A selected dialogue language guides permitted speech, but never replaces "
                "source audio the user asks to preserve. Retain canonical angle-bracket source tags."
            )

            if resolved_control_specs:
                resolved_system += (
                    "\n\nRESOLVED CONTROL SPECIFICATIONS:\n- "
                    + "\n- ".join(resolved_control_specs)
                    + "\nThese full specifications are binding. Their concrete behavior must "
                    "be visible or audible in the finished prompt; do not reduce them back to labels."
                )

            if resolved_mode in ("elaborate", "continuous_elaborate"):
                resolved_system += "\n\n" + cls.ELABORATE_DIRECTION_RULES
                if controls_only_generation:
                    resolved_system += (
                        "\n\nCONTROLS-ONLY ORIGINAL DIRECTION:\n"
                        "There is no user-written premise. Invent the complete concept yourself from "
                        "the visible connected sources and the selected controls. The controls are "
                        "the primary creative brief, not optional styling. Choose a specific role for "
                        "every visible subject, a concrete location compatible with source ownership, "
                        "a motivated action with chronological development, a camera design that "
                        "fully performs the selected motion, a recognizable realization of the genre "
                        "and look, an integrated soundscape and a memorable resolved ending. Do not "
                        "describe the act of inventing; output only the finished footage."
                    )

        if "orbit" in direction_text.lower():
            if resolved_mode in ("elaborate", "continuous_elaborate"):
                resolved_system += "\n\n" + cls.ELABORATE_ORBIT_RULES
            else:
                resolved_system += (
                    "\nSelected Orbit Camera: describe an actual arc around the subject, "
                    "with changing viewpoint and background parallax, not a zoom or pan in place. "
                    "Respect explicit source-camera preservation."
                )

        continuous_count = max(1, min(12, int(continuous_scene_count)))
        if is_sequence:
            response_schema = {
                "name": "h3_compact_continuous_scene_prompts",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "scene_prompts": {
                            "type": "array",
                            "minItems": continuous_count,
                            "maxItems": continuous_count,
                            "items": {
                                "type": "string",
                                "minLength": 1100 if resolved_mode == "continuous_elaborate" else 30,
                                **({"maxLength": 7000} if resolved_mode == "continuous_elaborate" else {}),
                            },
                        }
                    },
                    "required": ["scene_prompts"],
                    "additionalProperties": False,
                },
            }
        else:
            response_schema = {
                "name": "h3_compact_multimodal_edit_prompt",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "edit_type": {
                            "type": "string",
                            "enum": [
                                "background_environment", "identity_person",
                                "wardrobe_appearance", "object_replacement",
                                "pose_action_motion", "style_relight",
                                "composite_multi_edit", "general_edit"
                            ],
                        },
                        "source_roles": {
                            "type": "array", "minItems": 1,
                            "items": {"type": "string", "minLength": 8},
                        },
                        "visual_evidence": {"type": "string", "minLength": 20},
                        "edit_prompt": {
                            "type": "string",
                            "minLength": 1600 if resolved_mode == "elaborate" else 80,
                            **({"maxLength": 7000} if resolved_mode == "elaborate" else {}),
                        }
                    },
                    "required": [
                        "edit_type", "source_roles", "visual_evidence", "edit_prompt"
                    ],
                    "additionalProperties": False,
                },
            }

        payload = {
            "model": str(getattr(llm, "model", "") or ""),
            "messages": [
                {"role": "system", "content": resolved_system},
                {"role": "user", "content": content},
            ],
            "max_tokens": (
                max(int(max_tokens), 3000)
                if resolved_mode == "elaborate"
                else
                max(int(max_tokens), 1800)
                if resolved_mode == "deep_edit"
                else max(int(max_tokens), continuous_count * 1000)
                if resolved_mode == "continuous_elaborate"
                else max(int(max_tokens), continuous_count * 480)
                if resolved_mode == "continuous_edit"
                else max(int(max_tokens), continuous_count * 400)
                if resolved_mode == "continuous"
                else int(max_tokens)
            ),
            "temperature": float(temperature),
            "seed": int(seed),
            "reasoning": {"enabled": bool(reasoning)},
            "response_format": {
                "type": "json_schema",
                "json_schema": response_schema,
            },
            "provider": {"require_parameters": True},
        }
        debug_text = cls._debug_request(llm, payload) if debug_request else ""
        result = _external_llm_request(llm, payload)

        def parse_json_content(value):
            if not isinstance(value, str):
                return value
            candidate = value.strip()
            fenced = re.match(
                r"^```(?:json)?\s*(.*?)\s*```$", candidate,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if fenced:
                candidate = fenced.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as first_error:
                try:
                    return json.loads(candidate, strict=False)
                except json.JSONDecodeError:
                    start = candidate.find("{")
                    end = candidate.rfind("}")
                    if start >= 0 and end > start:
                        return json.loads(candidate[start:end + 1], strict=False)
                    raise first_error

        repair_retry_used = False
        try:
            raw = result["choices"][0]["message"]["content"]
            # Capture assistant content before parsing, tag repair or task headers.
            # Excludes the provider envelope, credentials and reasoning content.
            first_raw_text = (
                raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            )
            try:
                parsed = parse_json_content(raw)
                raw_response = first_raw_text if debug_request else ""
            except json.JSONDecodeError:
                repair_retry_used = True
                repair_payload = {
                    "model": str(getattr(llm, "model", "") or ""),
                    "messages": [
                        {"role": "system", "content": (
                            "Repair an interrupted or malformed director response. Return one "
                            "complete JSON object matching the supplied schema exactly. Preserve "
                            "all usable visual facts, source tags, chronology, creative direction "
                            "and wording from the partial response. Complete unfinished prose "
                            "coherently. Output JSON only, with valid escaping and closed strings."
                        )},
                        {"role": "user", "content": (
                            "DIRECTOR MODE: " + str(edit_mode)
                            + "\nUSER REQUEST: " + request_text
                            + ("\nDIRECTION CONTROLS:\n" + direction_text if direction_text else "")
                            + "\n\nMALFORMED OR INTERRUPTED RESPONSE:\n" + first_raw_text
                        )},
                    ],
                    "max_tokens": payload["max_tokens"],
                    "temperature": 0.0,
                    "seed": int(seed),
                    "reasoning": {"enabled": False},
                    "response_format": payload["response_format"],
                    "provider": {"require_parameters": True},
                }
                repair_result = _external_llm_request(llm, repair_payload)
                repaired_raw = repair_result["choices"][0]["message"]["content"]
                parsed = parse_json_content(repaired_raw)
                if debug_request:
                    repaired_text = (
                        repaired_raw if isinstance(repaired_raw, str)
                        else json.dumps(repaired_raw, ensure_ascii=False)
                    )
                    raw_response = (
                        "ATTEMPT 1 — INVALID\n" + first_raw_text
                        + "\n\nATTEMPT 2 — REPAIRED\n" + repaired_text
                    )
                result = repair_result
            if is_sequence:
                scene_prompts = [
                    cls._canonicalize_source_tags(str(item).strip())
                    for item in parsed["scene_prompts"]
                ]
                # Repair omitted persistent tags locally instead of spending
                # credits on a second OpenRouter request.
                if persistent_tags:
                    repaired_prompts = []
                    for item in scene_prompts:
                        missing = [tag for tag in persistent_tags if tag not in item]
                        if missing:
                            item = (
                                "Keep all visual assignments from "
                                + " and ".join(missing)
                                + " active in this scene. "
                                + item
                            )
                        repaired_prompts.append(item)
                    scene_prompts = repaired_prompts
                if len(scene_prompts) != continuous_count or any(
                    not item for item in scene_prompts
                ):
                    raise ValueError("continuous prompt count mismatch")
                edit_type = (
                    "continuous_elaborate_sequence"
                    if resolved_mode == "continuous_elaborate"
                    else "continuous_edit_sequence" if resolved_mode == "continuous_edit"
                    else "continuous_sequence"
                )
                source_roles = []
                visual_evidence = ""
            else:
                edit_type = str(parsed["edit_type"]).strip()
                source_roles = [str(role).strip() for role in parsed["source_roles"]]
                visual_evidence = str(parsed["visual_evidence"]).strip()
                edit_prompt = cls._canonicalize_source_tags(
                    str(parsed["edit_prompt"]).strip()
                )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Compact H3 Edit Director returned an invalid response.") from error

        if videos:
            required_header = (
                "[video editing + reference generation]"
                if pictures or has_mood_image or has_mood_video
                else "[video editing]"
            )
        else:
            required_header = "[reference generation]"
        header_pattern = (
            r"^\s*\[(?:video\s+editing(?:\s*\+\s*reference\s+generation)?|"
            r"reference\s+generation|image\s+editing(?:\s*\+\s*reference\s+generation)?)\]\s*"
        )
        if is_sequence:
            scene_prompts = [
                f"{required_header}\n\n" + re.sub(
                    header_pattern, "", item, count=1, flags=re.IGNORECASE
                ).strip()
                for item in scene_prompts
            ]
            edit_prompt = json.dumps(
                {"scene_prompts": scene_prompts}, ensure_ascii=False
            )
            preview_prompt = "\n\n".join(
                f"SCENE {index}\n{item}"
                for index, item in enumerate(scene_prompts, 1)
            )
        else:
            edit_prompt = re.sub(
                header_pattern, "", edit_prompt, count=1, flags=re.IGNORECASE,
            ).strip()
            edit_prompt = f"{required_header}\n\n{edit_prompt}"
            preview_prompt = edit_prompt

        connected_tags = {
            *(f"<Picture {index}>" for index, _image in pictures),
            *(f"<Video {index}>" for index, _video in videos),
        }
        if has_mood_image:
            connected_tags.add("<Mood Image 1>")
        if has_mood_video:
            connected_tags.add("<Mood Video 1>")
        warnings = []
        if not is_sequence:
            role_text = " ".join(source_roles)
            if not any(tag in role_text for tag in connected_tags):
                warnings.append("source roles lack a connected tag")
            for tag in connected_tags:
                if tag.lower() in request_text.lower() and tag not in edit_prompt:
                    warnings.append(f"explicit {tag} absent from final prompt")
            evidence_terms = {
                word.lower() for word in re.findall(r"[A-Za-z][A-Za-z-]{4,}", visual_evidence)
            }
            prompt_terms = {
                word.lower() for word in re.findall(r"[A-Za-z][A-Za-z-]{4,}", edit_prompt)
            }
            if len(evidence_terms & prompt_terms) < 2:
                warnings.append("weak visual-evidence integration")
            if resolved_mode == "elaborate":
                elaborate_word_count = len(re.findall(r"\b[\w'-]+\b", edit_prompt))
                if elaborate_word_count < 300:
                    warnings.append(
                        f"Elaborate prompt is underdeveloped ({elaborate_word_count} words)"
                    )

        displayed_mode = {
            "deep_edit": "edit", "elaborate": "Elaborate", "continuous": "Enhance",
            "continuous_edit": "Continuous Edit",
            "continuous_elaborate": "Continuous Elaborate",
        }.get(
            resolved_mode, resolved_mode
        )
        validation = (
            f"{displayed_mode} {edit_type} ready · {continuous_count if is_sequence else 1} prompt(s) · "
            f"{len(pictures)} picture(s) · "
            f"{len(videos)} video(s) · "
            f"{video_frame_total} sampled frame(s) · mood image "
            f"{'yes' if has_mood_image else 'no'} · mood video "
            f"{'yes' if has_mood_video else 'no'} · "
            f"I2V {'on' if i2v_mode else 'off'} · "
            + ("grounding verified" if not warnings else "warnings: " + "; ".join(warnings))
            + (" · JSON repaired after one retry" if repair_retry_used else "")
        )
        usage = result.get("usage") or {}
        usage_stats = (
            f"input: {usage.get('prompt_tokens', '?')} · "
            f"output: {usage.get('completion_tokens', '?')} · "
            f"total: {usage.get('total_tokens', '?')}"
        )
        if llm is not None:
            usage_stats += (
                f" · provider: {type(llm).__name__} · "
                f"model: {getattr(llm, 'model', 'external')}"
            )
        _set_held_director_plan(cache_key, {
            "cache_kind": "h3_compact_multimodal_edit",
            "provider_identity": provider_identity,
            "debug_request": debug_text,
            "raw_response": raw_response,
            "direction_context": direction_text,
            "resolved_mode": resolved_mode,
            "i2v_mode": bool(i2v_mode),
            "edit_prompt": edit_prompt,
            "preview_prompt": preview_prompt,
            "continuous_scene_count": continuous_count,
            "validation": validation,
        })
        return io.NodeOutput(
            edit_prompt, validation, usage_stats,
            max(1, min(12, int(continuous_scene_count))),
            debug_text,
            raw_response,
            ui=ui.PreviewText(preview_prompt),
        )


class H3StoryboardMotionDirector(io.ComfyNode):
    """Write direct video prompts after inspecting the accepted storyboard sheet."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3StoryboardMotionDirector",
            display_name="H3 Storyboard Motion Director — Image-Aware",
            category="text/minimax_h3",
            search_aliases=[
                "storyboard animation director", "panel motion prompts",
                "image aware h3 director",
            ],
            description=(
                "Acts as an automatic assistant director: it detects the real panel "
                "count from the accepted storyboard and writes one direct H3 motion "
                "prompt per panel. Language, performance, camera grammar and motion "
                "are inferred from the images and story."
            ),
            inputs=[
                io.String.Input(
                    "api_key", default="", placeholder="sk-or-v1-...",
                    extra_dict={"password": True},
                ),
                io.String.Input("model", default=DEFAULT_MODEL),
                io.Image.Input(
                    "storyboard_sheet",
                    tooltip="Connect the final accepted 1x2, 2x2, or 3x2 storyboard sheet.",
                ),
                io.String.Input(
                    "direction_context", multiline=True, dynamic_prompts=False,
                    default="", force_input=True,
                    tooltip=(
                        "Connect plan_json from the first Director. It contains the "
                        "original user intent and resolved production-direction contract. "
                        "The accepted storyboard still wins on visible opening-state detail."
                    ),
                ),
                io.Int.Input("max_tokens", default=3072, min=1024, max=8192),
                io.Float.Input("temperature", default=0.35, min=0.0, max=1.5, step=0.05),
                io.Boolean.Input("reasoning", default=False),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input(
                    "image_max_dimension", default=1536, min=768, max=2048,
                    step=64, advanced=True,
                ),
                io.Int.Input(
                    "timeout_seconds", default=300, min=30, max=900, advanced=True,
                ),
                io.Boolean.Input(
                    "hold_prompts", display_name="Hold Motion Prompts", default=False,
                    tooltip=(
                        "Reuse the last successful image-aware prompts without another "
                        "OpenRouter call. Changes are ignored until disabled."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("prompt_lines"),
                io.String.Output("first_prompt"),
                io.Int.Output("prompt_count"),
                io.String.Output("validation"),
                io.String.Output("usage_stats"),
                io.String.Output("credits_remaining"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls, api_key: str, model: str, storyboard_sheet, direction_context: str,
        max_tokens: int, temperature: float, reasoning: bool, seed: int,
        image_max_dimension: int, timeout_seconds: int,
        hold_prompts: bool = False, unique_id=None,
    ) -> io.NodeOutput:
        if storyboard_sheet is None or not hasattr(storyboard_sheet, "shape"):
            raise ValueError("Storyboard Motion Director requires a storyboard image.")
        shape = tuple(int(value) for value in storyboard_sheet.shape)
        if len(shape) < 3 or shape[1] < 1 or shape[2] < 1:
            raise ValueError(f"Invalid storyboard image shape: {shape}.")
        raw_context = str(direction_context or "").strip()
        try:
            parsed_context = json.loads(raw_context)
        except (TypeError, json.JSONDecodeError):
            parsed_context = None
        contract_count = 0
        if isinstance(parsed_context, dict):
            contract_shots = parsed_context.get("shots") or []
            if isinstance(contract_shots, list):
                contract_count = len([
                    shot for shot in contract_shots if isinstance(shot, dict)
                ])
        sheet_ratio = float(shape[2]) / float(shape[1])
        # The accepted production contract is authoritative. Overall sheet
        # aspect ratio cannot identify the grid now that cells may be square,
        # landscape or portrait: for example a 2x2 sheet of 16:9 panels has a
        # 16:9 total ratio and the old heuristic incorrectly called it 6 panels.
        count = contract_count if contract_count in (2, 4, 6) else min(
            ((2, 0.5), (4, 1.0), (6, 1.5)),
            key=lambda item: abs(sheet_ratio - item[1]),
        )[0]
        scene_duration_seconds = 5.0
        steps = 8
        cache_key = f"{cls.__name__}:{str(unique_id or 'default')}"
        if bool(hold_prompts):
            held = _get_held_director_plan(cache_key)
            if (
                held
                and held.get("cache_kind") == "storyboard_motion_prompts"
                and int(held.get("prompt_count", 0)) == count
            ):
                preview = held["preview"]
                return io.NodeOutput(
                    held["prompt_lines"], held["first_prompt"],
                    int(held["prompt_count"]), held["validation"] + " · HOLD",
                    "Held prompts · LLM not called", "Credits unchanged",
                    ui=ui.PreviewText(preview),
                )
            if held and held.get("cache_kind") == "storyboard_motion_prompts":
                raise RuntimeError(
                    "Hold Motion Prompts contains "
                    f"{int(held.get('prompt_count', 0))} prompts, but the current "
                    f"director contract requires {count}. No API call was made. "
                    "Disable Hold Motion Prompts once to regenerate the correct count."
                )

        key = str(api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "An OpenRouter API key is required in the node or the "
                "OPENROUTER_API_KEY environment variable."
            )
        columns, rows = {2: (1, 2), 4: (2, 2), 6: (3, 2)}[count]
        dialogue_rule = (
            "Infer whether dialogue exists and its natural language from the original "
            "story and visible scene. Never invent dialogue when the story is nonverbal. "
            "When dialogue is motivated, use MiniMax syntax "
            "`(Sx) says: <d>[Language] exact words</d>`."
        )
        system = f"""
You are the image-aware motion director for MiniMax H3 Reference to Video.
The supplied image is the final accepted storyboard: exactly {count} panels in a
{columns}-column by {rows}-row grid, read left-to-right across the top row and then
left-to-right across the bottom row. Analyze what is ACTUALLY visible in every panel.

Return exactly {count} self-contained production-ready video prompts in scene_prompts.
Every downstream clip receives only its own cropped panel as <Picture 1>.

DIRECTION CONTRACT AUTHORITY:
- The attached DIRECT PRODUCTION CONTRACT preserves the user's original request plus the
  first director's resolved genre, secondary genre, visual look, motion language, spoken
  language, sound policy, story bible and per-scene intentions. Treat every explicit value
  other than Auto or None as mandatory production direction.
- The current cropped panel remains authoritative for visible opening facts: present cast,
  wardrobe, hair, props, location, lighting, composition and pose. Never overwrite those
  pixels merely to recreate an earlier textual plan.
- Use sequence_contract to understand narrative purpose and boundaries. Reconcile its planned
  action with the accepted pixels, preserving intent while preventing the next panel's beat
  from leaking into the current clip.
- Resolve Auto values intelligently from the user intent, the complete storyboard progression
  and the current panel rather than defaulting to a generic cheerful or cinematic treatment.

BOUNDARY CONTRACT:
- Panel N is the literal visible opening state of clip N: cast, subject count, identity
  presentation, wardrobe, hair, props, location, lighting, composition and pose.
- The defining visual event already shown in panel N happens at the opening of clip N;
  never make clip N-1 perform it early.
- Panel N+1 is a RESERVED FUTURE BEAT. Clip N may create anticipation or a threshold
  state for it, but must not complete, reveal or duplicate its defining action, new cast,
  new wardrobe, new location or decisive composition.
- If panel N shows an impact, kiss, landing, reveal or other peak already in progress,
  begin at that exact peak and animate its completion and reaction instead of replaying setup.
- For the final panel, complete the requested outcome and settle on a resolved state.

SCENE ISOLATION:
- Mention only people visibly present in the current panel. Never import a person from
  another panel, the original references, or a shared global cast.
- State the exact visible wardrobe and location independently in every prompt. Never
  restore old clothing or a previous location. Never normalize intentional changes.
- <Picture 1> controls the opening visual state. The user story controls intended motion;
  when prose and pixels conflict, start from the pixels and adapt the motion coherently.
- Do not mention the full storyboard, grids, panels, future reference pictures or analysis
  inside the final prompts.

GENRE EXECUTION:
- Infer the active production genre from the original story intent and visible storyboard,
  then make it control performance, proximity, gaze, touch, camera relationship,
  pacing and sound in every clip, not only lighting or color treatment.
- For OnlyFans-style creator direction, use intimate self-produced energy, confident seductive
  body language, charged mutual attraction, tactile interaction and passionate escalation
  appropriate to the current beat. A public or introductory beat may remain teasing and restrained;
  a private payoff may become openly passionate. Avoid generic cheerful couple posing when the
  active genre calls for erotic or intimate tension.

CAMERA AND ACTION:
- Give every clip purposeful motion and a distinct coverage strategy appropriate to its
  visible composition and narrative function. Across the sequence vary wide, medium,
  profile, over-the-shoulder, close-up, insert/detail, high/low angle or tracking coverage
  when motivated; avoid repeating near-identical medium shots.
- Preserve readable anatomy, contact points, geography and cause-and-effect. Choose focused
  action and pacing that can be expressed clearly in one short H3 clip.
- Each prompt must contain: concise visible opening state; chronological movement;
  camera behavior; lighting; synchronized sound; optional dialogue; concrete ending state;
  and one short clause reserving the next panel's defining beat without describing it.
- {dialogue_rule}
- Infer a distinct, motivated motion and camera style for each shot instead of applying one
  global motion preset to the complete sequence.

OUTPUT:
Return only one JSON object containing one key, scene_prompts, whose value is an array
of exactly {count} complete strings. No other keys or commentary.
""".strip()
        if isinstance(parsed_context, dict):
            contract = dict(parsed_context.get("direction_contract") or {})
            sequence = []
            for index, shot in enumerate(parsed_context.get("shots") or [], 1):
                if not isinstance(shot, dict):
                    continue
                sequence.append({
                    "scene": index,
                    "storyboard_keyframe_intent": str(
                        shot.get("storyboard_prompt") or ""
                    ).strip(),
                    "planned_action": str(shot.get("prompt") or "").strip(),
                })
            contract["sequence_contract"] = sequence
            context_text = json.dumps(contract, ensure_ascii=False, indent=2)
        else:
            context_text = raw_context
        user_text = "\n\n".join((
            "DIRECT PRODUCTION CONTRACT:\n" + context_text,
            "Inspect the attached accepted storyboard and direct the motion now. "
            "Use the contract for intent and direction, while pixels remain authoritative "
            "for each clip's visible opening state.",
        ))
        content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {
                "url": _image_data_url(storyboard_sheet, int(image_max_dimension))
            }},
        ]
        payload = {
            "model": str(model).strip() or DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "reasoning": {"enabled": bool(reasoning)},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "storyboard_motion_prompts",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"scene_prompts": {
                            "type": "array", "minItems": count, "maxItems": count,
                            "items": {"type": "string", "minLength": 1},
                        }},
                        "required": ["scene_prompts"],
                        "additionalProperties": False,
                    },
                },
            },
            "provider": {"require_parameters": True},
        }
        result = _openrouter_request(key, payload, int(timeout_seconds))
        try:
            response_text = result["choices"][0]["message"]["content"]
            prompts = _parse_direct_prompt_response(response_text, count)
        except (KeyError, IndexError, TypeError, RuntimeError) as error:
            raise RuntimeError(
                f"Storyboard Motion Director did not return {count} usable prompts: {error}"
            ) from error
        usage = result.get("usage") or {}
        usage_stats = (
            f"input: {usage.get('prompt_tokens', '?')} · "
            f"output: {usage.get('completion_tokens', '?')} · "
            f"total: {usage.get('total_tokens', '?')}"
        )
        credits = _credits(key, min(30, int(timeout_seconds)))
        validation = (
            f"Image-aware motion prompts ready · {count} panels from "
            f"{'director contract' if contract_count in (2, 4, 6) else 'legacy ratio fallback'} · "
            f"sheet {shape[2]}×{shape[1]} · "
            "current beat protected from future-beat leakage · no creative JSON plan"
        )
        prompt_lines = _format_direct_chain_payload(
            prompts, scene_duration_seconds, steps, seed, "Cinematic Cuts"
        )
        preview = "\n\n".join(
            f"SCENE {index:02d}\n{prompt}"
            for index, prompt in enumerate(prompts, 1)
        )
        record = {
            "cache_kind": "storyboard_motion_prompts",
            "prompt_lines": prompt_lines,
            "first_prompt": prompts[0],
            "prompt_count": count,
            "validation": validation,
            "usage_stats": usage_stats,
            "credits_remaining": credits,
            "preview": preview,
        }
        _set_held_director_plan(cache_key, record)
        return io.NodeOutput(
            prompt_lines, prompts[0], count, validation, usage_stats, credits,
            ui=ui.PreviewText(preview),
        )


class SimpleH3PromptLinesToList(io.ComfyNode):
    """Convert tagged prompt blocks into a native ComfyUI STRING list."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SimpleH3PromptLinesToList",
            display_name="Simple H3 Prompt Lines → List",
            category="text/minimax_h3",
            search_aliases=["CR Prompt List", "prompt list", "scene prompt list"],
            description=(
                "Dependency-free prompt-list node. Each <<<H3_SCENE_NN>>> block "
                "becomes one independently executable STRING prompt. Legacy "
                "one-prompt-per-line input remains supported."
            ),
            inputs=[
                io.String.Input("prompt_lines", multiline=True, dynamic_prompts=False),
                io.String.Input("prepend_text", default=""),
                io.String.Input("append_text", default=""),
                io.Int.Input("start_index", default=0, min=0, max=9999),
                io.Int.Input("max_prompts", default=32, min=1, max=9999),
            ],
            outputs=[
                io.String.Output("prompt", is_output_list=True),
                io.String.Output("body_text", is_output_list=True),
                io.Int.Output("count"),
            ],
        )

    @classmethod
    def execute(cls, prompt_lines: str, prepend_text: str, append_text: str,
                start_index: int, max_prompts: int) -> io.NodeOutput:
        rows = _split_direct_prompt_block(prompt_lines)
        start = min(max(0, int(start_index)), len(rows))
        selected = rows[start:start + max(1, int(max_prompts))]
        if not selected:
            raise ValueError("Simple H3 Prompt Lines → List found no usable prompt lines.")
        prompts = [f"{prepend_text}{line}{append_text}" for line in selected]
        return io.NodeOutput(prompts, selected, len(selected))


class H3DirectPromptsToChainJSON(io.ComfyNode):
    """Build a deterministic sequential-chain plan from direct prompt blocks."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3DirectPromptsToChainJSON",
            display_name="Direct Prompts → Masked Chain",
            category="text/minimax_h3",
            search_aliases=[
                "direct prompts masked av", "prompt list chain plan",
                "sequential h3 prompts",
            ],
            description=(
                "Creates a minimal internal chain JSON from the experimental "
                "Director's tagged prompts. It does not call an LLM or rewrite "
                "the prompts. Connect plan_json to Simple H3 Chain Plan and use "
                "Simple H3 Context in masked_av for sequential continuity."
            ),
            inputs=[
                io.String.Input(
                    "direct_prompts", force_input=True,
                    tooltip="Connect H3 Director — Direct Scene Prompts prompt_lines.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "plan_json",
                    tooltip="Connect to Simple H3 Chain Plan plan_json_input.",
                ),
                io.Int.Output("scene_count"),
                io.String.Output("preview"),
            ],
        )

    @classmethod
    def execute(cls, direct_prompts: str) -> io.NodeOutput:
        metadata, _clean_blocks = _extract_direct_chain_metadata(direct_prompts)
        prompts = _split_direct_prompt_block(direct_prompts)
        if not prompts:
            raise ValueError("Direct Prompts → Masked Chain JSON found no scenes.")
        duration = float(metadata.get("scene_duration_seconds", 5.0))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("scene_duration_seconds must be finite and positive.")
        step_count = int(metadata.get("steps", 6))
        if step_count < 1:
            raise ValueError("steps must be at least 1.")
        seed = int(metadata.get("base_seed", 0)) & 0xFFFFFFFFFFFFFFFF
        director_mode = str(metadata.get("director_mode", "Continuous Story"))
        shots = []
        for index, prompt in enumerate(prompts, 1):
            # Stable, well-spaced uint64 seeds without changing prompt content.
            scene_seed = (
                seed + ((index - 1) * 0x9E3779B97F4A7C15)
            ) & 0xFFFFFFFFFFFFFFFF
            shots.append({
                "id": f"scene_{index:02d}",
                "prompt": prompt,
                "duration_seconds": duration,
                "steps": step_count,
                "seed": scene_seed,
            })
        plan = {
            "format": "simple_h3_direct_prompts_v1",
            "director_mode": str(director_mode),
            "defaults": {
                "duration_seconds": duration,
                "steps": step_count,
            },
            "shots": shots,
        }
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        preview = _format_direct_prompt_block(prompts)
        return io.NodeOutput(
            plan_json, len(shots), preview,
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
            "Plans one coherent horizontal storyboard strip plus independent Ref2VA "
            "shots. Panel N becomes the sole video reference for scene N."
        )
        for index, item in enumerate(schema.inputs):
            if item.id == "scene_count":
                schema.inputs[index] = io.Int.Input(
                    "scene_count", default=2, min=1, max=6,
                    tooltip=(
                        "Number of ordered storyboard panels and matching Ref2VA scenes. "
                        "Panels are always arranged left-to-right in one row."
                    ),
                )
                break
        return schema

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        storyboard_rules = """
MANDATORY STORYBOARD-CUT DIRECTION:
- Every scene is an independent cinematic shot beginning after a hard cut.
- Produce between one and six scenes exactly as requested. All storyboard_prompt
  compositions will be rendered simultaneously inside one horizontal strip,
  ordered strictly left-to-right. Design the complete strip as one coherent visual
  artifact so identity, wardrobe, hairstyle, accessories, props, architecture,
  palette, lighting logic and production design remain consistent across panels.
- Preserve identity, wardrobe, hairstyle, accessories, physical changes,
  persistent props, geography and narrative state across the complete story.
- Original reference poses, framing, expressions, camera angles and backgrounds are
  source observations, not continuity locks. Invent new poses and compositions for
  the storyboard unless the user explicitly asks to retain one of those properties.
- Never continue the previous camera trajectory, composition, exact pose or body
  motion. Freely choose the best new shot size, angle, lens, focus and movement.
- Never describe a seamless transition or one continuous take.
- Produce two completely separate reference vocabularies.
- storyboard_prompt_prefix is used only to generate still images. It must assign
  every connected original <Picture N> reference to stable subject labels S1,
  S2 and S3 as applicable, plus permanent visual rules.
- prompt_prefix is used only for video generation after the strip exists and is
  divided into individual panels. In it, <Picture 1> always means the single
  generated panel matching the current scene, never the complete strip and never
  an original uploaded reference. Define S1, S2 and S3 as
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
- Write storyboard_prompt_prefix and every storyboard_prompt in native MiniMax
  image-prompt language: concrete Subject/Picture bindings, retention intent,
  one finished composition, precise visible detail and no video terminology.
- Before writing the storyboard prompts, privately design one purposeful visual
  coverage progression for the complete strip. Every panel must have a distinct
  dramatic job: establish geography, reveal a relationship, emphasize an action,
  isolate a reaction, expose an important detail, intensify the conflict, or land
  the outcome. Do not output this private coverage plan as labels or metadata.
- First divide the complete user request into the requested number of successive
  cause-and-effect story milestones. Distribute the action across the whole strip:
  do not complete or summarize the entire request in panel 1, and do not merely
  repeat its climax in later panels. Each panel begins from the visible consequence
  of the preceding milestone and contributes one new readable development.
- Treat the panels as selected key moments from one coherent chronological sequence,
  not independent variations of one prompt. Preserve geography, screen direction,
  character objectives, current physical state and persistent changes across cuts,
  while allowing motivated location changes explicitly required by the story.
- Adjacent panels may not repeat substantially the same shot scale, camera height,
  camera side, angle, lens impression, subject arrangement or negative-space
  pattern. A minor crop, small zoom or slightly shifted camera is not sufficient.
- Adjacent panels must also depict different chronological action phases. Never
  write the same physical pose and activity twice with only intensified adjectives.
  Panel N must visibly inherit the result of panel N-1 and advance to a new readable
  state. For two panels, use a clear setup/cause in panel 1 and a later decisive
  development, reaction, consequence or resolution in panel 2.
- Choose the widest useful narrative separation that remains coherent for the
  available panel count. With only two panels, select two substantially different
  milestones rather than two neighboring instants of the same repeated motion.
- Characters act toward their objective, environment and one another. Their gaze,
  posture and hands must be task-motivated; never make them stop, pose, present the
  action to the viewer or look into the camera unless the user explicitly asks.
- Use deliberate cinematic variety selected for the story: establishing wide or
  full shots, medium two-shots, over-the-shoulder views, profiles, low/high angles,
  close-ups, extreme close-ups, inserts, foreground occlusion, depth staging and
  asymmetric compositions. Choose only useful coverage; do not cycle mechanically
  through a generic shot list.
- Preserve screen direction and spatial geography across cuts. Cross the axis only
  when the new composition clearly re-establishes orientation. Maintain coherent
  eyelines, subject handedness, prop positions and entrances/exits.
- Escalate visual emphasis across the strip. Reserve the strongest close-up,
  unusual angle or most graphic composition for the most important story beat;
  do not spend every panel at the same intensity.
- storyboard_prompt must state its exact shot scale, viewpoint, camera height,
  angle, foreground/midground/background arrangement, subject placement, eyelines,
  lens character and focal priority in concrete visible terms.
- Do not write vague phrases such as cinematic angle, dynamic composition or a
  slightly different view. Make the new camera geometry unmistakable from visible
  foreground, profile/face orientation, body overlap, depth layers and screen space.
- Scene prompt N must animate only panel N. The panel is authoritative for visible
  wardrobe, hairstyle, accessories, props, environment, lighting, composition and
  character placement. Do not restore visual details from the original uploads.
- Keep the video-side prompt_prefix extremely short. Do not redescribe identity,
  wardrobe, environment, lighting or composition already visible in the current
  panel. Each normal scene prompt should contain only the chronological action,
  expression/performance changes, camera movement, dialogue, sound and the exact
  ending state needed by that clip. Repeat a visual fact only when it changes.
- The storyboard side carries visual design; the Ref2VA side carries motion and
  sound. Never duplicate the complete storyboard description inside a video prompt.
- Write every normal scene prompt so the action begins naturally from the visible
  state designed in its matching panel, while allowing a deliberate camera move
  inside that shot. Do not describe interpolation toward the following panel.
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


class H3StoryDirectorStoryboardBlocks(H3StoryDirector):
    """Plan 1-4 storyboard sheets whose panels animate inside one clip each."""

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3StoryDirectorStoryboardBlocks"
        schema.display_name = "H3 Story Director — Multi-Storyboard Clips"
        schema.search_aliases = [
            "h3 storyboard blocks", "multi storyboard video", "storyboard clip director"
        ]
        schema.description = (
            "Creates 1-4 chronological storyboard blocks with 1, 2, or 4 panels "
            "per block. Each complete sheet later drives one Ref2VA clip."
        )
        replaced = []
        for item in schema.inputs:
            if item.id == "scene_count":
                replaced.extend((
                    io.Int.Input(
                        "storyboard_count", default=2, min=1, max=4,
                        tooltip="Number of complete storyboard sheets and resulting clips.",
                    ),
                    io.Combo.Input(
                        "panels_per_storyboard", options=["1", "2", "4"], default="4",
                        tooltip="Ordered panels interpreted inside every generated clip.",
                    ),
                ))
            elif item.id == "scene_duration_seconds":
                replaced.append(io.Combo.Input(
                    "clip_duration_seconds", options=["5", "10", "15"], default="10",
                    tooltip="Duration of every storyboard-driven video clip.",
                ))
            else:
                replaced.append(item)
        schema.inputs = replaced
        schema.outputs.append(io.String.Output(
            "block_config",
            tooltip="Connect to Simple H3 Storyboard Block Plan and Context.",
        ))
        return schema

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        storyboard_count = max(1, min(4, int(kwargs.pop("storyboard_count", 2))))
        panels_per = int(kwargs.pop("panels_per_storyboard", "4"))
        if panels_per not in (1, 2, 4):
            raise ValueError("panels_per_storyboard must be 1, 2, or 4.")
        clip_duration = int(kwargs.pop("clip_duration_seconds", "10"))
        if clip_duration not in (5, 10, 15):
            raise ValueError("clip_duration_seconds must be 5, 10, or 15.")
        total_panels = storyboard_count * panels_per
        kwargs["scene_count"] = total_panels
        kwargs["scene_duration_seconds"] = max(1.0, clip_duration / panels_per)
        block_rules = f"""
MANDATORY MULTI-STORYBOARD BLOCK CONTRACT:
- Plan exactly {storyboard_count} consecutive storyboard blocks with exactly
  {panels_per} ordered panels in every block ({total_panels} panels total).
- Block B contains scenes ((B-1)*{panels_per}+1) through (B*{panels_per}).
- Every complete block will be supplied to MiniMax H3 as one <Picture 1> and
  animated as one {clip_duration}-second video with internal motivated cuts.
- Within each block, distribute the chronological action across all panels and
  design distinctly different, readable camera coverage. Panel 1 establishes
  that block; the final panel lands its local consequence or handoff state.
- Across block boundaries preserve identity, wardrobe, physical state, props,
  geography, screen direction, story logic, soundscape and musical development.
- Normal scene prompts must describe the motion beginning at that panel and its
  chronological development. Do not use original Picture tags in video prompts.
""".strip()
        kwargs["system_prompt"] = "\n\n".join((
            str(kwargs.get("system_prompt") or "").strip(), block_rules,
        )).strip()
        result = super().execute(**kwargs)
        config_data = {
            "version": 1,
            "storyboard_count": storyboard_count,
            "panels_per_storyboard": panels_per,
            "clip_duration_seconds": clip_duration,
            "total_panels": total_panels,
        }
        config = json.dumps(
            config_data, ensure_ascii=False, separators=(",", ":")
        )
        output_args = list(result.args)
        try:
            enriched_plan = json.loads(str(output_args[0]))
            enriched_plan["storyboard_block_config"] = config_data
            output_args[0] = json.dumps(enriched_plan, ensure_ascii=False)
        except (IndexError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Multi-Storyboard Director could not attach its block settings to the plan."
            ) from error
        return io.NodeOutput(
            *output_args, config, ui=result.ui, expand=result.expand,
            block_execution=result.block_execution,
        )


__all__ = [
    "H3StoryDirector",
    "H3DirectPromptDirector",
    "SimpleH3PromptLinesToList",
    "H3StoryDirectorStoryboardCuts",
    "H3StoryDirectorStoryboardBlocks",
]


