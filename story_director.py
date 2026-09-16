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

import folder_paths
import numpy as np
from comfy_api.latest import io, ui
from PIL import Image, ImageDraw, ImageFont


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
DEFAULT_MODEL = "x-ai/grok-4.20"
_DIRECTOR_HOLD_CACHE = {}
_DIRECTOR_HOLD_LOCK = threading.Lock()
_DIRECTOR_HOLD_CACHE_LOADED = False


def _director_hold_cache_path() -> str:
    """Return a local, non-repository path for persistent held plans."""
    cache_directory = os.path.join(
        folder_paths.get_output_directory(), "Sexy AI Studio", "director_hold"
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

    H3_NATIVE_PROMPT_RULES = """
MINIMAX H3 NATIVE PROMPT COMPILATION
Compile the creative direction into MiniMax H3's native plain-text prompt grammar. Richness means
precise control that fits the available screen time, not maximum length. Use these four section
headings exactly, without Markdown bolding, and separate sections and shots with blank lines:

subject_definitions:
integrated_multimodal_description:
overall_soundscape:
non_diegetic_music:

- Define each persistent visible person or independently controlled object as <Subject N>. Bind a
  referenced subject to its canonical source in the definition, for example `<Subject 1> the woman
  fully referenced from <Picture 1>, ...`. Include only stable, visibly distinguishing traits needed
  to prevent identity, wardrobe or object drift. If no subjects exist, write `N/A`.
- In integrated_multimodal_description use `[Shot 1]` without a timecode. Add `[Shot 2] At
  00:SS.mmm, ...` only when a real cut or distinct timed beat is useful and physically achievable.
  Keep spatial positions explicit in multi-subject coverage. Describe important props and contacts
  in a framing where they can actually be seen.
- Bind every spoken line to its visible speaker with both the canonical subject tag and a stable
  speaker ID: `<Subject N> Name (Sx) says: <d>[Language] exact dialogue</d>`. Speaker IDs follow the
  chronological order of first vocal events; do not assume S1 equals Subject 1. Keep user-supplied
  dialogue verbatim. A visible non-speaker during another voice remains physically active but has
  mouth closed and speaks no dialogue. Do not create a long static silence beat.
- Use `<scenetrans>` only when one spoken line intentionally continues across a cut, `<cutoff>` only
  when speech is intentionally interrupted, and off-screen voiceover only when requested or clearly
  necessary; explicitly keep the visible subject's lips closed during voiceover.
- overall_soundscape contains only diegetic ambience, room tone and physical Foley, never dialogue
  text or score. non_diegetic_music contains only score direction, or `N/A` when no score is wanted.
- Finish the available time with continuing physical behavior or a completed result, not a frozen
  stare, generic hold, unexplained dead air or a second copy of the opening composition.
- Prefer positive, direct description. Use a narrowly targeted constraint only when it prevents a
  likely H3 failure such as duplicate subjects, voice swapping, unstable wardrobe or broken contact.

For I2V, place this exact alignment sentence before subject_definitions, followed by a blank line:
`For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
Then make [Shot 1] emerge directly from the visible frame-zero state.
""".strip()

    ELABORATE_DURATION_RULES = """
SCREEN-TIME CONTRACT — {seconds:.2f} SECONDS PER SCENE
Every scene prompt controls one clip of {seconds:.2f} seconds. Design only what can be clearly
performed and perceived in that interval. Budget dialogue at roughly 2.5 spoken words per second,
leaving time for breath, reactions, camera travel and physical actions. Do not use the word budget
as a quota and do not add dialogue merely to fill time.

For 1–5 seconds, use one focused beat and normally one continuous shot. For over 5–10 seconds, use
one developed action with at most two or three readable phases. For over 10–15 seconds, a compact
multi-beat scene or a small number of motivated cuts is possible. Place later-shot timecodes inside
the {seconds:.2f}-second boundary and derive them from the actual action/dialogue load rather than
reusing fixed timestamps. If speech ends early, complete the clip with a concrete continuing action
and resulting state. Never compress several complex actions or a dense exchange into an implausible
interval. Favor fewer precise, executable details over prose that competes for model attention.
""".strip()

    CONTINUOUS_H3_HANDOFF_RULES = """
ONE-PASS LOGICAL CONTINUITY FOR H3 CHAINS
Plan the complete sequence in this single response; no generated last-frame inspection will occur.
Therefore make every boundary deterministic in prose. End scene N on a specific evolving state:
framing, camera direction and momentum, visible cast positions, gaze, body configuration, hand/prop
contacts, active motion, lighting phase and sound phase. Begin scene N+1 from that same planned state
before advancing it. Do not reset to the original reference pose, re-establish the location, replay
a completed action or repeat the opening composition as the closer.

Treat each continuation as one unbroken evolution from its inherited opening. Do not place a hard
cut at a scene boundary. Camera paths continue from the prior endpoint, with coherent screen axis,
parallax and direction. End each scene on a moving or causally active closer that gives the next
scene useful motion to inherit.

Plan speaker handoffs explicitly. If the person who speaks first in scene N+1 is not the person
visible/speaking at the end of scene N, begin with a brief silent inherited beat, move or reframe
without cutting until the new speaker is visibly established, and start that speaker's <d> dialogue
only afterward. State that the outgoing visible person has mouth closed and speaks no dialogue
during the handoff. If the same visible speaker continues, speech may continue naturally without
inventing a new establishing pause. Keep a sequence-wide lighting lock, stable subject definitions,
screen positions, wardrobe, object counts and already-used dialogue while still advancing the action.
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
                io.Float.Input(
                    "seconds_per_scene", display_name="Seconds per Scene — Elaborate",
                    default=5.0, min=1.0, max=15.0, step=0.5, force_input=True,
                    tooltip=(
                        "Connected FLOAT containing the 1–15 second screen-time budget used by "
                        "Elaborate and Continuous Elaborate. Other modes ignore it."
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
        i2v_mode: bool = False, seconds_per_scene: float = 5.0,
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
        text_only = not (pictures or videos or has_mood_image or has_mood_video)

        is_sequence = requested_mode in (
            "continuous", "continuous_edit", "continuous_elaborate",
        )
        requested_count = max(1, min(12, int(continuous_scene_count)))
        try:
            scene_seconds = float(seconds_per_scene)
        except (TypeError, ValueError) as error:
            raise ValueError("seconds_per_scene must be a number between 1 and 15.") from error
        if not math.isfinite(scene_seconds) or not 1.0 <= scene_seconds <= 15.0:
            raise ValueError("seconds_per_scene must be between 1 and 15 seconds.")
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
                cls.H3_NATIVE_PROMPT_RULES,
                cls.ELABORATE_DURATION_RULES.format(seconds=scene_seconds),
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
                cls.CONTINUOUS_ELABORATE_RULES, cls.H3_NATIVE_PROMPT_RULES,
                cls.ELABORATE_DURATION_RULES.format(seconds=scene_seconds),
                cls.CONTINUOUS_H3_HANDOFF_RULES,
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

        if text_only:
            resolved_system += (
                "\n\nTEXT-ONLY GENERATION: No visual references are connected. "
                "Create the requested scene from the user's text and direction controls. "
                "Do not claim to have inspected images or invent reference tags. "
                "Source-preservation instructions apply only to connected sources; "
                "there are none. Keep the required output schema and scene count."
                " For text-only output, source_roles must be empty and visual_evidence "
                "must state that no visual source was supplied."
            )

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
        elaborate_min_chars = 450 if scene_seconds <= 5 else 700 if scene_seconds <= 10 else 900
        elaborate_tokens_per_scene = 500 if scene_seconds <= 5 else 700 if scene_seconds <= 10 else 900
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
                                "minLength": elaborate_min_chars if resolved_mode == "continuous_elaborate" else 30,
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
                            "type": "array", "minItems": 0 if text_only else 1,
                            "items": {"type": "string", "minLength": 8},
                        },
                        "visual_evidence": {"type": "string", "minLength": 20},
                        "edit_prompt": {
                            "type": "string",
                            "minLength": elaborate_min_chars if resolved_mode == "elaborate" else 80,
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
                max(int(max_tokens), elaborate_tokens_per_scene)
                if resolved_mode == "elaborate"
                else
                max(int(max_tokens), 1800)
                if resolved_mode == "deep_edit"
                else max(int(max_tokens), continuous_count * elaborate_tokens_per_scene)
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
            if connected_tags and not any(tag in role_text for tag in connected_tags):
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
            if not text_only and len(evidence_terms & prompt_terms) < 2:
                warnings.append("weak visual-evidence integration")
            if resolved_mode == "elaborate":
                elaborate_word_count = len(re.findall(r"\b[\w'-]+\b", edit_prompt))
                minimum_words = 70 if scene_seconds <= 5 else 110 if scene_seconds <= 10 else 150
                if elaborate_word_count < minimum_words:
                    warnings.append(
                        f"Elaborate prompt is underdeveloped ({elaborate_word_count} words)"
                    )
        if resolved_mode in ("elaborate", "continuous_elaborate"):
            native_prompts = scene_prompts if is_sequence else [edit_prompt]
            required_sections = (
                "subject_definitions:", "integrated_multimodal_description:",
                "overall_soundscape:", "non_diegetic_music:",
            )
            for index, native_prompt in enumerate(native_prompts, 1):
                missing_sections = [section for section in required_sections if section not in native_prompt]
                if missing_sections:
                    label = f"scene {index} " if is_sequence else ""
                    warnings.append(label + "missing H3 sections: " + ", ".join(missing_sections))
                if bool(i2v_mode) and "at 0.00 seconds into the target video" not in native_prompt:
                    warnings.append((f"scene {index} " if is_sequence else "") + "missing I2V alignment")

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
            + (f"{scene_seconds:g}s/scene · " if resolved_mode in ("elaborate", "continuous_elaborate") else "")
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


