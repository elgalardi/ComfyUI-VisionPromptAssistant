from __future__ import annotations

import base64
import io as binary_io
import json
import math
import os
import re
import urllib.error
import urllib.request

import numpy as np
from comfy_api.latest import io, ui
from comfy_execution.graph_utils import ExecutionBlocker
from PIL import Image


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
DEFAULT_MODEL = "x-ai/grok-4.20"
LANGUAGES = ["English", "Español", "日本語"]
GENRES = [
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
    "Erotic Drama (Adults 18+)",
    "Erotic Thriller (Adults 18+)",
    "Explicit Adult Film (Adults 18+)",
]
DEFAULT_SYSTEM_PROMPT = """You are the story director and continuity supervisor for MiniMax H3 Reference-to-Video productions. Turn the user's simple idea and reference pictures into one complete visual story with a clear beginning, development, and ending.

Treat every connected reference as a distinct person or subject. Use the exact tags <Picture 1>, <Picture 2>, and <Picture 3> when they are supplied. Define stable subject labels S1, S2, and S3 in the shared prompt. Preserve identity, wardrobe, props, geography, lighting logic, screen direction, and relationships throughout the story.

Write production-ready MiniMax H3 prompts. Every scene must state the visible action, camera framing and movement, environment, lighting, dialogue when useful, and diegetic sound. Scene 1 establishes the story. Every later scene must explicitly continue the final pose, movement, object state, camera direction, and location established by the preceding scene. End every non-final scene on a clear unfinished action that the next scene can continue. The last scene resolves the user's idea with a deliberate ending.

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


def _story_schema(scene_count: int) -> dict:
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
                    "Complete production prompt for this scene, including continuity, "
                    "visible action, camera, lighting, dialogue when useful, and sound."
                ),
            },
        },
        "required": ["id", "prompt"],
        "additionalProperties": False,
    }
    return {
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
                    "each supplied Picture tag and all permanent continuity rules."
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


def _safe_id(value: str, index: int) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip())
    value = value.strip("_-")[:80]
    return value or f"scene_{index:02d}"


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


def _compile_story(
    raw: dict,
    scene_count: int,
    duration_seconds: float,
    steps: int,
    picture_count: int,
) -> tuple[str, str, str, str]:
    synopsis = str(raw.get("synopsis") or "").strip()
    story_bible = str(raw.get("story_bible") or "").strip()
    prompt_prefix = str(raw.get("prompt_prefix") or "").strip()
    shots = raw.get("shots")
    if not synopsis or not story_bible or not prompt_prefix:
        raise RuntimeError("The director response is missing its synopsis, story bible, or shared prompt.")
    if not isinstance(shots, list) or len(shots) != scene_count:
        actual = len(shots) if isinstance(shots, list) else 0
        raise RuntimeError(
            f"The director returned {actual} scenes, but {scene_count} were requested. The plan was not accepted."
        )

    missing_tags = [
        f"<Picture {index}>"
        for index in range(1, picture_count + 1)
        if f"<Picture {index}>" not in prompt_prefix
    ]
    if missing_tags:
        raise RuntimeError(
            "The shared prompt did not assign every connected reference: "
            + ", ".join(missing_tags)
        )

    compiled_shots = []
    seen_ids = set()
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            raise RuntimeError(f"Scene {index} is not a structured scene object.")
        shot_id = _safe_id(shot.get("id", ""), index)
        if shot_id in seen_ids:
            shot_id = f"{shot_id}_{index:02d}"
        seen_ids.add(shot_id)
        prompt = str(shot.get("prompt") or "").strip()
        if len(prompt) < 80:
            raise RuntimeError(
                f"Scene {index} is too short to be a production-ready continuity prompt."
            )
        compiled_shots.append({"id": shot_id, "prompt": prompt})

    plan = {
        "prompt_prefix": prompt_prefix,
        "defaults": {
            "duration_seconds": float(duration_seconds),
            "steps": int(steps),
        },
        "shots": compiled_shots,
    }
    validation = (
        f"Valid: {scene_count} scenes · {picture_count} references · "
        f"{float(duration_seconds):g}s requested per scene · {int(steps)} steps"
    )
    return json.dumps(plan, ensure_ascii=False, indent=2), story_bible, synopsis, validation


def _openrouter_request(api_key: str, payload: dict, timeout_seconds: int) -> dict:
    request = urllib.request.Request(
        OPENROUTER_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
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
        raise RuntimeError(f"OpenRouter returned HTTP {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not connect to OpenRouter: {error.reason}") from error


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


class H3StoryDirector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3StoryDirector",
            display_name="H3 Story Director",
            category="text/minimax_h3",
            search_aliases=["story planner", "h3 director", "openrouter story"],
            description=(
                "Turns a simple story idea and up to three reference images into a "
                "validated MiniMax H3 Contex Loop plan through OpenRouter."
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
                    default=(
                        "The two musicians build their music studio from an empty "
                        "room and finish by recording their first song together."
                    ),
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=DEFAULT_SYSTEM_PROMPT,
                ),
                io.Image.Input("image_0", optional=True),
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
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
                io.Combo.Input("genre", options=GENRES, default="Cinematic Drama"),
                io.Combo.Input(
                    id="language",
                    display_name="Diálogo",
                    options=LANGUAGES,
                    default="English",
                    tooltip=(
                        "Dialogue and spoken-voice language only. The synopsis, "
                        "story bible, plan, and MiniMax H3 scene prompts are always "
                        "written in English."
                    ),
                ),
                io.String.Input(
                    "additional_direction",
                    multiline=True,
                    default=(
                        "Keep both protagonists active in the story. Use realistic "
                        "room construction, equipment setup, and music-production sound."
                    ),
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
            ],
            outputs=[
                io.String.Output("plan_json"),
                io.String.Output("story_bible"),
                io.String.Output("synopsis"),
                io.String.Output("validation"),
                io.String.Output("usage_stats"),
                io.String.Output("credits_remaining"),
            ],
        )

    @classmethod
    def execute(
        cls,
        api_key: str,
        model: str,
        story_idea: str,
        system_prompt: str,
        scene_count: int,
        scene_duration_seconds: float,
        steps: int,
        draft_only: bool,
        genre: str,
        language: str,
        additional_direction: str,
        max_tokens: int,
        temperature: float,
        reasoning: bool,
        seed: int,
        image_max_dimension: int,
        timeout_seconds: int,
        image_0=None,
        image_1=None,
        image_2=None,
    ) -> io.NodeOutput:
        api_key = str(api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not api_key:
            raise ValueError(
                "An OpenRouter API key is required in the node or the "
                "OPENROUTER_API_KEY environment variable."
            )
        if not str(story_idea or "").strip():
            raise ValueError("A story idea is required.")
        scene_count = int(scene_count)
        if not 1 <= scene_count <= 32:
            raise ValueError("scene_count must be between 1 and 32.")

        pictures = [image for image in (image_0, image_1, image_2) if image is not None]
        adult_direction = ""
        if "Adults 18+" in genre:
            adult_direction = (
                " This is an adults-only genre. Every depicted participant must be "
                "an explicitly consenting adult aged 18 or older. Never create sexual "
                "content involving a minor or a person whose age is ambiguous."
            )

        content = [{
            "type": "text",
            "text": (
                f"Create exactly {scene_count} "
                f"{'standalone scene' if scene_count == 1 else 'connected scenes'}. "
                f"Genre: {genre}. "
                "Write the synopsis, story bible, prompt prefix, and every scene "
                "prompt entirely in English. "
                f"Only dialogue, lyrics, narration, and other spoken words may be "
                f"written in {language}; describe their delivery and surrounding "
                "audio instructions in English. If a scene has no spoken content, "
                "do not add dialogue merely to demonstrate the selected language. "
                "Keep MiniMax tags such as <Picture 1> unchanged. "
                f"Each scene will be generated "
                f"for approximately {float(scene_duration_seconds):g} seconds.\n\n"
                f"Story idea:\n{str(story_idea).strip()}\n\n"
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

        payload = {
            "model": str(model or DEFAULT_MODEL).strip(),
            "messages": [
                {"role": "system", "content": str(system_prompt or "").strip()},
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
                    "schema": _story_schema(scene_count),
                },
            },
            "provider": {"require_parameters": True},
        }
        result = _openrouter_request(api_key, payload, int(timeout_seconds))
        try:
            content_text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("OpenRouter returned an unexpected response.") from error
        raw_story = _parse_json_response(content_text)
        plan_json, story_bible, synopsis, validation = _compile_story(
            raw_story,
            scene_count,
            float(scene_duration_seconds),
            int(steps),
            len(pictures),
        )

        usage = result.get("usage") or {}
        usage_stats = (
            f"input: {usage.get('prompt_tokens', '?')} · "
            f"output: {usage.get('completion_tokens', '?')} · "
            f"total: {usage.get('total_tokens', '?')}"
        )
        credits = _credits(api_key, min(30, int(timeout_seconds)))
        preview = f"{synopsis}\n\n{validation}\n\n--- PLAN JSON ---\n{plan_json}"
        plan_output = ExecutionBlocker(None) if draft_only else plan_json
        return io.NodeOutput(
            plan_output,
            story_bible,
            synopsis,
            validation,
            usage_stats,
            credits,
            ui=ui.PreviewText(preview),
        )


__all__ = ["H3StoryDirector"]
