from __future__ import annotations

import json
import re

from comfy_api.latest import io, ui

from .story_director import (
    H3CompactMultimodalEditDirector,
    _external_llm_request,
    _get_held_director_plan,
    _image_data_url,
    _set_held_director_plan,
)


class LTX25CompactI2VDirector(io.ComfyNode):
    """Compact multimodal director specialized for LTX 2.5 image-to-video."""

    DEFAULT_SYSTEM_PROMPT = """
You are a precise multimodal director for LTX 2.5 image-to-video with synchronized audio.
Inspect the supplied FIRST FRAME and turn the user's request into one production-ready English
prompt. Return only the requested JSON. Never mention an input image, first frame, reference,
analysis, model, node, or these rules in the generated prompt.

The supplied image is the literal opening frame and highest-priority visual anchor. Begin from its
visible subjects, count, identity-defining appearance, wardrobe, objects, setting, composition,
framing, viewpoint, depth planes, materials, lighting and color. Do not replace, duplicate or add a
person or prominent object unless explicitly requested. If the request conflicts with the image,
obey the request by describing a visible transition from the established opening state.

Write a single flowing cinematic paragraph, normally 90–180 words and never over 200 words. Start
directly with the main action. Describe events chronologically using active present-progressive
language and concrete temporal connectors such as as, while, then and finally. Include only useful
details: subject movements and reactions, stable appearance, environment and depth, shot size,
camera angle and movement, lens/depth-of-field behavior, lighting and color evolution, then any
requested change or concluding beat. Preserve spatial continuity, screen direction and consistent
geometry. Do not format the result as a shot list and do not add headings or negative prompts.

Treat camera behavior literally. Distinguish a pan, tilt, dolly, truck, crane, handheld move, orbit,
zoom and locked shot; do not substitute one for another. Keep the camera stable when no movement is
requested unless a restrained movement clearly supports the user's intent. Use physically coherent
parallax, occlusion, reflections, shadows and focus changes.

Integrate the soundscape in the same chronology: ambience, concrete sound effects, speech and music
only when requested or naturally supported. Keep dialogue brief and verbatim when supplied. Do not
invent narration, dialogue, lyrics, characters, plot events, cuts or location changes. Favor one
coherent continuous shot unless the user explicitly requests cuts or multiple shots.
""".strip()

    DEEP_RULES = """
DETAILED MODE
Privately audit the visible opening state, every explicit user request, chronological motion,
camera behavior and audio before writing. Preserve all visible properties outside the requested
change. Resolve interactions with clear participants, contact, weight, reaction and outcome. Spend
the available detail on concrete motion and cinematography, not explanations or repeated adjectives.
The finished prompt remains one flowing paragraph of no more than 200 words.
""".strip()

    SEQUENCE_RULES = """
SEQUENCE MODE
Return exactly the requested number of standalone prompts in chronological order. Each prompt is
one flowing LTX 2.5 paragraph of 70–180 words and describes one contiguous generated block. Scene 1
begins from the supplied opening frame. Later scenes begin from the established final state of the
previous scene, repeat the minimum stable appearance and environment details needed to prevent
drift, and advance rather than replay the action. Do not announce continuity, refer to another
scene, or invent cuts. Reach the user's requested outcome in the final scene.
""".strip()

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTX25CompactI2VDirector",
            display_name="LTX 2.5 Director — Compact I2V",
            category="text/ltx25",
            search_aliases=[
                "ltx 2.5 compact director", "ltx i2v director",
                "ltx image to video prompt",
            ],
            description=(
                "LTX 2.5 counterpart to the compact H3 director. It inspects one "
                "I2V opening image and writes native chronological audiovisual prompts."
            ),
            inputs=[
                io.String.Input(
                    "video_request", multiline=True, dynamic_prompts=False,
                    default="Describe the action, camera behavior and sound for this video.",
                ),
                io.Combo.Input(
                    "director_mode",
                    options=["compact", "edit", "Enhance", "Continuous Edit"],
                    default="compact",
                    tooltip=(
                        "compact returns one concise prompt; edit returns one more rigorous "
                        "prompt; Enhance and Continuous Edit return a JSON scene_prompts list."
                    ),
                ),
                io.Image.Input(
                    "first_frame",
                    tooltip="The same opening image supplied to LTX 2.5 I2V conditioning.",
                ),
                io.String.Input(
                    "system_prompt", multiline=True, dynamic_prompts=False,
                    default=cls.DEFAULT_SYSTEM_PROMPT,
                ),
                io.Int.Input("max_tokens", default=900, min=256, max=3072),
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
                io.Boolean.Input("hold_prompt", display_name="Hold Prompt", default=False),
                io.Int.Input(
                    "continuous_scene_count", display_name="Scenes — Enhance / Continuous Edit",
                    default=3, min=1, max=12, step=1,
                ),
                io.String.Input(
                    "direction_context", optional=True, force_input=True,
                    tooltip="Compatible with H3 Compact Director — Direction Controls.",
                ),
                io.Custom("LLMMODEL").Input("llm", optional=True),
                io.Boolean.Input(
                    "bypass", default=False,
                    tooltip="Pass the video request through unchanged without calling the LLM.",
                ),
                io.Boolean.Input(
                    "debug_request", default=False,
                    tooltip="Expose the sanitized provider request without image data or secrets.",
                ),
            ],
            outputs=[
                io.String.Output("prompt"),
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
        cls, video_request: str, director_mode: str, first_frame,
        system_prompt: str, max_tokens: int, temperature: float, reasoning: bool,
        seed: int, image_max_dimension: int, timeout_seconds: int,
        hold_prompt: bool = False, continuous_scene_count: int = 3,
        direction_context=None, llm=None, bypass: bool = False,
        debug_request: bool = False, unique_id=None,
    ) -> io.NodeOutput:
        request_text = str(video_request or "").strip()
        scene_count = max(1, min(12, int(continuous_scene_count)))
        if bool(bypass):
            return io.NodeOutput(
                request_text, "BYPASS · user prompt passed through unchanged",
                "Bypass · no LLM called", scene_count,
                "BYPASS: no request sent." if debug_request else "",
                "BYPASS: no model response." if debug_request else "",
                ui=ui.PreviewText(request_text),
            )
        if not request_text:
            raise ValueError("LTX 2.5 Compact I2V Director requires a video_request.")
        if first_frame is None or not hasattr(first_frame, "shape") or int(first_frame.shape[0]) < 1:
            raise ValueError("Connect the opening I2V image to first_frame.")
        if llm is None:
            raise ValueError("Connect an OpenRouter, Ollama or compatible LLMMODEL provider.")

        resolved_mode = {
            "edit": "deep", "Enhance": "sequence", "enhance": "sequence",
            "Continuous Edit": "continuous_sequence",
            "Edit Continuo": "continuous_sequence",
        }.get(str(director_mode or "compact"), str(director_mode or "compact"))
        if resolved_mode not in ("compact", "deep", "sequence", "continuous_sequence"):
            resolved_mode = "compact"
        is_sequence = resolved_mode in ("sequence", "continuous_sequence")
        direction_text = str(direction_context or "").strip()
        if direction_text.lower() == "none":
            direction_text = ""

        cache_key = f"{cls.__name__}:{str(unique_id or 'default')}"
        provider_identity = json.dumps({
            "type": type(llm).__name__,
            "model": str(getattr(llm, "model", "")),
            "endpoint": str(getattr(llm, "server_url", getattr(llm, "base_url", ""))),
            "thinking": getattr(llm, "thinking", None),
        }, sort_keys=True)
        if bool(hold_prompt):
            held = _get_held_director_plan(cache_key)
            if held and held.get("cache_kind") == "ltx25_compact_i2v":
                if held.get("provider_identity") != provider_identity:
                    raise RuntimeError("The provider/model changed. Disable Hold once to regenerate.")
                if held.get("direction_context", "") != direction_text:
                    raise RuntimeError("Direction Controls changed. Disable Hold once to regenerate.")
                if held.get("resolved_mode") != resolved_mode:
                    raise RuntimeError("Director mode changed. Disable Hold once to regenerate.")
                if is_sequence and int(held.get("scene_count", 0)) != scene_count:
                    raise RuntimeError("Scene count changed. Disable Hold once to regenerate.")
                return io.NodeOutput(
                    held["prompt"], held["validation"] + " · HOLD",
                    "Held prompt · provider not called", scene_count,
                    ("HOLD: no request sent.\n" + held.get("debug_request", ""))
                    if debug_request else "",
                    held.get("raw_response", "") if debug_request else "",
                    ui=ui.PreviewText(held.get("preview", held["prompt"])),
                )

        resolved_system = str(system_prompt or cls.DEFAULT_SYSTEM_PROMPT).strip()
        if resolved_mode == "deep":
            resolved_system += "\n\n" + cls.DEEP_RULES
        if is_sequence:
            resolved_system += "\n\n" + cls.SEQUENCE_RULES
            resolved_system += f"\nReturn exactly {scene_count} strings in scene_prompts."
        if direction_text:
            resolved_system += (
                "\n\nSELECTED DIRECTION:\n" + direction_text +
                "\nRealize every non-none selection concretely while preserving the opening "
                "image and explicit user intent. Blend secondary choices as compatible accents."
            )

        schema = ({
            "name": "ltx25_compact_i2v_sequence", "strict": True,
            "schema": {
                "type": "object",
                "properties": {"scene_prompts": {
                    "type": "array", "minItems": scene_count, "maxItems": scene_count,
                    "items": {"type": "string", "minLength": 40},
                }},
                "required": ["scene_prompts"], "additionalProperties": False,
            },
        } if is_sequence else {
            "name": "ltx25_compact_i2v_prompt", "strict": True,
            "schema": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "minLength": 60}},
                "required": ["prompt"], "additionalProperties": False,
            },
        })
        content = [
            {"type": "text", "text": "RAW VIDEO REQUEST:\n" + request_text},
            {"type": "text", "text": (
                "OPENING I2V FRAME. Inspect its pixels. This exact image establishes frame zero; "
                "describe the video beginning from it without referring to it as an image."
            )},
            {"type": "image_url", "image_url": {
                "url": _image_data_url(first_frame[:1], int(image_max_dimension))
            }},
        ]
        payload = {
            "model": str(getattr(llm, "model", "") or ""),
            "messages": [
                {"role": "system", "content": resolved_system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max(int(max_tokens), scene_count * 360) if is_sequence else int(max_tokens),
            "temperature": float(temperature), "seed": int(seed),
            "reasoning": {"enabled": bool(reasoning)},
            "response_format": {"type": "json_schema", "json_schema": schema},
            "provider": {"require_parameters": True},
        }
        debug_text = H3CompactMultimodalEditDirector._debug_request(llm, payload) if debug_request else ""
        result = _external_llm_request(llm, payload)
        try:
            raw = result["choices"][0]["message"]["content"]
            raw_response = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if is_sequence:
                prompts = [str(item).strip() for item in parsed["scene_prompts"]]
                if len(prompts) != scene_count or any(not item for item in prompts):
                    raise ValueError("scene prompt count mismatch")
                prompt = json.dumps({"scene_prompts": prompts}, ensure_ascii=False)
                preview = "\n\n".join(
                    f"SCENE {index}\n{item}" for index, item in enumerate(prompts, 1)
                )
            else:
                prompt = str(parsed["prompt"]).strip()
                if len(prompt) < 60:
                    raise ValueError("prompt too short")
                preview = prompt
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("LTX 2.5 Compact I2V Director returned an invalid response.") from error

        word_counts = [
            len(re.findall(r"\b[\w'-]+\b", item))
            for item in (prompts if is_sequence else [prompt])
        ]
        warnings = []
        if any(count > 200 for count in word_counts):
            warnings.append("one or more prompts exceed 200 words")
        forbidden = re.compile(
            r"\b(?:input image|first frame|reference image|negative prompt|LTX(?:\s*2\.5)?)\b",
            re.IGNORECASE,
        )
        if any(forbidden.search(item) for item in (prompts if is_sequence else [prompt])):
            warnings.append("prompt contains implementation/reference language")
        validation = (
            f"{director_mode} LTX 2.5 I2V ready · "
            f"{scene_count if is_sequence else 1} prompt(s) · opening frame inspected · "
            + ("format valid" if not warnings else "warnings: " + "; ".join(warnings))
        )
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        usage_stats = (
            f"input: {usage.get('prompt_tokens', '?')} · "
            f"output: {usage.get('completion_tokens', '?')} · "
            f"total: {usage.get('total_tokens', '?')} · "
            f"provider: {type(llm).__name__} · model: {getattr(llm, 'model', 'external')}"
        )
        _set_held_director_plan(cache_key, {
            "cache_kind": "ltx25_compact_i2v",
            "provider_identity": provider_identity,
            "direction_context": direction_text,
            "resolved_mode": resolved_mode,
            "scene_count": scene_count,
            "prompt": prompt,
            "preview": preview,
            "validation": validation,
            "debug_request": debug_text,
            "raw_response": raw_response,
        })
        return io.NodeOutput(
            prompt, validation, usage_stats, scene_count,
            debug_text, raw_response if debug_request else "",
            ui=ui.PreviewText(preview),
        )
