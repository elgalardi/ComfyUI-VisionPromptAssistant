from __future__ import annotations

import base64
import io as binary_io
import json
import math
import urllib.error
import urllib.request

import numpy as np
import torch
import comfy.sd
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui
from PIL import Image, ImageDraw, ImageFont
from typing_extensions import override

from .story_director import (
    H3LLMModelAPI,
    H3OllamaModel,
    H3StoryDirector,
    H3StoryDirectorLLMAPI,
)
from .audio_transcriber import LocalWhisperTranscribe


VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

CLIP_TYPES = [
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "flux2",
    "ovis",
    "longcat_image",
    "cogvideox",
    "lens",
    "pixeldit",
    "ideogram4",
    "boogu",
    "krea2",
    "joyimage",
    "mage",
    "minimax",
]
TEXT_ENCODERS = folder_paths.get_filename_list("text_encoders")
PREFERRED_ENCODER = "qwen3-vl-4b-heretic_int8.safetensors"
DEFAULT_ENCODER = (
    PREFERRED_ENCODER
    if PREFERRED_ENCODER in TEXT_ENCODERS
    else (TEXT_ENCODERS[0] if TEXT_ENCODERS else "")
)
_CLIP_CACHE = {"key": None, "clip": None}
ABLITERATION_API_URL = "https://api.abliteration.ai/v1/chat/completions"
ABLITERATION_MODEL = "abliterated-model"


def _qwen_chat_prompt(
    system_prompt: str,
    user_prompt: str,
    pictures,
    max_length: int,
):
    """Build a Qwen3-VL chat prompt aligned with MiniMax H3 picture tags."""
    system_prompt = (system_prompt or "").strip()
    user_prompt = (user_prompt or "").strip()

    parts = []
    vision_inputs = []
    reserve = max(16, math.ceil(max_length * 0.12))
    target_tokens = max(1, max_length - reserve)
    if system_prompt:
        parts.append(
            f"<|im_start|>system\n{system_prompt}\n\n"
            f"HARD OUTPUT BUDGET: Finish the complete answer within approximately "
            f"{target_tokens} tokens. The generation limit is {max_length} tokens. "
            "Prioritize the essential picture assignments, subject, motion, and camera; "
            "be concise, do not start details you cannot finish, and end with a complete "
            "sentence before the limit.<|im_end|>\n"
        )

    parts.append("<|im_start|>user\n")
    parts.append(
        "Reference map for MiniMax H3. In your answer, use the exact angle-bracket "
        "tags shown below; input socket names themselves must not appear in the final prompt.\n"
    )

    picture_number = 0
    for socket_index, picture in pictures:
        picture_number += 1
        parts.append(
            f"image_{socket_index} maps to <Picture {picture_number}>: {VISION_BLOCK}\n"
        )
        vision_inputs.append(picture[:1])

    parts.append("\nUser request:\n")
    parts.append(f"{user_prompt}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts), vision_inputs


def _image_to_pil(image, max_dimension: int) -> Image.Image:
    """Convert the first IMAGE batch item to a bounded RGB PIL image."""
    pixels = image[0].detach().cpu().clamp(0.0, 1.0).numpy()
    pixels = (pixels * 255.0).round().astype(np.uint8)
    pil_image = Image.fromarray(pixels)
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    elif pil_image.mode == "L":
        pil_image = pil_image.convert("RGB")

    longest = max(pil_image.size)
    if longest > max_dimension:
        scale = max_dimension / longest
        size = (
            max(1, round(pil_image.width * scale)),
            max(1, round(pil_image.height * scale)),
        )
        pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
    return pil_image


def _pil_to_data_url(pil_image: Image.Image) -> str:
    """Encode a PIL image as a JPEG data URL."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    buffer = binary_io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _image_to_data_url(image, max_dimension: int) -> str:
    return _pil_to_data_url(_image_to_pil(image, max_dimension))


def _contact_sheet_data_url(images, max_dimension: int) -> str:
    """Combine 2-3 references into one labeled image for single-image APIs."""
    converted = [_image_to_pil(image, max_dimension) for image in images]
    if len(converted) == 1:
        return _pil_to_data_url(converted[0])

    columns = 2
    rows = math.ceil(len(converted) / columns)
    label_height = 44
    gap = 12
    cell_size = max(256, min(int(max_dimension), 1024))
    sheet_width = columns * cell_size + (columns + 1) * gap
    sheet_height = rows * (cell_size + label_height) + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_width, sheet_height), (24, 24, 24))

    draw = ImageDraw.Draw(sheet)
    try:
        label_font = ImageFont.load_default(size=24)
    except TypeError:
        label_font = ImageFont.load_default()
    for index, image in enumerate(converted):
        row, column = divmod(index, columns)
        x = gap + column * (cell_size + gap)
        y = gap + row * (cell_size + label_height + gap)
        label = f"<Picture {index + 1}>"
        draw.text(
            (x + 8, y + 8), label, fill=(255, 255, 255), font=label_font
        )

        available_height = cell_size
        scale = min(cell_size / image.width, available_height / image.height)
        size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        resized = image.resize(size, Image.Resampling.LANCZOS)
        paste_x = x + (cell_size - resized.width) // 2
        paste_y = y + label_height + (available_height - resized.height) // 2
        sheet.paste(resized, (paste_x, paste_y))

    return _pil_to_data_url(sheet)


def _response_text(message_content) -> str:
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(message_content or "").strip()


class LocalVisionPromptGenerator(io.ComfyNode):
    """Generate text locally from a system prompt, user prompt, and 1-3 images."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LocalVisionPromptGenerator",
            display_name="Vision Prompt Assistant",
            category="text",
            search_aliases=[
                "local vision prompt",
                "qwen vision prompt",
                "multi image generate text",
                "clip prompt assistant",
                "prompt assistant clip",
            ],
            description=(
                "Generates text locally with a compatible multimodal CLIP such "
                "as Qwen3-VL. Supports separate system/user prompts and up to "
                "three reference images. For fuller results, end the user prompt "
                "with the desired approximate token count, keeping it slightly "
                "below max_length (for example: 'Write about 220 tokens' with "
                "max_length set to 256)."
            ),
            inputs=[
                io.Combo.Input(
                    "clip_name",
                    options=TEXT_ENCODERS,
                    default=DEFAULT_ENCODER,
                ),
                io.Combo.Input("clip_type", options=CLIP_TYPES, default="ltxv"),
                io.Combo.Input(
                    "load_device",
                    options=["default", "cpu"],
                    default="default",
                    advanced=True,
                ),
                io.String.Input(
                    "user_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="Analyze the reference images and write a detailed generation prompt.",
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=(
                        "You write production-ready prompts for MiniMax H3 Reference to Video. "
                        "Use the exact supplied <Picture n> tags, clearly assigning identity, "
                        "appearance, style, motion, and camera. Return only the final "
                        "generation prompt."
                    ),
                ),
                io.Image.Input("image_0", optional=True),
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
                io.Int.Input(
                    "max_length",
                    default=256,
                    min=1,
                    max=4096,
                    tooltip=(
                        "Hard generation limit. For a fuller prompt, also request an "
                        "approximate token count near the end of user_prompt, slightly "
                        "below this value."
                    ),
                ),
                io.Boolean.Input("sampling", default=True),
                io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.01),
                io.Int.Input("top_k", default=40, min=0, max=1000),
                io.Float.Input("top_p", default=0.90, min=0.0, max=1.0, step=0.01),
                io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                io.Float.Input(
                    "repetition_penalty",
                    default=1.05,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[io.String.Output("generated_text")],
        )

    @classmethod
    def execute(
        cls,
        clip_name: str,
        clip_type: str,
        load_device: str,
        system_prompt: str,
        user_prompt: str,
        max_length: int,
        sampling: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        seed: int,
        image_0=None,
        image_1=None,
        image_2=None,
    ) -> io.NodeOutput:
        clip = cls._load_clip(clip_name, clip_type, load_device)
        pictures = [
            (index, image)
            for index, image in enumerate((image_0, image_1, image_2))
            if image is not None
        ]
        prompt, images = _qwen_chat_prompt(
            system_prompt,
            user_prompt,
            pictures,
            int(max_length),
        )

        tokens = clip.tokenize(
            prompt,
            images=images,
            skip_template=True,
            min_length=1,
        )
        generated_ids = clip.generate(
            tokens,
            do_sample=bool(sampling),
            max_length=int(max_length),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            min_p=float(min_p),
            repetition_penalty=float(repetition_penalty),
            presence_penalty=0.0,
            seed=int(seed),
        )
        return io.NodeOutput(clip.decode(generated_ids))

    @classmethod
    def _load_clip(cls, clip_name: str, clip_type: str, load_device: str):
        cache_key = (clip_name, clip_type, load_device)
        if _CLIP_CACHE["key"] == cache_key and _CLIP_CACHE["clip"] is not None:
            return _CLIP_CACHE["clip"]

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        model_options = {}
        if load_device == "cpu":
            cpu = torch.device("cpu")
            model_options["load_device"] = cpu
            model_options["offload_device"] = cpu

        resolved_type = getattr(
            comfy.sd.CLIPType,
            clip_type.upper(),
            comfy.sd.CLIPType.STABLE_DIFFUSION,
        )
        loaded_clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=resolved_type,
            model_options=model_options,
        )
        _CLIP_CACHE["key"] = cache_key
        _CLIP_CACHE["clip"] = loaded_clip
        return loaded_clip


class AbliterationVisionPrompt(io.ComfyNode):
    """Generate a vision-aware prompt through Abliteration.ai."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AbliterationVisionPrompt",
            display_name="Abliteration Vision Prompt",
            category="text",
            search_aliases=[
                "abliteration ai",
                "uncensored vision prompt",
                "api vision prompt",
            ],
            description=(
                "Generates a prompt through Abliteration.ai's OpenAI-compatible "
                "vision API. Supports separate system/user prompts and up to "
                "three images. Images are resized and compressed before upload."
            ),
            inputs=[
                io.String.Input(
                    "api_key",
                    default="",
                    placeholder="ak_...",
                    tooltip=(
                        "Abliteration.ai API key. The interface masks this value, "
                        "but a saved workflow may still contain it."
                    ),
                    extra_dict={"password": True},
                ),
                io.String.Input(
                    "user_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default=(
                        "Analyze the reference images and write a detailed "
                        "MiniMax H3 generation prompt."
                    ),
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=(
                        "You write production-ready prompts for MiniMax H3 "
                        "Reference to Video. Use the exact supplied <Picture n> "
                        "tags, clearly assigning identity, appearance, style, "
                        "motion, and camera. Return only the final generation prompt."
                    ),
                ),
                io.Image.Input("image_0", optional=True),
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
                io.Int.Input("max_tokens", default=256, min=1, max=4096),
                io.Float.Input(
                    "temperature", default=0.4, min=0.0, max=2.0, step=0.05
                ),
                io.Boolean.Input(
                    "thinking",
                    default=False,
                    tooltip="Disable for faster prompt enhancement.",
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
                    tooltip=(
                        "Images are resized to this maximum width or height before "
                        "upload. Lower values are faster and cost fewer tokens."
                    ),
                ),
                io.Int.Input(
                    "timeout_seconds",
                    default=180,
                    min=15,
                    max=600,
                    advanced=True,
                ),
            ],
            outputs=[
                io.String.Output("generated_text"),
                io.String.Output("usage_stats"),
                io.String.Output("credits_remaining"),
            ],
        )

    @classmethod
    def execute(
        cls,
        api_key: str,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        thinking: bool,
        seed: int,
        image_max_dimension: int,
        timeout_seconds: int,
        image_0=None,
        image_1=None,
        image_2=None,
    ) -> io.NodeOutput:
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("Abliteration.ai API key is required.")

        content = []
        pictures = [
            image
            for image in (image_0, image_1, image_2)
            if image is not None
        ]
        if pictures:
            labels = ", ".join(
                f"<Picture {number}>" for number in range(1, len(pictures) + 1)
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Reference images are mapped in connection order as {labels}. "
                        "The following visual is a labeled contact sheet; treat every "
                        "labeled panel as a separate reference image and use its exact "
                        "<Picture n> tag in the final prompt. Do not ignore any panel."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _contact_sheet_data_url(
                            pictures, int(image_max_dimension)
                        )
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": f"User request:\n{(user_prompt or '').strip()}",
            }
        )

        payload = {
            "model": ABLITERATION_MODEL,
            "messages": [
                {"role": "system", "content": (system_prompt or "").strip()},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "thinking": bool(thinking),
            "seed": int(seed),
        }
        request = urllib.request.Request(
            ABLITERATION_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=int(timeout_seconds)
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(details)
                details = parsed.get("error", {}).get("message", details)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise RuntimeError(
                f"Abliteration.ai returned HTTP {error.code}: {details}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not connect to Abliteration.ai: {error.reason}"
            ) from error

        try:
            generated_text = _response_text(
                result["choices"][0]["message"]["content"]
            )
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "Abliteration.ai returned an unexpected response."
            ) from error
        if not generated_text:
            raise RuntimeError("Abliteration.ai returned an empty response.")

        usage = result.get("usage") or {}
        usage_stats = (
            f"input: {usage.get('prompt_tokens', '?')} · "
            f"output: {usage.get('completion_tokens', '?')} · "
            f"total: {usage.get('total_tokens', '?')}"
        )
        remaining = result.get("remaining_credits")
        used = result.get("estimated_credits_used")
        estimated_cost = result.get("estimated_cost_usd")
        if remaining is None:
            credits_remaining = "Remaining credits: not reported"
        else:
            credits_remaining = f"Remaining credits: {remaining:,}"
        if used is not None:
            credits_remaining += f" · used: {used:,}"
        if isinstance(estimated_cost, (int, float)):
            credits_remaining += f" · estimated cost: ${estimated_cost:.6f}"
        return io.NodeOutput(generated_text, usage_stats, credits_remaining)


class PreviewVisionPrompt(io.ComfyNode):
    """Display generated prompt text and pass it through unchanged."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PreviewVisionPrompt",
            display_name="Preview Vision Prompt",
            category="text",
            search_aliases=["preview prompt", "show vision prompt", "show text"],
            description=(
                "Displays a generated vision prompt inside the node without "
                "saving it, and passes the text through unchanged."
            ),
            inputs=[io.String.Input("prompt", force_input=True)],
            outputs=[io.String.Output("prompt")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, prompt: str) -> io.NodeOutput:
        return io.NodeOutput(prompt, ui=ui.PreviewText(prompt))


class H3EditDuration24FPS(io.ComfyNode):
    """Provide one frame-exact 24 fps duration source for Director and VHS."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3EditDuration24FPS",
            display_name="H3 Edit Duration — 24 FPS",
            category="conditioning/minimax",
            search_aliases=[
                "h3 frame load cap", "24 fps duration", "vhs frame cap",
                "minimax edit duration",
            ],
            description=(
                "Keeps H3 Story Director and VHS Load Video on one exact duration. "
                "Connect duration_seconds to the Director and frame_load_cap to VHS."
            ),
            inputs=[
                io.Float.Input(
                    "duration_seconds",
                    default=5.0,
                    min=0.5,
                    max=15.0,
                    step=0.5,
                    tooltip="Requested source-video duration at exactly 24 fps.",
                ),
            ],
            outputs=[
                io.Float.Output(
                    "duration_seconds",
                    tooltip="Connect to H3 Story Director scene_duration_seconds.",
                ),
                io.Int.Output(
                    "frame_load_cap",
                    tooltip="Connect directly to VHS Load Video frame_load_cap.",
                ),
            ],
        )

    @classmethod
    def execute(cls, duration_seconds: float) -> io.NodeOutput:
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_seconds must be a finite positive number.")
        exact_frames = duration * 24.0
        frame_load_cap = int(round(exact_frames))
        if not math.isclose(exact_frames, frame_load_cap, abs_tol=1e-9):
            raise ValueError(
                "duration_seconds must resolve to a whole frame at 24 fps; "
                f"{duration:g}s equals {exact_frames:g} frames."
            )
        return io.NodeOutput(duration, frame_load_cap)


class LocalVisionPromptExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LocalVisionPromptGenerator,
            AbliterationVisionPrompt,
            H3LLMModelAPI,
            H3OllamaModel,
            H3StoryDirector,
            H3StoryDirectorLLMAPI,
            LocalWhisperTranscribe,
            PreviewVisionPrompt,
            H3EditDuration24FPS,
        ]


async def comfy_entrypoint() -> LocalVisionPromptExtension:
    return LocalVisionPromptExtension()


WEB_DIRECTORY = "./web"


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
