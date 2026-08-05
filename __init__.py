from __future__ import annotations

import math
import torch
import comfy.sd
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui
from typing_extensions import override


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


class LocalVisionPromptExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LocalVisionPromptGenerator, PreviewVisionPrompt]


async def comfy_entrypoint() -> LocalVisionPromptExtension:
    return LocalVisionPromptExtension()


WEB_DIRECTORY = "./web"


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
