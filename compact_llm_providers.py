"""Multimodal providers sharing the director's chat-completion contract."""
import base64
import importlib
import io as binary_io
import json
import os

import folder_paths
import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io


class OpenRouterConnection:
    def __init__(self, api_key, model, timeout_seconds):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://openrouter.ai/api/v1"

    def h3_chat_completion(self, payload):
        from .story_director import _openrouter_request
        key = str(self.api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not key:
            raise ValueError("Set the API key in H3 OpenRouter Model or OPENROUTER_API_KEY.")
        request = dict(payload)
        request["model"] = self.model
        return _openrouter_request(key, request, int(self.timeout_seconds))


class H3OpenRouterModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from .story_director import DEFAULT_MODEL
        return io.Schema(
            node_id="H3OpenRouterModel",
            display_name="H3 OpenRouter Model",
            category="text/minimax_h3",
            inputs=[
                io.String.Input("api_key", default="", extra_dict={"password": True}),
                io.String.Input("model", default=DEFAULT_MODEL),
                io.Int.Input("timeout_seconds", default=300, min=30, max=1800),
            ],
            outputs=[io.Custom("LLMMODEL").Output("llm_model"),
                     io.String.Output("credits_remaining")],
        )

    @classmethod
    def execute(cls, api_key, model, timeout_seconds):
        from .story_director import _credits
        if not str(model).strip():
            raise ValueError("Select an OpenRouter model.")
        key = str(api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        credits = _credits(key, min(int(timeout_seconds), 15)) if key else "Credits: API key missing"
        return io.NodeOutput(OpenRouterConnection(api_key, model.strip(), timeout_seconds), credits)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Balance is external state; refresh before each queued execution.
        return float("nan")


def qwen_messages(messages, schema):
    """Keep source labels and image ordering; never remap video frames to pictures."""
    parts, images = [], []
    for message in messages:
        role = message.get("role", "user")
        parts.append(f"<|im_start|>{role}\n")
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            for item in content:
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")) + "\n")
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if not url.startswith("data:image/"):
                        raise ValueError("Local Qwen requires embedded image references.")
                    with Image.open(binary_io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
                        pixels = np.array(image.convert("RGB"), copy=True)
                    images.append(torch.from_numpy(pixels).float().unsqueeze(0) / 255.0)
                    parts.append("<|vision_start|><|image_pad|><|vision_end|>\n")
        if role == "system":
            parts.append(
                "\nReturn only valid JSON matching this schema, without markdown or commentary:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        parts.append("<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts), images


class QwenLocalConnection:
    def __init__(self, clip_name, clip_type, load_device):
        self.model = clip_name
        self.clip_type = clip_type
        self.load_device = load_device

    def h3_chat_completion(self, payload):
        # Use the shared local loader and ComfyUI's memory manager.
        package = importlib.import_module(__package__)
        clip = package.LocalVisionPromptGenerator._load_clip(
            self.model, self.clip_type, self.load_device
        )
        schema = payload["response_format"]["json_schema"]["schema"]
        prompt, images = qwen_messages(payload["messages"], schema)
        tokens = clip.tokenize(prompt, images=images, skip_template=True, min_length=1)
        temperature = float(payload.get("temperature", 0.2))
        generated = clip.generate(
            tokens, do_sample=temperature > 0, max_length=int(payload["max_tokens"]),
            temperature=max(0.01, temperature), top_k=40, top_p=0.9,
            min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0,
            seed=int(payload.get("seed", 0)),
        )
        text = clip.decode(generated).strip()
        # Tolerate fenced JSON, but never rewrite its semantics or invent fields.
        if text.startswith("\x60\x60\x60") and text.endswith("\x60\x60\x60"):
            text = text.split("\n", 1)[-1].rsplit("\x60\x60\x60", 1)[0].strip()
        try:
            json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Local Qwen returned invalid/truncated JSON. Increase max_tokens "
                "or reduce the scene count; no paid fallback was called."
            ) from error
        return {"choices": [{"message": {"content": text}}], "usage": {}}


class H3QwenLocalModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names = folder_paths.get_filename_list("text_encoders")
        preferred = "qwen3-vl-4b-heretic_int8.safetensors"
        default = preferred if preferred in names else (names[0] if names else "")
        return io.Schema(
            node_id="H3QwenLocalModel",
            display_name="H3 Qwen3-VL Model (Local ComfyUI)",
            category="text/minimax_h3",
            inputs=[
                io.Combo.Input("clip_name", options=names, default=default),
                io.Combo.Input("clip_type", options=["ltxv", "minimax"], default="ltxv"),
                io.Combo.Input("load_device", options=["default", "cpu"], default="default"),
            ],
            outputs=[io.Custom("LLMMODEL").Output("llm_model")],
        )

    @classmethod
    def execute(cls, clip_name, clip_type, load_device):
        folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return io.NodeOutput(QwenLocalConnection(clip_name, clip_type, load_device))
