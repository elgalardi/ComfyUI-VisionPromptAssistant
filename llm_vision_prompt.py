"""Simple multimodal prompt assistant with interchangeable LLM providers."""
import json
from comfy_api.latest import io
from .story_director import _external_llm_request, _image_data_url, _completion_content


class LLMVisionPromptAssistant(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLMVisionPromptAssistant", display_name="Vision Prompt Assistant — LLM",
            category="text",
            inputs=[
                io.Custom("LLMMODEL").Input("llm"),
                io.String.Input("user_prompt", multiline=True, dynamic_prompts=True, default="Create a cinematic scene from the reference images."),
                io.String.Input("system_prompt", multiline=True, default="Write a creative, detailed MiniMax H3 generation prompt in English. Follow the user request. Use the supplied <Picture n> tags for reference identities. Describe concrete action, camera, setting, lighting and sound. Keep identities distinct."),
                io.Image.Input("image_0", optional=True),
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
                io.Int.Input("max_tokens", default=1024, min=128, max=8192),
                io.Float.Input("temperature", default=0.7, min=0.0, max=1.0, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF),
                io.Int.Input("image_max_dimension", default=1024, min=256, max=2048, step=64),
            ], outputs=[io.String.Output("generated_text")])

    @classmethod
    def execute(cls, llm, user_prompt, system_prompt, max_tokens=1024,
                temperature=0.7, seed=0, image_max_dimension=1024,
                image_0=None, image_1=None, image_2=None):
        content = [{"type": "text", "text": str(user_prompt)}]
        number = 0
        for image in (image_0, image_1, image_2):
            if image is not None and image.shape[0] > 0:
                number += 1
                content.extend([{"type": "text", "text": f"<Picture {number}> reference:"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image[:1], int(image_max_dimension))}}])
        schema = {"type": "object", "properties": {"prompt": {"type": "string"}},
                  "required": ["prompt"], "additionalProperties": False}
        payload = {"model": str(getattr(llm, "model", "") or ""),
            "messages": [{"role": "system", "content": str(system_prompt) +
                '\nReturn only JSON with one string field "prompt" containing the complete final generation prompt. No reasoning or commentary.'},
                {"role": "user", "content": content}],
            "max_tokens": int(max_tokens), "temperature": float(temperature), "seed": int(seed),
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "vision_prompt", "strict": True, "schema": schema}}}
        result = _external_llm_request(llm, payload)
        raw = _completion_content(result, payload, llm, "Vision Prompt Assistant")
        from .compact_llm_providers import _local_qwen_json
        parsed = json.loads(_local_qwen_json(raw, schema))
        prompt = parsed.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The LLM did not return a non-empty prompt string.")
        return io.NodeOutput(prompt.strip())
