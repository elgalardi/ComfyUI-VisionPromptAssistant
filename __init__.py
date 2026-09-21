"""Compact H3 and Music 3 directors with interchangeable LLM providers."""
from comfy_api.latest import ComfyExtension, io, ui

from .story_director import (
    H3LLMModelAPI, H3OllamaModel, H3CompactDirectionControls,
    H3CompactMultimodalEditDirector,
)
from .compact_llm_providers import H3OpenRouterModel, H3QwenLocalModel
from .compact_prompt_select import H3CompactPromptSelect
from .music_director import MiniMaxMusic3CompactDirector


class PreviewVisionPrompt(io.ComfyNode):
    """Display generated prompt text and pass it through unchanged."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PreviewVisionPrompt",
            display_name="Preview Vision Prompt",
            category="text",
            search_aliases=["preview prompt", "show vision prompt", "show text"],
            description="Displays a generated prompt without saving it and passes the text through unchanged.",
            inputs=[io.String.Input("prompt", force_input=True)],
            outputs=[io.String.Output("prompt")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, prompt: str) -> io.NodeOutput:
        return io.NodeOutput(prompt, ui=ui.PreviewText(prompt))


class LocalVisionPromptExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PreviewVisionPrompt,
            H3CompactMultimodalEditDirector, H3CompactDirectionControls,
            H3CompactPromptSelect, MiniMaxMusic3CompactDirector,
            H3OpenRouterModel, H3OllamaModel, H3LLMModelAPI, H3QwenLocalModel,
        ]


async def comfy_entrypoint() -> LocalVisionPromptExtension:
    return LocalVisionPromptExtension()


WEB_DIRECTORY = "./web"
__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
