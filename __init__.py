"""Compact H3 and Music 3 directors with interchangeable LLM providers."""
from comfy_api.latest import ComfyExtension, io

from .story_director import (
    H3LLMModelAPI, H3OllamaModel, H3CompactDirectionControls,
    H3CompactMultimodalEditDirector,
)
from .compact_llm_providers import H3OpenRouterModel, H3QwenLocalModel
from .compact_prompt_select import H3CompactPromptSelect
from .music_director import MiniMaxMusic3CompactDirector


class LocalVisionPromptExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            H3CompactMultimodalEditDirector, H3CompactDirectionControls,
            H3CompactPromptSelect, MiniMaxMusic3CompactDirector,
            H3OpenRouterModel, H3OllamaModel, H3LLMModelAPI, H3QwenLocalModel,
        ]


async def comfy_entrypoint() -> LocalVisionPromptExtension:
    return LocalVisionPromptExtension()


WEB_DIRECTORY = "./web"
__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
