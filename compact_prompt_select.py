"""Adapt compact sequence output to a single image/video conditioning prompt."""
import json
from comfy_api.latest import io


def select_prompt(text, index=0):
    text = str(text or '').strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if not isinstance(value, dict) or 'scene_prompts' not in value:
        return text
    prompts = value['scene_prompts']
    if not isinstance(prompts, list) or not 0 <= index < len(prompts):
        raise ValueError('The director response has no prompt at the requested scene index.')
    prompt = prompts[index]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('The director returned an empty or invalid scene prompt.')
    return prompt.strip()


class H3CompactPromptSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3CompactPromptSelect',
            display_name='H3 Compact Director — Select Scene Prompt',
            category='text/minimax_h3',
            inputs=[io.String.Input('prompt', multiline=True),
                    io.Int.Input('scene_index', default=0, min=0, max=11)],
            outputs=[io.String.Output('prompt')],
        )

    @classmethod
    def execute(cls, prompt, scene_index):
        return io.NodeOutput(select_prompt(prompt, scene_index))
