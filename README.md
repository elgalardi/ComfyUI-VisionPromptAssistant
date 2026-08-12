# Vision Prompt Assistant

A local ComfyUI text-generation node with separate system and user prompts,
designed to write reference-aware prompts for MiniMax H3.

The node is intended for full multimodal text encoders supported by ComfyUI,
such as Qwen3-VL. It does not call OpenRouter or any other external API.

Recommended local setup:

- Download
  [`qwen3-vl-4b-heretic_int8.safetensors`](https://huggingface.co/DreamFast/Qwen3-VL-4b-Heretic-ComfyUI)
  into `ComfyUI/models/text_encoders`, then select it directly in the node.
- Start with loader type `ltxv`, or compare it with `stable_diffusion`.
- Connect up to three reference images. Socket names map to MiniMax picture
  tags in connection order: the first connected image becomes `<Picture 1>`,
  the second becomes `<Picture 2>`, and the third becomes `<Picture 3>`.

Video and audio sockets are intentionally not included. The recommended
Qwen3-VL encoder can inspect images, but it does not reliably understand a
complete video or listen to an audio waveform through ComfyUI's text-generation
interface. Keeping image inputs only makes the node's behavior predictable.

The loaded encoder is cached and reused while the model, type, and device remain
unchanged, so editing prompts does not reread the checkpoint from disk.

`max_length` is the hard maximum for newly generated tokens. The assistant
automatically gives the model a slightly smaller target budget so it has room
to finish its final sentence instead of being cut off at the hard limit.

For fuller results, specify the desired approximate length at the end of the
`user_prompt`, choosing a value slightly below `max_length`. For example, with
`max_length` set to `256`, finish the request with: `Write about 220 tokens.`
This gives the vision model an explicit length target and helps prevent an
otherwise useful final prompt from ending too early.

The `user_prompt` and `system_prompt` boxes accept text directly or a STRING
cable using ComfyUI's normal widget-to-input conversion. `user_prompt` is listed
first so it is the preferred STRING route when the assistant is bypassed; the
system instruction is not intended as passthrough output.

The MiniMax H3 conditioning encoder is truncated and is not suitable for text
generation.

## Preview Vision Prompt

Connect the generated string to **Preview Vision Prompt** to display the full
prompt inside the graph without saving it. The text is also passed through as a
`STRING` output for downstream nodes.

**Vision Prompt Assistant**, **Abliteration Vision Prompt**, and **Preview
Vision Prompt** are included in this package.

## Abliteration Vision Prompt

**Abliteration Vision Prompt** is a separate API-based alternative that sends
the system prompt, user prompt, and up to three images to Abliteration.ai's
OpenAI-compatible vision endpoint. It uses the hosted `abliterated-model`, so it
does not load a local vision encoder or consume VRAM needed by the video model.

Create an API key at [Abliteration.ai](https://abliteration.ai/), paste it into
the masked `api_key` field, and connect zero to three images. Connected images
are mapped in order to `<Picture 1>`, `<Picture 2>`, and `<Picture 3>`. Images
are resized to `image_max_dimension` and encoded as JPEG before upload; `1024`
is the default balance between visual detail, latency, and token cost. When two
or three images are connected, the node combines them into one labeled contact
sheet before upload. This preserves all references on API backends that process
only the first visual block while keeping the MiniMax `<Picture n>` mapping.

Set `thinking` to `false` for faster prompt enhancement. The node returns the
generated prompt as `STRING` plus a second `usage_stats` string with the token
counts reported by the service. A third `credits_remaining` string reports the
credit balance returned after the request, the credits used by that request,
and its estimated USD cost when those values are supplied by the API.

The API-key field is visually masked, but—as with many API nodes—the value may
still be serialized inside a saved workflow. Remove the key before sharing a
workflow JSON or image/video containing embedded workflow metadata, and rotate
the key immediately if it is exposed.

## H3 Story Director

**H3 Story Director** turns a short story idea and up to three character or
subject images into a complete, ordered MiniMax H3 production plan. It uses an
OpenRouter vision model with strict structured output, then validates the
response locally before returning a `plan_json` string.

The default model is `x-ai/grok-4.20`, which supports multiple image inputs and
strict structured outputs on OpenRouter. The model field remains editable.
Reasoning is disabled by default for lower latency and cost and can be enabled
for more complex story structures.

Language is selectable between English, Spanish, and Japanese. Genre is a
15-item menu covering drama, action, thriller, horror, comedy, romance,
science fiction, fantasy, documentary, music video, anime, animated movie,
erotic drama, erotic thriller, and explicit adult film. Adult genres require
all depicted participants to be clearly consenting adults aged 18 or older and
reject sexual treatment of minors or age-ambiguous references.

The model writes the synopsis, continuity bible, shared reference assignments,
and scene prompts. The node—not the model—enforces the requested scene count,
duration, and sampler steps. Every connected image must be assigned through its
exact `<Picture 1>`, `<Picture 2>`, or `<Picture 3>` tag or the plan is rejected.
Incomplete JSON, missing scenes, duplicate IDs, and underspecified scene prompts
also stop before video generation begins.

Connect `plan_json` to the `plan_json_input` socket on **MiniMax H3 Contex Loop
Plan**. After the Director runs, its frontend synchronizes the accepted JSON
back into the Plan node's visual scene cards so the story can be reviewed and
edited before an expensive render. Keep `draft_only` enabled for this first
run: it blocks downstream video generation while still filling the cards. Once
the plan is approved, disconnect the Director cable and queue the workflow;
Chain Plan then uses the synchronized local copy without paying for another API
request. Start with three scenes at five seconds, then increase the length after
validating the workflow.

The OpenRouter API-key field is masked visually but may still be serialized in
the workflow. Remove the key before sharing workflow metadata. For hosted
installations such as Runpod, leave the field empty and provide the key through
the `OPENROUTER_API_KEY` environment variable instead. A value entered in the
node takes priority over the environment variable.
