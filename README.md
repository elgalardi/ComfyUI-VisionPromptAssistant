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
- Connect up to three images, three reference videos, three paired video
  soundtracks, and three standalone audio references.
- Socket names map to MiniMax tags in connection order: `image_0` becomes
  `<Picture 1>`, `video_0` becomes `<Video 1>`, and `audio_0` becomes
  `<Audio 1>` when they are the first connected reference of each type.
- Videos are represented to the vision model by up to eight frames sampled
  across their duration. Audio contributes duration/channel information and
  the correct H3 tag; the Qwen vision encoder does not listen to the waveform.

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

Both **Vision Prompt Assistant** and **Preview Vision Prompt** are included in
this package.
