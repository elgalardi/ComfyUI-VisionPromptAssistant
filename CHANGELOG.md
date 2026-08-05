# Changelog

## 1.1.0 - 2026-08-05

- Simplified Vision Prompt Assistant to three image reference inputs.
- Removed experimental video, paired-video-audio, and standalone audio inputs
  because the local vision encoder could not interpret them reliably.
- Updated the default system prompt and MiniMax reference mapping for pictures.

## 1.0.1 - 2026-08-05

- Added guidance to request an approximate token count at the end of the user
  prompt, slightly below `max_length`, for fuller generated prompts.

## 1.0.0 - 2026-08-05

- Added Vision Prompt Assistant with local multimodal text generation.
- Added separate user and system prompt boxes with direct STRING connections.
- Added MiniMax H3 reference tags for pictures, videos, paired soundtracks, and audio.
- Added automatic generation-token budgeting to reduce truncated prompts.
- Added Preview Vision Prompt with a resizable text display and STRING passthrough.
