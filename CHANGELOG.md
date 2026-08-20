# Changelog

## 1.5.0 - 2026-08-20

- Added separate `OpenRouter` and `Gemma` Director Profiles without changing
  the public H3 plan format or downstream node compatibility.
- Added Gemma's structured scene worksheet with duration-aware action beats,
  physical performance, camera, environment, sound, dialogue, final state, and
  private coverage validation.
- Added optional Ollama thinking and updated new Ollama nodes for the tested
  `huihui_ai/gemma-4-abliterated:12b` model with a 32768-token context default.
- Standardized generated speaker attribution as `(S1)` through `(S4)` while
  retaining the existing stable subject-label system.
- Added deterministic normalization for bare or square-bracket speaker labels.
- Strengthened explicit-action coverage and final-state description for Gemma
  while leaving the established OpenRouter schema compact.

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
