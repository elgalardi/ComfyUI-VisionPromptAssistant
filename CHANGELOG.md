# Changelog

## Unreleased

- H3 Open Director: the four reference picture sockets became one growing list
  like MiniMax H3 Reference to Video: connecting a picture adds the next slot,
  up to 9. The N-th connected picture is <Picture N> (empty slots skipped).
  The pictures are the first inputs of the node, then Seconds per Scene (now
  optional, 10 s when unconnected). Workflows saved with the old four sockets
  are moved automatically: their picture connections go into the list, in
  order, and the old sockets disappear. Videos and guides are unchanged.
- H3 Open Director: each Direction Control reaches the LLM as one plain
  sentence ("It is set in the 1980s.", "Clothes: as in each character's
  reference picture."), with no rules attached. Random picks are drawn from the
  director's seed and listed in validation. "No dialogue" is stated firmly
  (no words, no <d> or speaker tags), drops the density line, and validation
  warns if dialogue still appears.
- Video Task moved from Direction Controls to H3 Transfer & Edit Director (last
  input, default Swap). A workflow that had Edit, Add or Remove on the controls
  must set it on the director now.
- H3 Compact Multimodal Edit Director is now H3 Transfer & Edit Director (v149):
  only Transfer, Transfer Keep Audio, Transfer Keep Audio Basic and Edit remain,
  default Transfer. Normal and Compact and their creative code were removed;
  creative scenes belong to H3 Open Director. Same node id and inputs, so saved
  workflows load; one saved in Normal or Compact must switch mode. Transfer and
  Edit send exactly the same requests and return the same prompts as v147.
- The creative engine behind those modes (Enhance, Elaborate, Elaborate Plus,
  Creative Plus planning, the structured creative schema and their rule
  blocks) is removed from story_director.py too: the two director files are
  about 2,300 lines shorter than in v147. Offline tests for the removed modes
  were dropped.
- H3 Open Director: a reply cut off by the token limit is never passed on as a
  prompt; it is asked again with more room. Missing scenes are asked for
  (continuing the written ones) instead of repeating the last scene.
  Temporary provider errors retry the same request; a refused JSON format
  retries as plain text; total calls are capped at 4.
- H3 Open Director: new `on_failure` option (default Stop) stops the workflow
  with a clear message when the scenes cannot be written; "Use request" keeps
  the old pass-through.
- H3 Open Director: Hold is kept per workflow, node, system prompt and scene
  setup, and only complete results are saved.
- H3 Open Director: scenes still continue from the previous scene's last moment,
  and after that may cut to another place or time when asked. usage_stats shows
  the max_tokens actually requested.
- H3 Compact Multimodal Edit Director (v146): posture survives cuts. Every shot
  after a cut restates where each subject is and how (seated, standing,
  kneeling, lying); a post-processor adds ", still seated on the sofa," when
  a later shot forgets it, and keeps changes the shot itself shows.
- Cuts written in MiniMax's official wording ("the camera cuts to", "the shot
  transitions to") are kept as timed cuts instead of being dropped or doubled;
  "cut back to" keeps "back".
- The shot budget is enforced in code: extra cuts become camera motion inside
  the same shot. Every shot lasts at least 2 s, including the first and last.
- Cuts without a usable time (including speaker-change cuts) are placed by how
  much text and dialogue comes before them instead of the midpoint.
- Removed the "Fix" wording from the director's descriptions and console logs.
- H3 Compact Multimodal Edit Director (v147), Normal and Compact: prompts follow
  the layout MiniMax documents for each input mode. Text to video: only
  integrated_multimodal_description / overall_soundscape / non_diegetic_music,
  style at the start of [Shot 1], people described in words. Image to video:
  the alignment line plus those three fields. Reference pictures: the six
  Ref2VA sections with the task in summary ([reference generation], plus
  keyframe completion when Picture 1 is also the first frame). Cuts use
  "the camera cuts to". Transfer and Edit keep their layouts.
- Dialogue that crosses a cut is written in two parts with <scenetrans> at
  both connecting points, as the official guide asks.

## 2.0.0 - 2026-09-15

- Breaking cleanup: removed legacy image/video/storyboard/LTX directors,
  transcription and unused utility nodes. Old workflows using those node IDs
  require an earlier release.
- Retained the compact H3 director, direction controls, prompt selector,
  Music 3 director and OpenRouter/Ollama/API/local Qwen LLM loaders.
- Added text-only requests without requiring a visual reference.
- Director seed can share a PrimitiveInt input with the generation sampler.
- Removed the unused faster-whisper dependency.
- Stored held plans under ComfyUI output/Sexy AI Studio/director_hold.

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
