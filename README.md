# Vision Prompt Assistant 2.0

Compact MiniMax H3 and Music 3 prompt directors with interchangeable LLM loaders.

## Included nodes

- `LocalVisionPromptGenerator` (Vision Prompt Assistant): local Qwen vision
  prompting with up to three images, system/user prompts and sampling controls.
  The original node ID and sockets are preserved for existing workflows.

- `PreviewVisionPrompt`: display prompt text without saving a file.

- `H3CompactMultimodalEditDirector`: compact/edit/elaborate and continuous scene
  prompts, with optional references or text-only requests.
- `H3CompactDirectionControls`: genre, motion, visual look and dialogue controls.
- `H3CompactPromptSelect`: select one prompt from structured scene output.
- `MiniMaxMusic3CompactDirector`: music brief, supplied or generated lyrics,
  and instrumental mode. Retained for later music testing.
- `H3OpenRouterModel`: OpenRouter provider; key input or `OPENROUTER_API_KEY`.
- `H3OllamaModel`: local Ollama provider.
- `H3LLMModelAPI`: compatible external chat-completion API provider.
- `H3QwenLocalModel`: local Qwen LLM loader.

Connect a provider's `LLMMODEL` output to the director. Provider credentials
belong in the loader or environment, never in a shared workflow. External
providers can incur charges. Local providers require their own installed models.

For reproducible settings, connect one `PrimitiveInt` to the director seed and
the sampler seed (range 0–4294967295). Reusing a seed does not guarantee an
external LLM will return identical text: save and reuse the generated prompt
when exact prompt recall is needed. Hold data is stored in
`output/Sexy AI Studio/director_hold/plans.json`.

## Cinematic craft

The video and visual directors include compact cinematic-craft instructions: causal progression,
persistent physical state, motivated framing and concrete staging. Existing mode, source-role,
scene/panel-count and JSON contracts take priority. Video sequences are still planned in one call;
image prompts use static viewpoints, and local edits preserve untargeted content. These are prompt
instructions, not a guarantee of generated continuity or an analysis of audio from sampled frames.

Elaborate and Continuous Elaborate now compare three concise creative treatments internally and
commit to one coherent approach; Enhance compares three within a smaller detail budget. This adds
no API stages or output fields. Explicit requests, reference roles, locked controls and edit scope
still take priority. Compact, Edit and faithful Storyboard adaptation do not receive this extra
exploration layer. These instructions do not guarantee that every LLM explores distinct ideas.

Enhance is compact Elaborate: the same creative exploration with concentrated wording, not
an expanded Compact paraphrase. It designs complementary staging, viewpoint, performance and
light choices, with distinct visual purposes across scenes while preserving the user's action
and locked controls. It targets 90–140 English
words per scene for 1–5 seconds, 120–200 for over 5–10, and 160–240 for longer scenes. These are
guidelines, not truncation limits; required dialogue is preserved. It retains the existing
H3 section grammar inside each scene: `subject_definitions:`,
`integrated_multimodal_description:`, `overall_soundscape:` and `non_diegetic_music:`.
Subject IDs bind canonical source tags in short definitions, the integrated description keeps
the creative action and camera design, and sound/music use concise lines. Roughly half-length
Elaborate prose is a target, not automatic truncation; missing sections produce a validation warning.
It retains the existing
`scene_prompts` contract and Scenes control: select one scene for a single prompt. The automatic
output-token allowance is 350 / 500 / 650 per scene respectively, or the user's `max_tokens` if
higher. All scenes use one normal LLM request; the existing malformed-response repair may still
make one additional call. Actual charges depend on provider pricing and token usage; a lower
allowance is not a guaranteed discount. Existing temperature and seed settings remain unchanged.
Disable Hold to request fresh prompts after updating, and restart ComfyUI to load the new code.

### Shots per scene

Enhance, Elaborate and Continuous Elaborate expose `Shots — Per Scene` (1–3).
1 preserves automatic shot design; it does not force a single take. 2 or 3 requests
exactly that many shots inside each scene's existing duration, not additional scenes.
The LLM plans structured shots in one request; the node renders the same H3 text
sections and `[Shot N]` labels with scene-local cut times. Invalid timestamps are
redistributed locally without dropping shots. A provider returning the wrong shot
count produces a validation warning rather than discarding generated descriptions.

Multi-shot selection authorizes internal cuts over general continuous-take guidance;
other reference, action and direction locks remain active. The next scene's first
shot inherits the previous scene's final-shot action, framing and audiovisual state
before its first internal cut. This is logical planning, not generated-frame review
or a guarantee of pixel-perfect continuity. For I2V, only the first scene is aligned
to the supplied opening image; subsequent scenes follow the inherited state.
Compact, Edit and Storyboard keep their existing shot behavior. Disable Hold when
changing Shots or multi-shot duration. No new output sockets or sampler changes.

## Pending improvements

- Visual director for chained video: inspect the actual generated tail frames before writing
  the next segment, then adapt its action and camera continuity to the observed result.
  Carry forward reference roles, completed events and the requested final outcome. Derive any
  timestamp compensation from the actual trimmed context and FPS. This remains future work;
  the current director plans the entire sequence in one call without seeing generated segments.

## Install / update

Install through Comfy Registry or clone this repository into ComfyUI's
`custom_nodes` directory, then restart ComfyUI. Updating an existing clone uses
`git pull`. Core ComfyUI supplies the Python imaging/tensor dependencies;
faster-whisper is no longer required by this package.

## Breaking cleanup

Version 2.0 removes the other story/video/LTX directors, transcription and
unused utility nodes. Older workflows using removed IDs must be migrated or
use an earlier version. This package does not modify the generation model,
sampler recipe, or LoRAs.

Offline checks: `python -B tests/test_compact_provider_routing.py`.
No model generation is performed by that test suite.
# Vision Prompt — Visual Director

`bypass` sends `request` unchanged to `prompt`, including whitespace, without an
LLM call. It takes priority over Hold and all planning controls, returns an empty
`sheet_plan`, and does not overwrite the saved Hold result. The LLM connection is
optional for bypass but required for normal generation. Connected upstream nodes
may still execute under ComfyUI's graph evaluation.

`target_model` selects `MiniMax H3` (backward-compatible default) or `Qwen`,
independently of the LLM provider. Both targets support all four modes. H3 retains
`<Picture N>` / `<Subject N>` definitions and retention sections. Qwen uses an
actionable prose prompt with `<imageN>` for multiple connected images and natural
reference wording for one image. Its sizing advice (`wh_ratio` / `ratio_follow`)
is exposed in `sheet_plan`, not injected into the image prompt or wired to canvas
controls. Configure actual generation dimensions separately. Change targets with
Hold off; held prompts from another target cannot be reused. Qwen's sheet plan is
its rewritten prompt plus sizing metadata, not the H3 panel-ledger schema.

`Edit` modifies an existing image or sheet: connect the source to
`reference_image_1` and write the requested change. Images 2–4 are optional
supporting references. The prompt uses `<Picture N>` and `<Subject N>`, with
subject definitions and explicit change/preserve instructions. Panel count,
layout, aspect ratio and annotation controls are ignored in Edit; request any
intentional layout or text changes in the brief. It preserves untargeted content
by instruction, not by a pixel mask. Connect the source to the downstream image
editor too. Hold retains the previous result even after changing modes; disable
it to generate a new edit prompt.

`Hold` reuses this node's last successful prompt and plan without calling the LLM.
Generate once with Hold off; while on, all input changes are intentionally ignored.
Text-only plans persist in the existing `output/Sexy AI Studio/director_hold` cache.

`Vision Prompt — Optional Reference Resize` accepts a missing IMAGE and returns
None with width/height zero. This is not a black placeholder or a universal bypass:
the downstream node must accept absent references. The installed native
TextEncodeQwenImage21 explicitly skips None images. Supports total-pixel sizing,
fit, stretch and centered crop with native ComfyUI interpolation. It does not
replicate KJ padding, masks or GPU/VSR options. Leave this resize active; bypass
the upstream Load Image to disable a reference. Required image consumers still
need a real image.

`Vision Prompt — Visual Director` is independent of the H3 video director.
It writes one image prompt for an entire sheet, not one prompt per generated frame.
Choose `Image` for a standalone image prompt from a written idea, with no visual
input required. Connected images optionally supply references; sampled video can
also inform the composition. This mode supports MiniMax H3 and Qwen, Direction
Controls, Hold and bypass. Scenes / Panels, layout and annotation widgets are
ignored; it does not add a sheet grid or captions. Actual images are still created
by the downstream generation workflow, not this director.
Choose `Storyboard`, `Character Sheet`, or `Custom Sheet`; `Scenes / Panels` sets
the number of panels/views in that one image (1–24 in the UI). It supports grid,
horizontal and vertical layouts, whole-sheet aspect ratio, and `panels_only`,
`brief_labels` or `production_notes`. Production direction is English; the
annotation language controls visible lettering. Image models may render lettering
incorrectly, especially in dense sheets.

Connect an existing `LLMMODEL` provider and optionally the existing Direction
Controls output to `direction_context`. Up to four reference images are supported.
`reference_video` accepts a decoded IMAGE batch, such as VHS output, not a video
file path. This experimental input uniformly samples 3, 5 or 10 frames; it does
not analyze audio or guarantee complete action coverage. Visual inputs require a
vision-capable LLM. Text-only planning also works. All planning uses one LLM call;
there is no automatic paid retry, image generation, model download or file output.

Connect `prompt` to the text-conditioning path in your Qwen Edit or other image
workflow. Connect the actual identity references to that generator as well, in
the same connected-input order shown in the prompt. Video samples are planning
evidence only and are not numbered image-generation references. This node does
not pass through image tensors or configure the sampler/canvas automatically.
Set the image workflow's actual aspect ratio/resolution to match the sheet.
`sheet_plan` exposes the panel plan; `validation` reports warnings; `usage_stats`
reports provider usage. The seed controls the LLM request where supported, not
the image sampler: connect/set the sampler's seed separately for repeatability.
More panels may require a larger `max_tokens`; truncated responses fail clearly
instead of returning an incomplete sheet. Exact rendering and identity retention
depend on the downstream image model and connected references.

# Storyboard mode

Select `Storyboard` in the compact director and connect a planning sheet to
`storyboard_image`. Use a vision-capable LLM. `continuous_scene_count` selects
1–12 generated scenes and `seconds_per_scene` sets each scene's duration (1–30 seconds).
An empty request means faithful adaptation; use the request box for intentional
changes. Increase `image_max_dimension` when small panel annotations are unreadable.

The director plans all scenes in one response, with numbered shots inside each
scene. It uses the existing `scene_prompts` JSON output contract and permits
storyboard-motivated cuts instead of enforcing continuous-camera boundaries.
Invalid, missing, duplicated or out-of-range shot timestamps are automatically
redistributed within each scene, preserving all shots in their supplied order.
Valid timing is retained to millisecond precision; repairs require no extra LLM call.
Planning warnings appear in the preview and `validation`; they do not stop the
workflow. Check them before sampling. Panel interpretation and timing feasibility
remain LLM judgments, not guarantees.

The sheet is only planning input: do not connect it to H3's reference/first-frame
inputs. Separate identity references may use the existing reference sockets.
This mode does not crop panels or rewire the generation workflow automatically.
Disable Hold after changing the sheet or request. Existing modes ignore the new
optional socket. No per-scene last-frame inspection is performed.
