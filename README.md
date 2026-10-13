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

## Transfer mode

Select `Transfer` in the H3 director for faithful source adaptation rather than creative story
development. Explicit user changes take priority, then reference identity/Driven Strength, then
observed Driven action, mood, setting and camera. Creativity is limited to explicitly requested
changes. A reference woman in jeans plus a video of a running footballer and a request for office
clothes produces that woman running in office clothes, not an invented football story.

Transfer supports Driven Image, Driven Video and all three strength levels. It uses the existing
`scene_prompts` sequence output even for one scene; set `Scenes — Sequence` to 1 for a single clip
or for a workflow that supplies one source-video block at a time. Multiple scenes partition the
observed/requested action chronologically, without compulsory escalation, new locations or cuts.
The creative Cut toggle is inactive in this mode; evidenced or explicitly requested cuts remain valid.

Dialogue language alone does not request speech. Unrequested dialogue, lyrics, narration and score
are excluded; `non_diegetic_music` is `N/A` without a music request. Sampled frames do not provide
audio or frame-exact motion. Explicitly requested dialogue/creative changes remain supported.
Transfer uses its own source-faithful system contract instead of the editable base system prompt;
compatible Direction Controls remain subordinate to the request and observed action. No creative
treatment or extra review call is added. Normal Driven analysis and JSON repair behavior remain.
Output-token budgeting scales like Enhance with scene count/duration. Disable Hold once when
switching modes; cached prompts from another mode cannot be reused. These are LLM instructions,
not a guarantee of exact motion transfer or perfect downstream model adherence.

## Cinematic craft

The video and visual directors include compact cinematic-craft instructions: causal progression,
persistent physical state, motivated framing and concrete staging. Existing mode, source-role,
scene/panel-count and JSON contracts take priority. Video sequences are still planned in one call;
image prompts use static viewpoints, and local edits preserve untargeted content. These are prompt
instructions, not a guarantee of generated continuity or an analysis of audio from sampled frames.

Elaborate and Continuous Elaborate now compare three concise creative treatments internally and
commit to one coherent approach; Enhance compares three within a smaller detail budget. This adds
no API stages or output fields. Explicit requests, reference roles, locked controls and edit scope
still take priority. Compact and Edit do not receive this extra
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

### Cut between scenes

Enhance and Continuous Elaborate expose `Cut` (default false). Off preserves
continuous handoffs. On designs motivated scene-boundary cuts with distinct framing,
angle or composition while preserving identities, wardrobe, geography, causal order
and the preceding scene's resulting state. Each scene also develops a distinct
action, interaction, reveal or consequence, not merely a new view of a repeated beat.
Open premises allow a coherent mini-story and motivated time/location changes;
Cut uses separate story-design instructions instead of continuous/edit preservation
and handoff rules. It plans distinct scene purposes/events and camera strategies,
using Driven as a starting premise rather than repeating its setting indefinitely.
An internal `cut_scene_plan` pairs each scene's event, setting and framing before
writing the prompts; `change_from_previous` specifies the new event/result and
an unmistakable coverage contrast. Adjacent scenes must develop both activity and coverage.
This outline is omitted from public outputs and adds no separate LLM call.
Cut uses a single story contract plus explicit priorities rather than repeated override blocks.
Each event achieves a visible result; a later scene acts on an earlier discovery instead of
rediscovering it. Selected genre develops unlocked story/world without changing assigned design.
All modes budget movement, reactions, dialogue and prop handling against clip duration.
Fast Motion keeps brisk, readable mechanics rather than packing in extra events.
Camera direction specifies a concrete viewpoint/path; authorized setting changes
reinterpret Driven lighting through suitable sources without overriding explicit locks or style.
Explicit event order, outcomes and source locks remain binding. Non-narrative footage
varies choreography, play or visual ideas without forced conflict. There is no fixed internal shot count.
Single-scene Elaborate, Compact and Edit remain unaffected.

Pair Cut manually with Masked Cut in the workflow. This changes prompts only,
not context or sampler settings. For I2V only scene 1 aligns to the supplied first
frame; later scenes inherit story state, not the original pose. H3 sections and
string prompt outputs remain unchanged, with no generated-frame review.
Each clip restarts at `[Shot 1]` without a timestamp. Accumulated opening timestamps
are removed and later shot times are rebased to that clip, including held prompts.
Disable Hold when changing Cut. Restart ComfyUI and refresh the browser; replace
an old director instance if its retired Shots widget persists.

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

# Driven Image / Driven Video

The H3 director no longer exposes Mood inputs or a Storyboard mode/planning socket.
Connect Driven Image or decoded frames to Driven Video with a vision-capable LLM.
`Driven Strength` applies to both inputs in every active director mode:

- **1 — Identity First:** keep reference face, body proportions, hair, outfit and accessories;
  use Driven action, pose, interaction, mood, camera and lighting, adapting the setting as needed.
- **2 — Balanced:** keep reference face and primarily its body proportions; transfer Driven
  wardrobe, accessories, hair, pose, activity and environment, fitting them to the reference body.
- **3 — Full Transfer (default):** retain reference facial identity while transferring Driven
  wardrobe, hair, body proportions, pose, activity and world. This preserves the existing policy.

Explicit user changes/locks, age and an active I2V opening frame take priority at every level.
Without character references there is no reference appearance to preserve; observed roles form
the character descriptions. Without a Driven input the setting has no effect. Strength selects
prompt rules, not a sampler weight, and does not add LLM calls. Disable Hold once when changing
strength; a cached prompt from another level is rejected without calling the LLM.

The following full-transfer examples describe level 3; levels 1 and 2 preserve the attributes above.
These sources drive the full scene: activity, wardrobe/accessories, environment,
staging, lighting, palette, visual style, mood and visible interactions.
Driven Video additionally contributes observed motion/progression;
sparse sampled frames provide neither audio nor exact unsampled choreography.

Numbered character references supply the cast's identities, not automatically locked clothing.
For example, Picture 1 (woman), Picture 2 (man) and a Driven Image of a football
match produce those characters playing football in adapted kits and boots, with
the match's setting, lighting and atmosphere, not new player faces.
User instructions override conflicting driver evidence; explicit locks and reference
ownership still apply. Drivers may replace incidental reference outfits and mood,
but not identity, explicitly assigned wardrobe or preserved audio. The adapted
wardrobe/world remains consistent across scenes unless the requested progression changes it.
Driver people supply roles, never additional principal identities: a referenced woman
and man plus a cheerleader driver means the referenced woman performs that role and
interacts with the man, not a third cheerleader. Explicit user role assignments take
priority; ambiguous roles are allocated within the existing cast. Only an explicit
request may add principal characters. Incidental crowds do not become interacting leads.
With both sources, compatible image staging and video action are combined; no
unsupported events or dialogue are invented from missing evidence.

All active director modes receive this scenario-transfer contract, while retaining
their existing final outputs and detail budgets. Driven uses two LLM stages:
first image/video-to-prompt analysis produces a structured anonymous scene blueprint
covering activity, garment design, hair styling, body proportions, pose, setting, lighting,
mood, style, props and spatial roles. Reference facial identity stays intact while mapped
Driven hair, build, silhouette/proportions and pose can form the adapted character.
Explicit body/pose/hair/outfit locks, age and an active I2V opening frame take priority.
Perspective and garment volume are not anatomical evidence; illustrated proportions are
adapted to coherent anatomy within the chosen style. The observed interaction supplies
the premise for a plausible next action, not an unrelated story or a repeated static pose.
Visible neckline, sleeves, cutout geometry/placement, silhouette, materials and accessories
are described rather than reduced to a generic garment label; hidden details are not invented.
The complete mapped outfit persists across scenes. A local user override changes only the
specified item: requesting a cap does not also discard the Driven sweatshirt or vest.
Fast Motion specifies readable subject action and concrete camera behavior, not generic
`responsive camera work` or obligatory camera movement.
Disable Hold once after this update to replace older Driven analysis without body/pose fields.
Then the director sees that text and the numbered reference images, not the original
driver people's images. It assigns each role to an existing Picture tag using a
validated internal mapping before writing the prompt. This separates scenario extraction
from identity casting rather than treating Driven as a third character reference.
Final prompts describe transferred outfits and scenery directly, without Driven
assignment boilerplate or `from the Driven role` annotations.
Analysis adds one LLM request with a 700–1200 output-token allowance, using the same
provider/model and seed. Its usage is reported separately; Hold reuses the finished
plan without repeating analysis. No paid calls are made by the offline tests.
Final prose and downstream identity retention remain model-dependent; the mapping
does not guarantee that the generator perfectly follows the finished prompt.

Restart ComfyUI, refresh the browser and recreate old director instances if retired
Mood/Storyboard sockets persist. Reconnect those sources to the new Driven sockets
and disable Hold to generate a fresh interpretation. The separate Visual Director's
storyboard-sheet creation modes are not changed by this removal.
