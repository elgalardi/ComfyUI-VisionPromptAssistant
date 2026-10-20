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

Transfer has two prompt routes selected by the existing connections. Source Video 1/2 uses
`<Video 1>` / `<Video 2>` source-linked edit wording: the requested change, assigned Picture
appearance when present, and unaffected observed performance. Socket numbering is retained,
including when only Source Video 2 is connected. Driven-only input retains direct scene
description rather than becoming a numbered video source. No new sockets, analysis calls,
workflow rewiring or generation settings are introduced; Enhance is unchanged. Restart and
disable Hold once when testing a different input route.

Numbered-video Transfer returns internal structured scene fields: the source-video selection,
edit instruction, subject definitions, retention analysis, observed performance and sound/music.
The node assembles the six full-reference sections in order: `subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`.
Summary binds the edit to the selected `<Video n>`; each Subject definition states which source
owns identity/appearance and which owns performance. Unrequested content is retained, not a
background or wardrobe the user explicitly changes. No universal adapter trigger is inserted.
This prevents losing the source label when the LLM writes generic generation prose. It does not
prove that the model correctly interpreted motion; speech remains disallowed unless requested.
Public `scene_prompts` are still strings. Driven-only Transfer and Enhance retain their formats.
With character references and video/Driven evidence, Transfer first returns an internal
`transfer_cast`: separate observed reference/performance outfits and the selected wardrobe owner.
The node builds shared Subject definitions from that selection rather than free-form definitions.
For a Driven-owned outfit it uses the mapped blueprint role's observed garments directly.
Explicit video-clothing requests override strength defaults and styling controls. User-designed
outfits, reference outfits and established first-block designs remain selectable independently.
Actions refer to those Subjects rather than repeating garment descriptions. This adds no LLM
call; the small internal fields share the existing output budget. Source-video observations and
ownership choices still depend on the LLM's visual/language interpretation, not a verified vision
measurement. Other modes and Driven analysis are unchanged.
Numbered-video Transfer uses its own attribute-ownership rules, not Driven Strength defaults.
For whole-character transfer, the assigned image supplies visible identity, hair, body, outfit
and accessories; the video supplies performance, not the photograph's pose or incidental phone.
Accessories here means worn items such as jewelry, not incidental handheld props. Source-video
Transfer excludes photo-only props throughout the prompt, checks hand/contact consistency, and
describes moving footage rather than video stills unless that treatment is requested.
Face-only swaps and explicit wardrobe/environment changes override only their named attributes.
Without a replacement image, source appearance is retained except for requested changes.
The [adapter author's guidance](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)
also separates replacement appearance from source performance; no extra trigger word was trained.
This route follows the reference-label/retention structure in the
[official full-reference guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt).
It remains prompt guidance, not a guarantee of character identity or frame-exact motion transfer.

Transfer separates a character image's identity/appearance from a video's performance at all
three existing Driven Strength levels. Keeping the image's appearance does not keep its static
pose, smile, gaze, framing or background. Observed video action, contacts and physical progression
govern the footage unless explicitly overridden; user-assigned image roles and I2V frame zero
remain protected. Driven-only performance is described directly without a fabricated `<Video n>`
binding. This adjustment is confined to Transfer rules: Driven analysis, strength policies and
Enhance are unchanged. It does not add speech analysis or guarantee lip sync from sampled frames.

Select `Transfer` in the H3 director for faithful source adaptation rather than creative story
development. Explicit user changes take priority, then reference identity/Driven Strength, then
observed Driven action, mood, setting and camera. Creativity is limited to explicitly requested
changes. A reference woman in jeans plus a video of a running footballer and a request for office
clothes produces that woman running in office clothes, not an invented football story.

Transfer supports Driven Image, Driven Video and all three strength levels. It uses the existing
`scene_prompts` sequence output even for one scene; set `Scenes — Sequence` to 1 for a single clip
or for a workflow that supplies one source-video block at a time. Multiple scenes partition the
observed/requested action chronologically, without compulsory escalation, new locations or cuts.

Dialogue language alone does not request speech. Unrequested dialogue, lyrics, narration and score
are excluded; `non_diegetic_music` is `N/A` without a music request. Sampled frames do not provide
audio or frame-exact motion. Explicitly requested dialogue/creative changes remain supported.
Transfer uses its own source-faithful system contract instead of the editable base system prompt.
Auto/none Direction Controls are neutral. Active selections explicitly override source defaults
within their scope: camera/motion controls affect capture/performance; genre/look can redesign
wardrobe, environment, lighting, palette and atmosphere without inventing a plot or cast.
Written user locks and I2V frame zero take priority; unselected attributes retain their source
assignments. Language alone does not add speech, and genre/look alone does not add music. No creative
treatment or extra review call is added. Normal Driven analysis and JSON repair behavior remain.
Output-token budgeting scales like Enhance with scene count/duration. Disable Hold once when
switching modes; cached prompts from another mode cannot be reused. These are LLM instructions,
not a guarantee of exact motion transfer or perfect downstream model adherence.

For per-block continuous Transfer, connect the Source Video Block `request` output to the
director request and keep the existing Prompt Override/Save checkpoint path. The block supplies
the first accepted prompt as the established design and the previous prompt as ending-state
context. Transfer retains casting, appearance and selected styling while following new source
action; it does not restart the first pose, redesign each window or restart requested music.
No extra LLM call is added. Without that request/state wiring, independent runs have no history.

## Cinematic craft

### Writer-director development

Direction Controls omit Auto/none categories and keep explicit selections in priority order.
Creative modes resolve them before designing footage: genre supplies a dramatic/visual mechanism,
period/world looks guide unlocked appearance and production design, and motion/camera controls
guide readable physical cadence and coverage rather than blanket exaggeration. A selected era
does not need to be repeated in the user request. Explicit locks, Driven appearance, I2V and
faithful edits remain authoritative. Creative modes can develop a story within continuous
coverage; they are no longer subject to the generic no-new-plot edit rule.

Enhance, Elaborate and Elaborate Plus now distinguish an open story brief from a faithful
edit. Open premises authorize a concrete situation, character goal, resistance, choice and payoff,
with purposeful dialogue when permitted. Explicit action-only briefs, source plates, Driven Video
chronology, reference roles, I2V opening frames and locked controls remain authoritative. Transfer,
Compact, Edit and Continuous Edit do not receive this new creative mandate.

The creative modes choose story shape, subtext, performance, viewpoint, blocking, lens perspective,
light and sound as one coherent treatment, not a mandatory checklist or extra screenplay output.
They retain their current JSON, scene counts, output budgets and provider calls. The additional
system instructions consume input tokens; this is not a zero-cost change. Disable Hold once after
updating to test the new writing. Restart ComfyUI to load it. Offline tests verify routing and
contracts, not artistic quality or downstream H3 adherence.

Selected period/world looks in creative modes now adapt unlocked wardrobe, grooming, production
design and sound as well as photography. Color-only/stock looks do not require an era change.
Explicit locks, Driven appearance policy and I2V opening-frame facts remain protected.
Sequence blocks continue camera, action and physical state. No additional
provider calls or output-budget increases are introduced; input instructions are longer.
Creative responses now resolve a shared `target_design` (separate identity and target appearance
per subject, plus visual language), then opening framing, action, visible result and sound/music
fields per scene. The node renders these into the
same public string prompts with all four H3 headings, reusing the design without requesting a
separate planning pass. Empty audio fields render as `N/A`, not invented music. Single Elaborate
uses the same structured compiler; faithful modes and previously held string prompts are unchanged.
Opening framing and earned end-state fields are compiled into the footage text, not discarded
planning metadata. A pan is not a new shot marker. Open continuous stories should enact their idea
and show a consequence rather than end by proposing future action; explicit open endings,
trailers, faithful chronology and appearance locks remain valid. The period treatment calls for
intentional unlocked styling rather than copying source clothes under an era label. These are
writing contracts, not a deterministic check of historical accuracy or narrative quality.
Dialogue guidance favors short purposeful exchanges with room for
performance rather than explaining the plot. The existing total output limit includes internal
design/planning fields; scene count and sockets are unchanged. Artistic adherence still needs
an actual generation test, and a malformed-JSON repair may still use the existing retry.
The implementation draws on coordinated art/costume/cinematography described by
[ASC — Robert Yeoman](https://theasc.com/article/yeoman-asc-awards-honoree/)
and purposeful coverage planning in
[StudioBinder — shot-list design](https://www.studiobinder.com/blog/whats-in-your-shot-list/).

Requested formats steer the treatment: drama, trailers, comedy, horror, commercials, music videos
and other or mixed forms share the same route, not separate mandatory formulas. Factual/product
briefs do not authorize invented facts or claims. Inspiration from Higgsfield's documented prompt
enhancement and ByteDance's narrative examples does not reproduce a private system prompt or
Seedance's model capabilities:
[Higgsfield Cinema Studio](https://higgsfield.ai/blog/how-we-built-cinema-studio),
[Seedance 2.5](https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5).

Research references (original implementation; no external package or copied prompt required):
[John August — scenes](https://johnaugust.com/2007/write-scene),
[dialogue](https://johnaugust.com/2007/how-to-write-dialogue),
[ASC — translating a script into a visual approach](https://theasc.com/article/shot-craft-analyzing-a-script/),
[Director Skills — scene construction](https://github.com/0xhughs/director-skills/blob/main/skills/screenplay-and-scene-writing/references/scene_construction.md)
and [cinematic language](https://github.com/0xhughs/director-skills/blob/main/skills/style-cinematography-director/references/cinematic_language.md).

The video and visual directors include compact cinematic-craft instructions: causal progression,
persistent physical state, motivated framing and concrete staging. Existing mode, source-role,
scene/panel-count and JSON contracts take priority. Video sequences are still planned in one call;
image prompts use static viewpoints, and local edits preserve untargeted content. These are prompt
instructions, not a guarantee of generated continuity or an analysis of audio from sampled frames.

Elaborate and Elaborate Plus now compare three concise creative treatments internally and
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

### Continuous sequences

The director has no Cut toggle. Enhance and Elaborate Plus carry
camera/action and established physical state forward between blocks. Creativity develops the
story within that continuous coverage rather than forcing changes of viewpoint or location.
Existing SimpleH3Chain context modes, including `masked_av`, are unchanged. Restart ComfyUI and
refresh the browser; recreate the director if an obsolete Cut widget remains. Disable Hold once
to regenerate rather than reuse a previously stored edited sequence.

## Selective directing craft

General narrative/world/form/rendering genres have distinct mechanisms rather than one generic
genre instruction: crime needs a specific consequential crime/clue, heist an objective/security
constraint, documentary evidence without invented facts, animation a coherent rendering grammar.
Compatible genres develop one situation, not unrelated alternating clips. Period looks can style
unlocked garments/grooming/props/world; capture/grade-only looks do not independently change those
attributes. Cinematic does not imply darkness or slow motion, and camera selections override a
look's default operator behavior. Keep Reference, user/source locks and requested action still win.

Selected controls compile into observable behavior on independent axes: body cadence/force,
camera mechanics, look, genre and design. Fast Motion requests faster-than-everyday primary action
with quick initiation and compact recovery, not more events, sped-up speech or slow-motion posing.
Fast subjects can combine with a locked camera or orbit; smooth/fluid camera motion need not slow
the acting. Orbit specifies pivot/arc/parallax; dolly translates, crash zoom changes focal length,
and whip pan does not imply a cut. User/source locks and faithful audio/choreography remain protected.
Creative Plus receives the resolved control definitions in its existing text-only planning request.
No new widgets/options, extra calls or output budgets; only active controls add these instructions.

Requested fictional conversations need concrete spoken lines, not generic talking instructions.
The creative director chooses a relevant topic, connected replies and stable voice habits; words
appear once in the integrated description, with delivery outside dialogue tags and active listeners.
Subject N uses speaker SN regardless of speaking order. A language selection alone does not require
speech, and silence/source-audio locks remain protected. No extra calls or output fields are added.

Standard keeps continuous coverage by default; written requests can direct cuts and montage.
In both Standard and Creative Plus, active Direction Controls resolve into final design and
performance choices before footage: look/era into unlocked appearance/world, genre into situation,
motion into cadence and camera into coverage. Selected period looks receive only their own brief
production-design options, not all eras' examples; capture-only looks and Auto/none do not authorize
restyling. Existing scene/design fields ask for concrete garments, materials, settings and behavior.
Plus may improve provisional story beats but retains the resolved art direction across shots/blocks.
Explicit user/reference/source locks and I2V facts win. No new calls, fields or output budget are added;
the selected options add a small amount of instruction input. These are prompt contracts, not a
guarantee that the model realizes the look. Transfer's request path is unchanged.
Creative Plus decides motivated coverage in its existing treatment: one sustained shot or
edited views for location/time changes, reveals, details and reactions. There is no coverage
selector, fixed shot count or midpoint cut. Explicit single takes and locked source coverage win.
It shares Standard's private scene fields: `opening_framing`, `integrated_multimodal_description`,
`visible_result` and sound/music. Timed `[Shot N]` cuts belong in the integrated description;
the renderer does not force a shot count or split generation clips. Shot-number and clip-local
timestamp normalization remains in place. Sampler/context behavior and Transfer are unchanged.
Character references have exclusive role ownership: supporting people and background extras
do not inherit a lead's identity. Recognizable invented supporting cast receive independent
visual handles; distant extras use scale-appropriate silhouettes, uniforms and activity.
These instructions also apply to Creative Plus planning, without another request. Faithful
video edits preserve existing source participants and replace only the assigned performer.
This mitigates prompt ambiguity, not model-level global reference conditioning or guaranteed
identity separation in the generated video.
Reference numbering follows `Picture N → Subject N → (SN)` rather than entrance or
speaking order. For explicit unambiguous subject definitions, the compiler reorders
definitions and renumbers Subject/speaker mentions together, leaving Picture tags
and spoken words unchanged. Ambiguous multi-picture identities are not guessed.

### Creative Profile

`Elaborate Plus` is the renamed `Continuous Elaborate` edit mode, with the same sequence behavior.
Saved UI workflows migrate the old label on load; the backend retains its internal mode identifier.
This is separate from the `Creative Plus` profile described below.

Creative modes budget the narrative across the selected scene count and duration: one scene
contains a mini-arc, two share setup/development and resolution, and longer sequences distribute
changing beats toward a visible payoff. The last scene reserves time for completion and reaction,
not another setup. Standard and Creative Plus share this contract without additional calls.
Explicit open endings, trailers, loops and source-locked chronology remain protected; source-block
continuations do not invent a fresh story or forced finale for every block. This is prompt guidance,
not a guarantee that the video model executes every beat.

The director's appended **Creative Profile** widget defaults to `Standard` (unchanged).
`Creative Plus` adds one text-only treatment request before the normal prompt-writing request
in Enhance, Elaborate and Elaborate Plus. It prepares a compact premise, character
wants/relationships/voice/movement, production design, scene beats and continuity locks.
The planner gets the brief, controls, source-slot roles and any existing Driven blueprint,
not another batch of image/video tokens. It cannot inspect reference pixels; final direction
reconciles the treatment against the actual references and explicit locks. The planning output
budget is 770–1540 tokens for 1–12 scenes; the existing final-prompt budget is unchanged.
Normal runs use two requests, or three when Driven analysis is connected. Planning input/output
usage is reported separately. Quality and actual credit cost require comparison runs.

Creative Plus uses the same writing/craft tools and final scene schema as Standard. Its compact
treatment suggests character voices, purposeful coverage, authorized design and causal development;
the final writer can improve or replace weak suggestions after seeing the references. A few better
decisions matter more than extra events or complexity. Tone and stakes follow the user's brief,
not an obligatory threat, conspiracy or new subplot. The continuity notes retain established facts,
not provisional camera/pose choices. Closed scripts and source performance remain protected.
Plus no longer stacks the faithful-edit/continuous-take rule sets over its creative brief or uses
a separate structured shot list. Shared control execution still applies without Standard's generic
no-new-cuts footer. Custom system instructions remain authoritative. Standard/Transfer behavior,
request counts, output budgets and public fields are unchanged.

For open fictional stories, a short user idea is the required core, not one literal clause per
scene. Creative Plus aims to fulfill a brief core early when physically feasible, then develops
causal consequences, decisions and changing relationships rather than stretching setup. It plans
readable acting/viewpoint changes and reserves time inside the last clip for a visible payoff and
reaction. This is duration-aware guidance, not a fixed scene quota, mandatory cuts/new locations
or a forced happy ending. Closed scripts, isolated action studies, faithful source chronology,
supplied endings and deliberate open forms stay protected. The same guidance survives an invalid
extra-treatment fallback and does not add calls, fields or output-token allowance.

Creative Plus distinguishes narrative continuity from an unbroken camera take. Its existing
production-design/continuity notes retain recurring location layouts and progressive wardrobe,
props and actions. Scene beats choose whether and why to cut. A change of location establishes
the complete destination after the cut rather than gradually blending sets; unimportant travel
can be elided when coverage is unlocked. Shot actions own their dialogue once, and the final
result belongs to the last shot. In AV continuations, inherited footage remains the opening;
planned cuts belong later in the generated interval, not inside the protected prefix. The
director does not inspect generated endpoints or know an arbitrary workflow's prefix duration:
these are prompt instructions, not timeline enforcement or a guarantee against model morphing.
No extra model call or output-token allowance is added; more coverage shares the existing budget.

Malformed planning JSON is repaired locally only for trailing commas and unescaped control
characters inside strings. Unrecoverable or structurally invalid treatments are skipped: the
normal director request uses Standard's creative treatment instructions, with no planning retry.
Validation and usage report this fallback and still count the attempted planning call. Provider
and transport failures remain explicit; no incomplete story text is fabricated to close JSON.

Source-block continuations with `FIRST ACCEPTED BLOCK` reuse the matching node's cached
treatment. After a cold resume they use that accepted first prompt as the established design
instead of paying for another planning call; a complete private bible is not reconstructed.
New first-block runs plan again. Hold reuses the saved result without any model call; changing
profile requires disabling Hold once. Bypass skips both stages. Transfer, Compact and faithful
Edit modes do not activate Creative Plus. No new public output ports or prompt headings are added.

Direction Controls includes independent **Wardrobe** and **Environment** choices:
`Auto` keeps existing behavior; `Keep Reference` preserves the assigned source design;
`Create New` authorizes a concrete new outfit or setting suited to the brief and selected look.
These override source/Driven defaults only on the selected axis. Written user assignments
and I2V frame-zero constraints win; Keep Reference protects that axis from genre restyling.
A new design is chosen for the sequence and carried into subsequent blocks, not regenerated
per scene. New wardrobes do not replace facial identity or video choreography. The controls
are appended after Dialogue, preserving existing widget order. No extra LLM stage is added.
With Create New in Enhance/Elaborate, the existing structured response includes a brief internal
source-to-outfit contrast and/or a concrete per-scene environment field. The renderer emits the
chosen garments and setting, not the contrast audit; Picture bindings are identity-scoped for
redesigned outfits. Named venue categories and explicit garment details stay protected while
unlocked silhouette, material, layout and lighting choices can change. These checks improve
instruction specificity, not a deterministic guarantee of novelty or a new validation failure.

**Body Proportions** is independent: `Auto` leaves the existing policy unchanged;
`Keep Reference` retains the assigned reference's visible build; `Creative Adaptation`
permits a coherent target build/proportion treatment within the brief, without automatic
exaggeration or facial/age changes. Pose and choreography remain assigned to their original
sources, and clothing fits the chosen body without replacing a protected outfit. Written
body locks and I2V frame-zero facts win. The chosen body persists across blocks. Creative
modes compile a concise `target_body` into subject definitions; Transfer uses its existing
identity-details field. No extra LLM stage or changed default behavior is introduced.

Transfer resolves identity, wardrobe, body performance, camera and world as independent
responsibilities. Performer bindings follow the original role through crossings, not a fixed
screen position. Camera-only and body-only requests remain scoped; no compulsory camera style,
beautification, mouth closure or invented transcript is added. Driven analysis is unchanged.
Enhance, Elaborate and Elaborate Plus use a small internal narrative canon (role, want,
pressure, voice, movement and relationship), observable subtext, listener reactions and
framing-aware detail. Delivery notes stay outside dialogue tags. These are selective principles,
not imported platform templates or mandatory aesthetics. Existing JSON/source tags, output
budgets and request counts are unchanged; no separate bible or interview is required.
Disable Hold once to regenerate with the revised instructions. Model results still need testing.

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
