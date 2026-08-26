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

## Local Whisper Transcribe

**Local Whisper Transcribe** accepts a native ComfyUI `AUDIO` value, converts it
to mono 16 kHz in memory, and transcribes it locally with faster-whisper. It
offers `large-v3` for parity with the standalone Captioner and `large-v3-turbo`
for faster experiments, plus automatic language detection, VAD, beam-size and
CUDA/CPU compute controls. Outputs include the complete transcript, timestamped
segment JSON, detected language and a concise status string.

Models download on first use to `ComfyUI/models/faster-whisper`. The Whisper
backend is deliberately short-lived: after every transcription the node unloads
CTranslate2, releases Python references and clears available CUDA cache before
MiniMax H3 begins loading. Silent audio is detected before loading the model.

## H3 Story Director

### Experimental H3 Toolkit prompt rules

Enable `toolkit_prompt_rules` to request self-contained per-scene rendered prompts,
positive camera descriptions, Subject-bound retention (`<Subject N>` defined from
`<Picture N>`), and non-blocking lint findings in the validation output. It is off
by default so existing workflows retain their established prompt contract.

### Experimental Power mode

Enable `power_prompt_rules` for the strictest Director workflow. Power creates a
private production blueprint before writing scenes, maps every explicit user request
to a scene, assigns one scoped job to every reference, limits action density to the
available duration, and records an entry/exit state ledger for cast, pose, contact,
wardrobe, props, geography, camera, lighting and audio. Continuous Story requires
each scene to inherit the preceding exit state exactly; Cinematic Cuts may reset the
camera while retaining the latest world state. It also enforces official Subject /
Picture / speaker roles, H3 dialogue markup, sound/music separation and a final-scene
payoff. A deterministic validator requests one complete automatic repair when the
first structured plan is inconsistent, then rejects a second invalid plan before GPU
generation. Power takes precedence if Toolkit is also enabled. It is experimental and
may use more LLM tokens than the standard Director. Its private blueprint remains
detailed, while the rendered H3 scenes use compact chronological prose: establish the
location and reference identities once, describe visible actions in order, record each
resulting state change where it occurs, and omit repeated contracts or planning labels.
The shared prefix contains only immutable identity/style information. Every scene receives
its own short positive context for its current wardrobe, environment, props, camera and
audio state, so completed changes cannot be overwritten by an obsolete global description.

**H3 Story Director** is a multimodal planner for MiniMax H3. It accepts an
optional story idea, up to four character or subject images, and an optional
source-video `IMAGE` frame batch from VHS Load Video. It uses an OpenRouter vision model with strict structured
output, then validates the response locally before returning a `plan_json` and
the first complete mode-specific prompt.

`Director Mode` provides three clear production paths:

- `Continuous Story` preserves pose, action, camera, location and sound state
  from one scene into the next.
- `Cinematic Cuts` preserves identity and narrative state while resetting camera
  position, framing, lens, pose and movement after every hard cut.
- `Edit` automatically writes still-image generation/edit prompts when no source
  video is connected. With a VHS `IMAGE` frame batch connected, it analyzes
  `<Video 1>` and automatically infers motion transfer, character replacement or
  insertion, wardrobe, environment, style, camera/choreography, or object editing
  from the user's prompt. There is no separate operation selector.

Video analysis samples 10 frames by default, distributed uniformly from 0% to
100% of the VHS `IMAGE` batch. Each sample is sent as a separate full-detail
image instead of being reduced into a contact sheet. The Director reconstructs
one chronological progression and must track subject pose, body-part movement,
direction, intensity, contact, clothing or visible nudity, visible anatomy,
objects, setting, lighting, framing, and camera behavior. Clearly visible adult
or explicit content is described directly and objectively without inventing
details that are not visible. Observation labels and analysis methodology are
forbidden from generation prompts. Analyses shorter than the required detail
threshold are rejected before generation. The same source video must be connected
to the H3 generation workflow as `<Video 1>`.

`mode_prompt` returns the first complete prompt adapted to the chosen mode.
`source_video_analysis` exposes the chronological interpretation for inspection
and is empty when Edit has no source video. `scene_count` is always respected:
requesting multiple video-edit scenes produces that many ordered, distinct prompts
instead of silently forcing the result to one scene.

For a standalone I2V generation, set `scene_count` to `1` and connect the
`scene_prompt` output directly to the MiniMax H3 I2V prompt input. This output
contains only the complete shared prefix plus the generated scene prompt; it
does not include JSON, duration, sampler steps, validation, or usage data. With
multi-scene plans it returns the first scene's complete prompt.

The default model is `x-ai/grok-4.20`, which supports multiple image inputs and
strict structured outputs on OpenRouter. The model field remains editable.
Reasoning is disabled by default for lower latency and cost and can be enabled
for more complex story structures.

`Director Profile` separates model-specific planning without changing any public
output or downstream H3 node. `OpenRouter` and `Gemma` use model-specific planning
instructions but compile to one canonical MiniMax contract: `<Subject N>` identifies
a persistent visible subject, `<Picture N>` identifies its concrete visual source,
and `(S1)` through `(S4)` are reserved exclusively for dialogue or singing. Subject
and Picture numbers are independent, so several people may originate from one
reference image. Both profiles share cast integrity, continuity, scene-detail and
genre/look locks. Gemma alone keeps its stricter completeness validation, compact
prompt repair and malformed-response retry behavior.

`Visual Look` controls capture aesthetics independently from genre and Motion
Style. Choices include cinematic, naturalistic, raw, amateur home video,
smartphone/UGC, webcam, camcorder, MiniDV, VHS, Super 8, 16mm, documentary,
found footage, reality TV, CCTV, bodycam, action camera, broadcast, glossy
commercial, fashion editorial, soft-focus dream imagery, and hidden-camera
observation. Non-cinematic selections explicitly suppress generic film polish,
dramatic grading, artificial shallow focus, and sweeping camera language. The
selected output dimensions are never overridden to imitate a legacy format.

Genre conventions must be demonstrated in each scene rather than merely named,
and dialogue or lyrics must remain exclusively in the selected language without
translations or bilingual repetition. A deterministic style contract is prepended
for both profiles so the selected primary genre, secondary genre and Visual Look
remain authoritative when MiniMax receives the final plan.

### External / datacenter LLM

**H3 Story Director — LLM Model (API)** provides the same inputs, multimodal
reference handling, validation, plan compiler, and outputs without calling
OpenRouter. Its `llm_model` socket accepts the package's **H3 LLM Model (API)**
output as well as the `LLMMODEL` output from YALLM's **LLM Model (API)** or
**LLM Provider (API)** nodes.

**H3 LLM Model (API)** connects to an OpenAI-compatible datacenter endpoint.
Enter either a base URL ending in `/v1` or the complete `/chat/completions`
URL, plus the served model ID and optional API key. The key widget is visually
masked but may still be serialized in workflow metadata; deployment systems
should inject or remove credentials before distributing workflows. YALLM can
instead keep connection profiles and credentials in its server-side YAML.

The external Director still performs all final plan validation locally. With a
YALLM model that does not expose structured-output parameters, it supplies the
exact JSON Schema in the system instruction and rejects malformed responses
before any MiniMax render begins.

**H3 Ollama Model (Local)** is the Ollama-protocol connection for the same
external Director. Its default server is `http://127.0.0.1:11434`, the standard
address when Ollama and ComfyUI run on one computer. The address accepts a local,
LAN, remote, or hosted server root, an address ending in `/api`, or the complete
`/api/chat` endpoint; an optional masked API key supports protected services. It
uses Ollama's native multimodal chat endpoint so up to four Director images, the
exact JSON Schema, seed, temperature, output-token budget, and context length
are preserved. `thinking=false` favors speed; `thinking=true` lets supported
models reason internally before returning the final structured plan. `keep_alive=false` sends `0` and unloads
the VLM before MiniMax begins; `keep_alive=true` sends `-1` and keeps it resident
for repeated plans. The currently tested local profile is `Gemma` with
`huihui_ai/gemma-4-abliterated:12b`. Select the matching Director Profile in
H3 Story Director — LLM Model (API); the profile does not select the model itself.

`story_idea` is optional. Leaving it empty enables Full Creative Control: the
Director invents the premise and complete narrative arc from the selected genre,
Motion Style, dialogue setting, scene count and duration, additional direction,
and any connected reference images. Writing a premise keeps the original guided
behavior. New nodes also start with an empty `additional_direction` field so the
example musician story never leaks into an unrelated creative run.

`Dialogue` is a fixed selector containing No dialogue, ten of the most widely
spoken languages by total speakers (English, Mandarin Chinese, Hindi, Spanish,
Standard Arabic, French, Bengali, Portuguese, Russian, and Indonesian), plus
Japanese for continuity with the original workflow. The synopsis, story bible,
JSON plan, and production directions remain in English.

When dialogue is enabled, the Director writes every actual spoken line in quotes,
assigns it as `(S1):`, `(S2):`, `(S3):`, or `(S4):`, and keeps the exchange naturally performable
inside the selected duration. It never leaves dialogue for MiniMax to invent.
Selecting No dialogue removes spoken dialogue, narration, voice-over, and
intelligible background speech.

`audio_content` independently controls the permitted vocal and musical content.
Its first and default option, `Auto`, infers the appropriate dialogue, singing,
instrumental score or intentional ambience from the complete story while keeping
audio choices coherent across scenes. `Dialogue Only` excludes score and singing while retaining ambience and Foley;
`Dialogue and Music` adds a non-vocal score that ducks beneath speech; `Singing
Music Only` removes spoken dialogue and requires exact sung lyrics whenever lyrics
are intelligible; and `Instrumental Music Only` prohibits every spoken or sung
voice. The `Dialogue` language applies to spoken lines and lyrics. When `No
dialogue` is combined with a dialogue mode, speech remains disabled. In singing
mode it uses a language explicitly requested by the story or non-lexical vocals.

Genre is an expanded menu covering the original drama, action, thriller, horror,
comedy, romance, science fiction, fantasy, documentary, music video, anime,
animation and adult categories, plus adventure, crime, detective mystery, film
noir, sitcom, slasher, superhero, western, martial arts, heist, espionage,
disaster, psychological drama, dark comedy, and musical. Commercial and social
formats include advertising, product showcase, fashion, beauty, food, luxury,
TikTok/Reels, YouTube, vlog, influencer/UGC, corporate, sports, travel,
educational and video-podcast productions. Additional adult formats include
OnlyFans-style creator video, glamour/boudoir, pornographic, fetish, sensual
romance, erotic comedy, amateur-style, POV, couples, BDSM-themed, consensual
fantasy roleplay, intimate art film, and explicit music-video formats. Adult genres require
all depicted participants to be clearly consenting adults aged 18 or older and
reject sexual treatment of minors or age-ambiguous references.

The first genre option, `Auto`, is not treated as
a literal genre. The Director infers a coherent genre, production format, tone,
audience and visual language from the written premise, additional direction,
reference images, source-video timeline and Director Mode. Explicit
written intent takes precedence over ambiguous visual clues. If the inferred format
is adult, the same consenting-adults-only validation remains mandatory.

`Motion Style` independently controls the global action/camera cadence without
changing scene duration. Its first and default option, `Auto`, infers the best
motion intensity and camera language from the prompt, references, source video,
genre, and Director Mode. Alongside Normal, Fast, Slow, Time Lapse, Stop Motion,
Hyperlapse, Speed Ramp, Minimal, Fluid, and Intense Dynamic motion, it includes
Super Fast, Super Slow, Handheld, Gimbal, Steadicam, Locked-Off, Dolly/Tracking,
Crane/Drone, Orbit, Whip Pan, and Crash Zoom camera styles. Each selection expands
into explicit physical-action and camera guidance for the LLM. More extreme choices
include Chaotic Erratic, Frenetic Kinetic, Unhinged Handheld, Surreal Unpredictable,
and Pulsing Rhythmic motion; calmer choices include Meditative Calm, Gentle Organic,
Dreamlike Floating, Static Tableau, and Slow Observational motion.

The model writes the synopsis, continuity bible, shared reference assignments,
and scene prompts. The node—not the model—enforces the requested scene count,
duration, and sampler steps. Explicit user wardrobe and appearance overrides are
stored as authoritative mutable state above conflicting reference-image details.
Subject definitions that are not active in every scene are automatically removed
from the shared prefix and routed only into scenes where that Subject appears;
this prevents future characters from leaking into earlier generations. Connected
images retain their exact `<Picture 1>` through `<Picture 4>` tags.
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
