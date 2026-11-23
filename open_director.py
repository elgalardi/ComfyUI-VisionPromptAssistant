"""H3 Open Director: your system prompt decides everything.

No modes and no house rules. The node sends your system prompt, the request, up to four reference images and ten
frames of an optional video to the connected LLM, and returns its scenes. The one fixed rule is the scene count:
with two or more scenes the LLM is told they are one continuous story, each scene starting where the previous
one ends. The output is the director's scene JSON, so it plugs into Simple H3 Chain unchanged. Nothing here
raises: on any failure the request comes out as the scene prompt and `notes` says why."""
import json
import re

from comfy_api.latest import io

VIDEO_FRAMES = 10
DEFAULT_SYSTEM_PROMPT = ("You are a film director writing prompts for MiniMax H3 video. Turn the user's request into a "
                         "vivid, specific, cinematic prompt in English.")


def _frames(value):
    try:
        return int(value.shape[0]) if value is not None and len(value.shape) >= 4 else 0
    except Exception:  # noqa: BLE001
        return 0


def _scene_rule(count, seconds):
    """The only rule the node adds: how many scenes, how long, and that they continue each other."""
    length = f"{seconds:g}-second"
    if count == 1:
        return (f"\n\nOUTPUT: return JSON {{\"scenes\": [\"...\"]}} with exactly 1 string: the complete prompt for one "
                f"{length} clip.")
    return (f"\n\nOUTPUT: return JSON {{\"scenes\": [...]}} with exactly {count} strings, one complete prompt per "
            f"{length} clip, in order. The {count} scenes are ONE continuous story or piece: scene 2 begins exactly "
            f"where scene 1 ends, and so on; nothing restarts between scenes.")


def _schema(count):
    return {"type": "object", "additionalProperties": False, "required": ["scenes"],
            "properties": {"scenes": {"type": "array", "minItems": count, "maxItems": count,
                                      "items": {"type": "string"}}}}


def _scenes_from(content, count):
    """Read the scenes out of whatever came back: the JSON asked for, a fenced or loose JSON, or plain text."""
    if isinstance(content, list):
        content = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    text = re.sub(r"^```[\w-]*\s*\n?|\n?\s*```$", "", str(content or "").strip()).strip()
    data = None
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1] if "{" in text else ""):
        try:
            data = json.loads(candidate)
            break
        except (ValueError, TypeError):
            continue
    scenes = None
    if isinstance(data, dict):
        for key in ("scenes", "scene_prompts", "prompts"):
            if isinstance(data.get(key), list):
                scenes = data[key]
                break
        if scenes is None and isinstance(data.get("prompt"), str):
            scenes = [data["prompt"]]
    elif isinstance(data, list):
        scenes = data
    if scenes is None and text:  # plain text: 'Scene 2' headings or '---' separators, else one scene
        parts = [p for p in re.split(r"(?im)^\s*(?:-{3,}|#*\s*scene\s+\d+\s*[:.)-]?)\s*$", text) if p.strip()]
        scenes = parts or [text]
    scenes = [s if isinstance(s, str) else json.dumps(s, ensure_ascii=False) for s in (scenes or [])]
    return [s.strip() for s in scenes if str(s).strip()]


class H3OpenDirector(io.ComfyNode):
    """Your system prompt, your request, references and an optional video: the LLM writes the scenes."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3OpenDirector",
            display_name="H3 Open Director",
            category="text/minimax_h3",
            search_aliases=["open director", "free director", "h3 director system prompt", "h3 scene writer"],
            description=("No modes and no house rules: your system prompt decides everything. Sends the system "
                         "prompt, the request, up to four reference images (Picture 1-4 by socket) and ten frames "
                         "of an optional video to the LLM. With 2+ scenes they continue one story. Output plugs "
                         "into Simple H3 Chain like the H3 Compact Multimodal Edit Director."),
            inputs=[
                io.Custom("LLMMODEL").Input("llm", tooltip="H3 OpenRouter Model, LLM Model (API), Ollama or Qwen local."),
                io.String.Input("system_prompt", multiline=True, default=DEFAULT_SYSTEM_PROMPT,
                                tooltip="Your instructions. Everything (format, style, rules) comes from here."),
                io.String.Input("request", multiline=True, default="", tooltip="What the user wants."),
                io.Int.Input("scene_count", default=1, min=1, max=12,
                             tooltip="With 2 or more, the scenes are one continuous story."),
                io.Float.Input("seconds_per_scene", default=10.0, min=1.0, max=30.0, step=0.5),
                io.Image.Input("reference_image_1", optional=True),
                io.Image.Input("reference_image_2", optional=True),
                io.Image.Input("reference_image_3", optional=True),
                io.Image.Input("reference_image_4", optional=True),
                io.Image.Input("video", optional=True, tooltip="A video's frames: 10 are taken, spread over the clip."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=1.2, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=0x7FFFFFFF, control_after_generate=True),
                io.Boolean.Input("hold", default=False,
                                 tooltip="Reuse the last good result of this node without calling the LLM."),
                io.Boolean.Input("bypass", default=False,
                                 tooltip="Send the request itself as every scene prompt, without calling the LLM."),
            ],
            outputs=[io.String.Output("scene_prompts", tooltip="JSON for Simple H3 Compact Continuous — Scene Count."),
                     io.String.Output("prompt", tooltip="The scenes as plain text (scene 1 alone when there is one)."),
                     io.String.Output("notes"),
                     io.Int.Output("scene_count")],
            hidden=[io.Hidden.unique_id],
        )

    @staticmethod
    def _content(request, pictures, video):
        from . import story_director as sd
        content = [{"type": "text", "text": "REQUEST:\n" + (request or "(no written request: work from the inputs)")}]
        for number, image in pictures:
            content.append({"type": "text", "text": f"<Picture {number}>:"})
            content.append({"type": "image_url", "image_url": {"url": sd._image_data_url(image[:1], 1024)}})
        count = _frames(video)
        if count:
            amount = min(VIDEO_FRAMES, count)
            picks = sorted({round(i * (count - 1) / max(1, amount - 1)) for i in range(amount)})
            content.append({"type": "text", "text": f"VIDEO: {len(picks)} frames spread evenly over the clip, in order."})
            for order, index in enumerate(picks, 1):
                at = round(100 * index / max(1, count - 1))
                content.append({"type": "text", "text": f"frame {order}/{len(picks)}, at {at}% of the clip:"})
                content.append({"type": "image_url", "image_url": {"url": sd._image_data_url(video[index:index + 1], 768)}})
        return content

    @staticmethod
    def _hold(unique_id, value=None):
        if unique_id in (None, ""):
            return None
        key = f"H3OpenDirector:{unique_id}"
        try:
            from . import story_director as sd
            if value is not None:
                sd._set_held_director_plan(key, value)
                return value
            held = sd._get_held_director_plan(key)
            return held if isinstance(held, dict) and isinstance(held.get("scenes"), list) and held["scenes"] else None
        except Exception as error:  # noqa: BLE001
            print(f"[H3 Open Director] Hold unavailable: {error}")
            return None

    @staticmethod
    def _out(scenes, notes, count):
        plain = scenes[0] if len(scenes) == 1 else "\n\n".join(f"Scene {i}:\n{s}" for i, s in enumerate(scenes, 1))
        return io.NodeOutput(json.dumps({"scene_prompts": scenes}, ensure_ascii=False), plain, notes, count)

    @classmethod
    def execute(cls, llm=None, system_prompt=DEFAULT_SYSTEM_PROMPT, request="", scene_count=1, seconds_per_scene=10.0,
                reference_image_1=None, reference_image_2=None, reference_image_3=None, reference_image_4=None,
                video=None, temperature=0.7, seed=0, hold=False, bypass=False, unique_id=None):
        request = str(request or "").strip()
        count = max(1, min(12, int(scene_count or 1)))
        seconds = max(1.0, float(seconds_per_scene or 10.0))
        if unique_id is None:
            unique_id = getattr(getattr(cls, "hidden", None), "unique_id", None)
        fallback = [request or "A cinematic scene."] * count
        if bypass:
            return cls._out(fallback, "Bypass: the request is every scene prompt · 0 calls", count)
        notes = []
        if hold:
            held = cls._hold(unique_id)
            if held and len(held["scenes"]) == count:
                return cls._out(held["scenes"], "Hold: last good result reused · 0 calls", count)
            notes.append("Hold had no saved result for this scene count: a new one was written and saved"
                         if unique_id not in (None, "") else "Hold unavailable (no node id): a new result was written")
        pictures = [(n, image) for n, image in enumerate(
            (reference_image_1, reference_image_2, reference_image_3, reference_image_4), 1) if _frames(image)]
        system = (str(system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT) + _scene_rule(count, seconds)
        tokens = min(16000, count * (900 + int(60 * seconds)) + 400)
        calls = 0
        try:
            if llm is None:
                raise ValueError("connect an LLM model")
            from . import story_director as sd
            payload = {"messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": cls._content(request, pictures, video)}],
                       "max_tokens": tokens, "temperature": float(temperature), "seed": int(seed),
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": "h3_open_director", "strict": True, "schema": _schema(count)}}}
            scenes, last_error = [], None
            for attempt in range(2):
                try:
                    calls += 1
                    result = sd._external_llm_request(llm, payload)
                    scenes = _scenes_from(sd._completion_content(result, payload, llm, "H3 Open Director"), count)
                    if scenes:
                        break
                except Exception as error:  # noqa: BLE001
                    last_error = error
                    if re.search(r"api key|HTTP 40[13]|Connect ", str(error), re.IGNORECASE):
                        raise
                    if attempt == 0 and "response_format" in payload and "Qwen" not in type(llm).__name__:
                        notes.append(f"JSON answer refused ({type(error).__name__}), asked again as plain text")
                        payload = {k: v for k, v in payload.items() if k != "response_format"}
            if not scenes:
                raise last_error or RuntimeError("the LLM returned no scenes")
            if len(scenes) != count:
                notes.append(f"the LLM wrote {len(scenes)} scene(s) for {count}: "
                             + ("extra scenes dropped" if len(scenes) > count else "the last one repeats to fill"))
                scenes = (scenes + [scenes[-1]] * count)[:count]
            cls._hold(unique_id, {"scenes": scenes})
            notes.insert(0, f"{count} scene(s) · {len(pictures)} picture(s)"
                         + (f", {min(VIDEO_FRAMES, _frames(video))} video frames" if _frames(video) else "")
                         + f" · {calls} call(s)")
            return cls._out(scenes, " · ".join(notes), count)
        except Exception as error:  # noqa: BLE001 - the node never fails
            notes.insert(0, f"request passed through as the scene prompt ({type(error).__name__}: {str(error)[:200]})")
            return cls._out(fallback, " · ".join(notes), count)
