"""H3 Open Director: your system prompt decides everything.

Same inputs and outputs as the H3 Compact Multimodal Edit Director (minus its mode selector), so it drops into the
same workflows. No modes and no house rules: the node sends your system prompt, the request, the connected
pictures, videos and guides to the LLM and returns what it writes. The one fixed rule is the scene count: with two
or more scenes the LLM is told they are one continuous story, each scene starting where the previous one ends.
Nothing here raises: on any failure the request comes out as the prompt and `validation` says why."""
import hashlib
import json
import re
import time

from comfy_api.latest import io

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
            f"where scene 1 ends, and so on; nothing restarts between scenes. After that opening moment a scene may "
            f"cut to another place or time, or skip ahead, when the request or these instructions ask for it.")


def _schema(count):
    return {"type": "object", "additionalProperties": False, "required": ["scenes"],
            "properties": {"scenes": {"type": "array", "minItems": count, "maxItems": count,
                                      "items": {"type": "string"}}}}


def _scenes_from(content):
    """Read the scenes out of the reply. Returns (scenes, complete). A reply that starts as JSON but does not parse
    (cut off by the token limit, or malformed) is incomplete: it is never passed on as a prompt. Free text that does
    not look like JSON is accepted ('Scene 2' headings or '---' split it)."""
    if isinstance(content, list):
        content = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    text = re.sub(r"^```[\w-]*\s*\n?|\n?\s*```$", "", str(content or "").strip()).strip()
    if not text:
        return [], False
    data, structured = None, text[:1] in "{["
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1] if text.find("{") < text.rfind("}") else ""):
        try:
            data = json.loads(candidate)
            break
        except (ValueError, TypeError):
            continue
    if data is None:
        if structured:
            return [], False  # JSON cut off or broken
        parts = [p.strip() for p in re.split(r"(?im)^\s*(?:-{3,}|#*\s*scene\s+\d+\s*[:.)-]?)\s*$", text) if p.strip()]
        return parts or [text], True
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
    found = []
    for item in scenes or []:
        if isinstance(item, dict):
            item = next((item[k] for k in ("prompt", "scene", "text") if isinstance(item.get(k), str)), None)
        if isinstance(item, str) and item.strip():
            found.append(item.strip())
    return found, bool(found)


_FORMAT_REFUSED = re.compile(r"response_format|json_schema|structured output|schema", re.IGNORECASE)
_TRANSIENT = re.compile(r"timed? ?out|timeout|429|rate.?limit|too many requests|\b5\d\d\b|overloaded|temporar|"
                        r"connection (?:reset|aborted)|remote ?disconnected|unavailable", re.IGNORECASE)
_FATAL = re.compile(r"api key|HTTP 40[13]|unauthori|forbidden|Connect an LLM|Select an OpenRouter model", re.IGNORECASE)
_INCOMPLETE = re.compile(r"truncated|malformed|no complete json|not valid json|invalid json|no usable content|"
                         r"length", re.IGNORECASE)


class H3OpenDirector(io.ComfyNode):
    """Your system prompt, your request and your sources: the LLM writes one scene or a continuous sequence."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3OpenDirector",
            display_name="H3 Open Director",
            category="text/minimax_h3",
            search_aliases=["open director", "free director", "h3 director system prompt", "h3 scene writer"],
            description=("Same inputs and outputs as the H3 Compact Multimodal Edit Director, without modes or house "
                         "rules: your system prompt decides everything. With 2+ scenes they continue one story."),
            inputs=[
                io.String.Input("edit_request", multiline=True, dynamic_prompts=False, default="",
                                tooltip="What the user wants."),
                io.Image.Input("reference_image_1", optional=True),
                io.Image.Input("reference_image_2", optional=True),
                io.Image.Input("reference_image_3", optional=True),
                io.Image.Input("reference_image_4", optional=True),
                io.Image.Input("source_video_1", optional=True),
                io.Image.Input("source_video_2", optional=True),
                io.Image.Input("driven_image", display_name="Guide Image", optional=True),
                io.Image.Input("driven_video", display_name="Guide Video", optional=True),
                io.Combo.Input("video_samples", options=["3", "5", "10"], default="10",
                               tooltip="Frames taken from each connected video, spread over the clip."),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=False, default=DEFAULT_SYSTEM_PROMPT,
                                tooltip="Your instructions. Everything (format, style, rules) comes from here."),
                io.Int.Input("max_tokens", default=1200, min=256, max=16000,
                             tooltip="A minimum: raised automatically when the scenes need more room (usage_stats "
                                     "shows the amount actually requested)."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=1.2, step=0.05),
                io.Boolean.Input("reasoning", default=False),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=True),
                io.Int.Input("image_max_dimension", default=1024, min=512, max=2048, step=64, advanced=True),
                io.Int.Input("timeout_seconds", default=300, min=30, max=900, advanced=True,
                             tooltip="Kept for workflow compatibility; the LLM model node sets the real timeout."),
                io.Boolean.Input("hold_prompt", display_name="Hold Edit Prompt", default=False,
                                 tooltip="Reuse the last good result of this node without calling the LLM."),
                io.Boolean.Input("i2v_mode", display_name="I2V Mode — Picture 1 is First Frame", default=False),
                io.Int.Input("continuous_scene_count", display_name="Scenes — Sequence", default=1, min=1, max=12,
                             tooltip="With 2 or more, the scenes are one continuous story."),
                io.Float.Input("seconds_per_scene", display_name="Seconds per Scene", default=10.0, min=1.0,
                               max=60.0, step=0.5, force_input=True),
                io.String.Input("direction_context", optional=True, force_input=True,
                                tooltip="Optional Direction Controls, passed to the LLM as written."),
                io.Custom("LLMMODEL").Input("llm", optional=True),
                io.Boolean.Input("bypass", default=False,
                                 tooltip="Send the request itself as every scene prompt, without calling the LLM."),
                io.Boolean.Input("debug_request", default=False),
                io.Combo.Input("on_failure", options=["Stop", "Use request"], default="Stop",
                               tooltip="When the LLM cannot write the scenes: Stop the workflow with a clear message "
                                       "(no GPU spent on an undirected video), or use the request itself as the prompt."),
            ],
            outputs=[io.String.Output("edit_prompt"), io.String.Output("validation"), io.String.Output("usage_stats"),
                     io.Int.Output("continuous_scene_count"), io.String.Output("debug_request"),
                     io.String.Output("raw_response")],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @staticmethod
    def _content(request, pictures, videos, guide_image, guide_video, samples, size, i2v, direction):
        from . import story_director as sd
        content = [{"type": "text", "text": "REQUEST:\n" + (request or "(no written request: work from the inputs)")}]
        if direction:  # already expanded with one note per control (direction_notes.expand)
            content.append({"type": "text", "text": direction})
        for number, image in pictures:
            note = " (the first frame of scene 1)" if i2v and number == 1 else ""
            content.append({"type": "text", "text": f"<Picture {number}>{note}:"})
            content.append({"type": "image_url", "image_url": {"url": sd._image_data_url(image[:1], size)}})

        def clip(label, video):
            count = _frames(video)
            amount = min(samples, count)
            picks = sorted({round(i * (count - 1) / max(1, amount - 1)) for i in range(amount)})
            content.append({"type": "text", "text": f"{label}: {len(picks)} frames spread evenly over the clip, in order."})
            for order, index in enumerate(picks, 1):
                at = round(100 * index / max(1, count - 1))
                content.append({"type": "text", "text": f"{label} frame {order}/{len(picks)}, at {at}% of the clip:"})
                content.append({"type": "image_url", "image_url": {"url": sd._image_data_url(video[index:index + 1],
                                                                                             min(size, 768))}})
        for number, video in videos:
            clip(f"<Video {number}>", video)
        if _frames(guide_image):
            content.append({"type": "text", "text": "GUIDE IMAGE:"})
            content.append({"type": "image_url", "image_url": {"url": sd._image_data_url(guide_image[:1], size)}})
        if _frames(guide_video) > 1:
            clip("GUIDE VIDEO", guide_video)
        return content

    @staticmethod
    def _hold_key(unique_id, extra_pnginfo, system, count, seconds):
        if unique_id in (None, ""):
            return None
        workflow = ""
        try:
            workflow = str(((extra_pnginfo or {}).get("workflow") or {}).get("id") or "")
        except Exception:  # noqa: BLE001
            pass
        recipe = hashlib.sha1(f"{system}\x00{count}\x00{seconds:g}".encode("utf-8")).hexdigest()[:12]
        return f"H3OpenDirector:{workflow or 'no-workflow-id'}:{unique_id}:{recipe}"

    @staticmethod
    def _hold(key, value=None):
        if not key:
            return None
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
    def _out(scenes, validation, usage, count, debug="", raw=""):
        prompt = scenes[0] if count == 1 else json.dumps({"scene_prompts": scenes}, ensure_ascii=False)
        return io.NodeOutput(prompt, validation, usage, count, debug, raw)

    @classmethod
    def execute(cls, edit_request="", reference_image_1=None, reference_image_2=None, reference_image_3=None,
                reference_image_4=None, source_video_1=None, source_video_2=None, driven_image=None,
                driven_video=None, video_samples="10", system_prompt=DEFAULT_SYSTEM_PROMPT, max_tokens=1200,
                temperature=0.7, reasoning=False, seed=0, image_max_dimension=1024, timeout_seconds=300,
                hold_prompt=False, i2v_mode=False, continuous_scene_count=1, seconds_per_scene=10.0,
                direction_context=None, llm=None, bypass=False, debug_request=False, on_failure="Stop",
                unique_id=None, extra_pnginfo=None, **_ignored):
        request = str(edit_request or "").strip()
        count = max(1, min(12, int(continuous_scene_count or 1)))
        seconds = max(1.0, float(seconds_per_scene or 10.0))
        hidden = getattr(cls, "hidden", None)
        if unique_id is None:
            unique_id = getattr(hidden, "unique_id", None)
        if extra_pnginfo is None:
            extra_pnginfo = getattr(hidden, "extra_pnginfo", None)
        fallback = [request or "A cinematic scene."] * count
        if bypass:
            return cls._out(fallback, "Bypass: the request is every scene prompt", "0 calls", count)
        notes = []
        user_system = str(system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
        hold_key = cls._hold_key(unique_id, extra_pnginfo, user_system, count, seconds)
        if hold_prompt:
            held = cls._hold(hold_key)
            if held and len(held["scenes"]) == count:
                return cls._out(held["scenes"], "Hold: last good result reused", "0 calls", count)
            notes.append("Hold had no saved result for this workflow, node, system prompt and scene setup: a new one "
                         "was written and saved" if hold_key else "Hold unavailable (no node id): a new result was written")
        pictures = [(n, image) for n, image in enumerate(
            (reference_image_1, reference_image_2, reference_image_3, reference_image_4), 1) if _frames(image)]
        videos = [(n, video) for n, video in enumerate((source_video_1, source_video_2), 1) if _frames(video) > 1]
        try:
            samples = max(1, int(video_samples))
        except (TypeError, ValueError):
            samples = 10
        system = user_system + _scene_rule(count, seconds)
        tokens = max(int(max_tokens or 1200), min(16000, count * (900 + int(60 * seconds)) + 400))
        spent = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        debug, raw = "", ""
        try:
            if llm is None:
                raise ValueError("connect an LLM model")
            from . import story_director as sd
            direction, drawn = "", []
            if str(direction_context or "").strip():
                try:
                    from .direction_notes import expand
                    direction, drawn = expand(direction_context, seed)
                except Exception as error:  # noqa: BLE001 - the plain lines still reach the LLM
                    direction = "DIRECTION CONTROLS:\n" + str(direction_context).strip()
                    notes.append(f"control notes unavailable ({type(error).__name__})")
            if drawn:
                notes.append("random: " + ", ".join(drawn))
            content = cls._content(request, pictures, videos, driven_image, driven_video, samples,
                                   int(image_max_dimension or 1024), bool(i2v_mode and pictures and pictures[0][0] == 1),
                                   direction)
            payload = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                       "max_tokens": tokens, "temperature": float(temperature), "seed": int(seed),
                       "reasoning": {"enabled": bool(reasoning)},
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": "h3_open_director", "strict": True, "schema": _schema(count)}}}
            if debug_request:
                shown = json.loads(json.dumps(payload, default=str))
                for item in shown["messages"][1]["content"]:
                    if item.get("type") == "image_url":
                        item["image_url"]["url"] = "(image omitted)"
                debug = json.dumps(shown, ensure_ascii=False, indent=1)
            scenes, last_error = [], None
            for attempt in range(3):  # first try plus at most two recoveries, each for its own kind of trouble
                try:
                    spent["calls"] += 1
                    result = sd._external_llm_request(llm, payload)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        spent[key] += int(((result or {}).get("usage") or {}).get(key) or 0) if isinstance(result, dict) else 0
                    answer = sd._completion_content(result, payload, llm, "H3 Open Director")
                    raw = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False, default=str)
                    scenes, complete = _scenes_from(answer)
                    if complete:
                        break
                    raise ValueError("the reply was cut off or malformed (no complete JSON)")
                except Exception as error:  # noqa: BLE001
                    last_error, message = error, str(error)
                    if _FATAL.search(message) or attempt == 2:
                        raise
                    plain_ok = "response_format" in payload and "Local Qwen" not in message  # Qwen needs the schema
                    if plain_ok and _FORMAT_REFUSED.search(message) and not _INCOMPLETE.search(message):
                        notes.append("JSON format refused by the provider, asked again as plain text")
                        payload = {k: v for k, v in payload.items() if k != "response_format"}
                    elif _TRANSIENT.search(message):
                        notes.append(f"temporary provider error ({message[:80]}), retried")
                        time.sleep(3)
                    elif _INCOMPLETE.search(message):
                        payload = dict(payload, max_tokens=min(16000, int(payload["max_tokens"] * 1.5)))
                        notes.append(f"incomplete reply, asked again with {payload['max_tokens']} tokens")
                    elif plain_ok:  # unclear provider error: one plain-text try, as before
                        notes.append(f"provider error ({message[:80]}), asked again as plain text")
                        payload = {k: v for k, v in payload.items() if k != "response_format"}
                    else:
                        raise
            if not scenes:
                raise last_error or RuntimeError("the LLM returned no scenes")
            if len(scenes) > count:
                notes.append(f"the LLM wrote {len(scenes)} scenes for {count}: extra scenes dropped")
                scenes = scenes[:count]
            if len(scenes) < count:  # ask only for the missing scenes, continuing the written ones
                written = len(scenes)
                notes.append(f"the LLM wrote {written} of {count} scenes: asked for the rest")
                follow = dict(payload, messages=[
                    {"role": "system", "content": user_system + _scene_rule(count, seconds)
                     + f"\n\nScenes 1 to {written} are already written (below). Write ONLY scenes {written + 1} to "
                       f"{count}: return JSON {{\"scenes\": [...]}} with exactly {count - written} strings."},
                    {"role": "user", "content": list(payload["messages"][1]["content"]) + [
                        {"type": "text", "text": "ALREADY WRITTEN SCENES:\n" + "\n\n".join(
                            f"Scene {i}:\n{text}" for i, text in enumerate(scenes, 1))}]}])
                if "response_format" in follow:
                    follow["response_format"] = {"type": "json_schema", "json_schema": {
                        "name": "h3_open_director", "strict": True, "schema": _schema(count - written)}}
                try:
                    spent["calls"] += 1
                    result = sd._external_llm_request(llm, follow)
                    more, complete = _scenes_from(sd._completion_content(result, follow, llm, "H3 Open Director"))
                    scenes += more[:count - written] if complete else []
                except Exception as error:  # noqa: BLE001
                    notes.append(f"the missing scenes failed ({type(error).__name__})")
                if len(scenes) < count:
                    raise RuntimeError(f"the LLM wrote only {len(scenes)} of {count} scenes")
            if (re.search(r"(?im)^\s*Dialogue:\s*No dialogue\s*$", str(direction_context or ""))
                    and any(re.search(r"<d>|\(S\d\)", text) for text in scenes)):
                notes.append("WARNING: the LLM wrote dialogue although No dialogue was chosen")
            cls._hold(hold_key, {"scenes": scenes})  # only complete, validated results
            notes.insert(0, f"H3 Open Director · {count} scene(s) of {seconds:g}s · {len(pictures)} picture(s)"
                         + (f" · {len(videos)} video(s)" if videos else "")
                         + (" · guide" if _frames(driven_image) or _frames(driven_video) > 1 else ""))
            usage = (f"{spent['calls']} call(s) · {spent['total_tokens']} tokens "
                     f"({spent['prompt_tokens']} in / {spent['completion_tokens']} out) · max_tokens requested "
                     f"{payload['max_tokens']}" + (f" (raised from {int(max_tokens)})" if tokens > int(max_tokens or 0) else ""))
            return cls._out(scenes, " · ".join(notes), usage, count, debug, raw)
        except Exception as error:  # noqa: BLE001
            reason = f"{type(error).__name__}: {str(error)[:300]}"
            if on_failure != "Use request":
                raise RuntimeError("H3 Open Director could not write the scenes, so the workflow stops here instead of "
                                   "generating an undirected video. " + reason + (" | " + " · ".join(notes) if notes else "")
                                   + " (Set on_failure to 'Use request' to continue with the request as the prompt.)")
            notes.insert(0, f"request passed through as the prompt ({reason})")
            return cls._out(fallback, " · ".join(notes), f"{spent['calls']} call(s)", count, debug, raw)
