"""Offline contract tests: no model loads, HTTP requests or paid generations."""
import ast
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_director():
    tree = ast.parse((ROOT / "story_director.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "H3CompactMultimodalEditDirector")
    calls = []
    scope = dict(
        json=json, math=math, os=os, re=re, DEFAULT_MODEL="internal-test",
        io=SimpleNamespace(ComfyNode=object, NodeOutput=lambda *v, **kw: v),
        ui=SimpleNamespace(PreviewText=lambda v: v),
        MOTION_STYLES={
            "Orbit Camera": "Circle around the main subject on a smooth controlled arc.",
        },
        VISUAL_LOOKS={
            "Cinematic": "Use deliberate feature-film composition and controlled lighting.",
        },
        _get_held_director_plan=lambda key: None,
        _set_held_director_plan=lambda key, record: None,
        _image_data_url=lambda *args: "data:image/png;base64,TEST",
        _credits=lambda *args: calls.append("credits") or "balance",
    )
    def reply(payload):
        if payload["response_format"]["json_schema"]["name"].endswith("scene_prompts"):
            item_schema = payload["response_format"]["json_schema"]["schema"]["properties"]["scene_prompts"]
            count = item_schema["minItems"]
            content = {"scene_prompts": [
                f"The subject from <Picture 1> advances through beat {index + 1}."
                for index in range(count)
            ]}
        else:
            content = dict(edit_type="general_edit",
                           source_roles=["<Picture 1> visual subject"],
                           visual_evidence="blue jacket",
                           edit_prompt="Preserve the blue jacket from <Picture 1>.")
        return {"choices": [{"message": {"content": json.dumps(content)}}], "usage": {}}
    def internal(key, payload, timeout):
        calls.append("internal")
        return reply(payload)
    def external(provider, payload):
        calls.append("external")
        scope["last_payload"] = payload
        return reply(payload)
    scope.update(_openrouter_request=internal, _external_llm_request=external)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "director", "exec"), scope)
    return scope, calls


class Frame:
    shape = (1, 16, 16, 3)
    def __getitem__(self, key):
        return self


class RoutingTests(unittest.TestCase):
    def test_debug_is_optional_and_does_not_add_calls(self):
        for enabled in (False, True):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(llm=SimpleNamespace(model="local-test"), debug_request=enabled)
            result = scope["H3CompactMultimodalEditDirector"].execute(**args)
            self.assertEqual(calls, ["external"])
            self.assertEqual(len(result), 6)
            if enabled:
                debug = json.loads(result[4])
                self.assertEqual(debug["images_omitted"], 1)
                self.assertNotIn("base64,TEST", result[4])
                self.assertIn("Test.", result[4])
                self.assertEqual(json.loads(result[5])["edit_prompt"],
                                 "Preserve the blue jacket from <Picture 1>.")
                self.assertNotIn("[reference generation]", result[5])
            else:
                self.assertEqual(result[4], "")
                self.assertEqual(result[5], "")

    def test_debug_redacts_native_images_and_credentials(self):
        scope, _ = load_director()
        body = {"messages": [{"content": "key: secret-value", "images": ["PRIVATE1", "PRIVATE2"]}],
                "authorization": "PRIVATE", "keep_alive": 0, "options": {"temperature": 0.2}}
        provider = SimpleNamespace(api_key="secret-value", build_request_payload=lambda p: body)
        result = scope["H3CompactMultimodalEditDirector"]._debug_request(provider, {})
        debug = json.loads(result)
        self.assertEqual(debug["request_kind"], "provider request body")
        self.assertEqual(debug["images_omitted"], 2)
        self.assertEqual(debug["body"]["keep_alive"], 0)
        self.assertNotIn("PRIVATE", result)
        self.assertNotIn("secret-value", result)
        self.assertEqual(body["messages"][0]["images"][0], "PRIVATE1")

    def kwargs(self):
        return dict(edit_request="Use <Picture 1>.",
                    edit_mode="compact", video_samples="3", system_prompt="Test.",
                    max_tokens=1200, temperature=0.2, reasoning=False, seed=0,
                    image_max_dimension=1024, timeout_seconds=30,
                    reference_image_1=Frame(), continuous_scene_count=2,
                    seconds_per_scene=5.0)

    def test_external_does_not_require_or_call_internal_openrouter(self):
        for mode in ("compact", "deep_edit", "continuous"):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, llm=SimpleNamespace(model="local-test"))
            result = scope["H3CompactMultimodalEditDirector"].execute(**args)
            self.assertEqual(calls, ["external"])
            self.assertEqual(result[3], 2)
            if mode == "continuous":
                self.assertEqual(len(json.loads(result[0])["scene_prompts"]), 2)

    def test_disconnected_fails_without_any_api_call(self):
        scope, calls = load_director()
        args = self.kwargs()
        with self.assertRaisesRegex(ValueError, "Connect an external provider"):
            scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(calls, [])

    def test_bypass_is_verbatim_and_needs_no_sources_or_provider(self):
        scope, calls = load_director()
        args = self.kwargs()
        original = "  My manual <Picture 1> prompt.\n\n"
        args.update(edit_request=original, reference_image_1=None, bypass=True,
                    hold_prompt=True, direction_context="Genre: Thriller")
        for mode in ("compact", "deep_edit", "continuous"):
            args["edit_mode"] = mode
            result = scope["H3CompactMultimodalEditDirector"].execute(**args)
            self.assertEqual(result[0], original)
            self.assertEqual(result[3], 2)
        self.assertEqual(calls, [])

    def test_direction_and_specificity_reach_each_director_mode(self):
        for mode in ("compact", "deep_edit", "continuous"):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, llm=SimpleNamespace(model="local-test"),
                        direction_context="Motion style: Orbit Camera")
            scope["H3CompactMultimodalEditDirector"].execute(**args)
            system = scope["last_payload"]["messages"][0]["content"]
            self.assertIn("CONCISE VISUAL SPECIFICITY", system)
            self.assertIn("requirements, not optional suggestions", system)
            self.assertIn("Motion style: Orbit Camera", system)
            self.assertIn("background parallax", system)
            self.assertEqual(calls, ["external"])

    def test_elaborate_mode_adds_rich_single_prompt_contract(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(
            edit_mode="Elaborate", llm=SimpleNamespace(model="local-test"),
            direction_context="Genre: Gothic Horror\nVisual look: Cinematic",
        )
        result = scope["H3CompactMultimodalEditDirector"].execute(**args)
        cls = scope["H3CompactMultimodalEditDirector"]
        payload = scope["last_payload"]
        system = payload["messages"][0]["content"]
        self.assertIn(cls.DEEP_EDIT_RULES, system)
        self.assertIn(cls.ELABORATE_RULES, system)
        self.assertIn(cls.H3_NATIVE_PROMPT_RULES, system)
        self.assertIn("5.00 SECONDS PER SCENE", system)
        self.assertIn("CREATIVE DIRECTOR MANDATE", system)
        self.assertIn("compact creative blueprint", system)
        self.assertIn("Creative objective", system)
        self.assertIn("Cinematic strategy", system)
        self.assertIn("Payoff", system)
        self.assertIn("DIRECT VISUAL PROSE IS MANDATORY", system)
        self.assertIn("Create a scene", system)
        self.assertIn("chronological micro-beats", system)
        self.assertIn("Genre: Gothic Horror", system)
        self.assertEqual(
            payload["response_format"]["json_schema"]["name"],
            "h3_compact_multimodal_edit_prompt",
        )
        self.assertGreaterEqual(payload["max_tokens"], 500)
        prompt_schema = payload["response_format"]["json_schema"]["schema"]["properties"]["edit_prompt"]
        self.assertGreaterEqual(prompt_schema["minLength"], 450)
        self.assertGreaterEqual(prompt_schema["maxLength"], 7000)
        self.assertTrue(result[0].startswith("[reference generation]"))
        self.assertTrue(result[1].startswith("Elaborate "))
        self.assertEqual(calls, ["external"])

    def test_continuous_elaborate_uses_continuous_edit_schema_and_rich_rules(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(
            edit_mode="Continuous Elaborate",
            llm=SimpleNamespace(model="local-test"),
            direction_context="Genre: Gothic Horror",
        )
        result = scope["H3CompactMultimodalEditDirector"].execute(**args)
        cls = scope["H3CompactMultimodalEditDirector"]
        payload = scope["last_payload"]
        system = payload["messages"][0]["content"]
        self.assertIn(cls.DEEP_EDIT_RULES, system)
        self.assertIn(cls.CONTINUOUS_EDIT_RULES, system)
        self.assertIn(cls.CONTINUOUS_ELABORATE_RULES, system)
        self.assertIn(cls.H3_NATIVE_PROMPT_RULES, system)
        self.assertIn(cls.CONTINUOUS_H3_HANDOFF_RULES, system)
        self.assertIn("no generated last-frame inspection", system)
        self.assertIn("DIRECT VISUAL PROSE IS MANDATORY IN EVERY SCENE", system)
        self.assertEqual(
            payload["response_format"]["json_schema"]["name"],
            "h3_compact_continuous_scene_prompts",
        )
        self.assertGreaterEqual(payload["max_tokens"], 2 * 500)
        self.assertEqual(len(json.loads(result[0])["scene_prompts"]), 2)
        self.assertTrue(result[1].startswith("Continuous Elaborate "))
        self.assertEqual(calls, ["external"])

    def test_elaborate_seconds_control_density_and_validation(self):
        for seconds, expected_chars, expected_tokens in (
            (4.0, 450, 500), (8.0, 700, 700), (15.0, 900, 900),
        ):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode="Continuous Elaborate", seconds_per_scene=seconds,
                        max_tokens=256, llm=SimpleNamespace(model="local-test"))
            scope["H3CompactMultimodalEditDirector"].execute(**args)
            payload = scope["last_payload"]
            system = payload["messages"][0]["content"]
            item = payload["response_format"]["json_schema"]["schema"]["properties"]["scene_prompts"]["items"]
            self.assertIn(f"{seconds:.2f} SECONDS PER SCENE", system)
            self.assertEqual(item["minLength"], expected_chars)
            self.assertEqual(payload["max_tokens"], 2 * expected_tokens)
            self.assertEqual(calls, ["external"])

        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode="Elaborate", seconds_per_scene=16.0,
                    llm=SimpleNamespace(model="local-test"))
        with self.assertRaisesRegex(ValueError, "between 1 and 15"):
            scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(calls, [])

        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode="Elaborate", seconds_per_scene="not-a-number",
                    llm=SimpleNamespace(model="local-test"))
        with self.assertRaisesRegex(ValueError, "must be a number"):
            scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(calls, [])

    def test_i2v_mode_treats_picture_one_as_exact_opening_frame(self):
        for mode in ("compact", "Elaborate", "Continuous Edit", "Continuous Elaborate"):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(
                edit_mode=mode, i2v_mode=True,
                llm=SimpleNamespace(model="local-test"),
            )
            result = scope["H3CompactMultimodalEditDirector"].execute(**args)
            cls = scope["H3CompactMultimodalEditDirector"]
            system = scope["last_payload"]["messages"][0]["content"]
            self.assertIn(cls.I2V_RULES, system)
            self.assertIn("exact visible opening frame", system)
            self.assertIn("I2V on", result[1])
            self.assertEqual(calls, ["external"])

    def test_i2v_mode_requires_picture_one(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(
            reference_image_1=None, reference_image_2=Frame(), i2v_mode=True,
            llm=SimpleNamespace(model="local-test"),
        )
        with self.assertRaisesRegex(ValueError, "reference_image_1"):
            scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(calls, [])

    def test_legacy_zero_scene_count_is_normalized_to_one(self):
        for mode in ("compact", "Elaborate", "Enhance", "Continuous Edit", "Continuous Elaborate"):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(
                edit_mode=mode, continuous_scene_count=0,
                llm=SimpleNamespace(model="local-test"),
            )
            result = scope["H3CompactMultimodalEditDirector"].execute(**args)
            self.assertEqual(result[3], 1)
            self.assertEqual(calls, ["external"])

    def test_elaborate_orbit_requires_explicit_camera_geometry(self):
        for mode in ("Elaborate", "Continuous Elaborate"):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(
                edit_mode=mode, llm=SimpleNamespace(model="local-test"),
                direction_context=(
                    "Genre: Cinematic Drama\n"
                    "Motion style: Orbit Camera\n"
                    "Visual look: Cinematic"
                ),
            )
            scope["H3CompactMultimodalEditDirector"].execute(**args)
            cls = scope["H3CompactMultimodalEditDirector"]
            system = scope["last_payload"]["messages"][0]["content"]
            self.assertIn(cls.ELABORATE_DIRECTION_RULES, system)
            self.assertIn(cls.ELABORATE_ORBIT_RULES, system)
            self.assertIn("clockwise or counterclockwise", system)
            self.assertIn("approximate arc in degrees", system)
            self.assertIn("constant radius", system)
            self.assertIn("exact viewpoint reached by the prior scene", system)
            self.assertEqual(calls, ["external"])

    def test_elaborate_modes_can_invent_from_direction_controls_without_prompt(self):
        for mode in ("Elaborate", "Continuous Elaborate"):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(
                edit_request="", edit_mode=mode,
                llm=SimpleNamespace(model="local-test"),
                direction_context=(
                    "Genre: Product Showcase\n"
                    "Motion style: Orbit Camera\n"
                    "Visual look: Cinematic"
                ),
            )
            scope["H3CompactMultimodalEditDirector"].execute(**args)
            payload = scope["last_payload"]
            system = payload["messages"][0]["content"]
            user_text = payload["messages"][1]["content"][0]["text"]
            self.assertIn("CONTROLS-ONLY ORIGINAL DIRECTION", system)
            self.assertIn("RESOLVED CONTROL SPECIFICATIONS", system)
            self.assertIn("Circle around the main subject", system)
            self.assertIn("feature-film composition", system)
            self.assertIn("No written premise was supplied", user_text)
            self.assertEqual(calls, ["external"])

    def test_empty_prompt_without_creative_controls_still_fails(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(
            edit_request="", edit_mode="Elaborate",
            llm=SimpleNamespace(model="local-test"), direction_context="",
        )
        with self.assertRaisesRegex(ValueError, "requires an edit_request"):
            scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(calls, [])

    def test_truncated_json_is_repaired_once_without_resending_images(self):
        scope, calls = load_director()
        attempts = []
        valid = {
            "edit_type": "general_edit",
            "source_roles": ["<Picture 1> visual subject"],
            "visual_evidence": "blue jacket and studio light",
            "edit_prompt": (
                "The subject from <Picture 1> moves through the illuminated studio while "
                "the blue jacket remains stable and the camera follows the action."
            ),
        }

        def flaky(provider, payload):
            attempts.append(payload)
            calls.append("external")
            if len(attempts) == 1:
                content = '{"edit_type":"general_edit","source_roles":["<Picture 1> visual subject"],"visual_evidence":"blue jacket","edit_prompt":"The subject'
            else:
                content = json.dumps(valid)
            return {"choices": [{"message": {"content": content}}], "usage": {}}

        scope["_external_llm_request"] = flaky
        args = self.kwargs()
        args.update(llm=SimpleNamespace(model="local-test"), debug_request=True)
        result = scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(calls, ["external", "external"])
        self.assertIn("JSON repaired after one retry", result[1])
        self.assertIn("ATTEMPT 1", result[5])
        self.assertIn("ATTEMPT 2", result[5])
        repair_content = attempts[1]["messages"][1]["content"]
        self.assertIsInstance(repair_content, str)
        self.assertNotIn("image_url", json.dumps(attempts[1]))

    def test_fenced_json_parses_without_retry(self):
        scope, calls = load_director()
        payload_content = {
            "edit_type": "general_edit",
            "source_roles": ["<Picture 1> visual subject"],
            "visual_evidence": "blue jacket and studio light",
            "edit_prompt": "The subject from <Picture 1> walks through the blue-lit studio.",
        }
        scope["_external_llm_request"] = lambda provider, payload: (
            calls.append("external") or {
                "choices": [{"message": {"content": "```json\n" + json.dumps(payload_content) + "\n```"}}],
                "usage": {},
            }
        )
        args = self.kwargs()
        args.update(llm=SimpleNamespace(model="local-test"))
        scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(calls, ["external"])

    def test_renamed_modes_keep_the_original_behavior(self):
        for label, old in (("edit", "deep_edit"), ("Enhance", "continuous")):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(llm=SimpleNamespace(model="local-test"), edit_mode=old)
            previous = scope["H3CompactMultimodalEditDirector"].execute(**args)
            old_payload = scope["last_payload"]
            args["edit_mode"] = label
            renamed = scope["H3CompactMultimodalEditDirector"].execute(**args)
            self.assertEqual(old_payload, scope["last_payload"])
            self.assertEqual(previous[0], renamed[0])
            self.assertTrue(renamed[1].startswith(label + " "))

    def test_continuous_edit_combines_edit_rules_and_scene_output(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(llm=SimpleNamespace(model="local-test"), edit_mode="Edit Continuo",
                    source_video_1=Frame(), direction_context="Motion style: Orbit Camera")
        result = scope["H3CompactMultimodalEditDirector"].execute(**args)
        cls = scope["H3CompactMultimodalEditDirector"]
        system = scope["last_payload"]["messages"][0]["content"]
        self.assertIn(cls.DEEP_EDIT_RULES, system)
        self.assertIn(cls.CONTINUOUS_EDIT_RULES, system)
        self.assertIn("Test.", system)  # visible system prompt remains active
        scenes = json.loads(result[0])["scene_prompts"]
        self.assertEqual(len(scenes), 2)
        self.assertTrue(all(s.startswith("[video editing + reference generation]") for s in scenes))
        self.assertTrue(result[1].startswith("Continuous Edit "))
        self.assertEqual(calls, ["external"])

    def test_hold_cannot_reuse_another_provider(self):
        scope, calls = load_director()
        scope["_get_held_director_plan"] = lambda key: {
            "cache_kind": "h3_compact_multimodal_edit", "provider_identity": ""
        }
        args = self.kwargs()
        args.update(hold_prompt=True, llm=SimpleNamespace(model="local-test"))
        with self.assertRaisesRegex(RuntimeError, "provider/model changed"):
            scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(calls, [])

    def test_contextual_reactions_only_apply_to_sequence_modes(self):
        for mode in ("compact", "edit", "Enhance", "Continuous Edit"):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(llm=SimpleNamespace(model="local-test"), edit_mode=mode)
            scope["H3CompactMultimodalEditDirector"].execute(**args)
            rules = scope["H3CompactMultimodalEditDirector"].CONTEXTUAL_REACTION_RULES
            system = scope["last_payload"]["messages"][0]["content"]
            if mode in ("Enhance", "Continuous Edit"):
                self.assertIn(rules, system)
                self.assertIn("change performance only when authorized", system)
                self.assertIn("do not imply consent or enjoyment", system)
                self.assertIn("write brief actual lines", system)
                self.assertIn("Keep supplied lines verbatim", system)
                self.assertIn("No dialogue", system)
            else:
                self.assertNotIn(rules, system)
            self.assertEqual(calls, ["external"])

    def test_qwen_keeps_video_and_picture_labels_and_image_order(self):
        import base64
        import io
        import numpy as np
        import torch
        from PIL import Image
        tree = ast.parse((ROOT / "compact_llm_providers.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "qwen_messages")
        scope = dict(base64=base64, binary_io=io, json=json,
                     np=np, torch=torch, Image=Image)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "qwen", "exec"), scope)
        content = []
        for label, color in (("<Video 1> SAMPLE 1/1", "red"), ("<Picture 2>", "blue")):
            buffer = io.BytesIO()
            Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            content += [{"type": "text", "text": label},
                        {"type": "image_url", "image_url": {"url": url}}]
        prompt, images = scope["qwen_messages"](
            [{"role": "system", "content": "Test."}, {"role": "user", "content": content}],
            {"type": "object"},
        )
        self.assertLess(prompt.index("<Video 1>"), prompt.index("<Picture 2>"))
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0][0, 0, 0].tolist(), [1, 0, 0])
        self.assertEqual(images[1][0, 0, 0].tolist(), [0, 0, 1])

    def test_genre_fidelity_is_scoped_and_camera_none_stays_free(self):
        for mode in ("Enhance", "Continuous Edit", "compact", "edit"):
            for direction in ("", "Genre: Gothic Horror"):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(llm=SimpleNamespace(model="local-test"), edit_mode=mode,
                            direction_context=direction)
                scope["H3CompactMultimodalEditDirector"].execute(**args)
                system = scope["last_payload"]["messages"][0]["content"]
                sequence = mode in ("Enhance", "Continuous Edit")
                self.assertEqual("PERFORMANCE AND COMPLETION" in system, sequence)
                self.assertEqual("GOTHIC HORROR EXECUTION" in system, sequence and bool(direction))
                if sequence:
                    self.assertIn("unconsciousness for death", system)
                    self.assertIn("avoid repeated descriptions", system)
                    self.assertIn("no particular movement is required without a selection", system)
                if mode == "Enhance":
                    self.assertNotIn("orbit", system.lower())
                    self.assertIn("does not impose its background, pose, expression or camera", system)
                    self.assertLess(len(system.split()), 1100)
                self.assertEqual(calls, ["external"])

    def test_sequence_detail_budget_and_causality_without_temperature_change(self):
        for mode, ceiling, tokens_per_scene in (
            ("Enhance", "up to 260", 400),
            ("Continuous Edit", "maximum of 320", 480),
        ):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(llm=SimpleNamespace(model="local-test"), edit_mode=mode,
                        max_tokens=256, temperature=0.2)
            scope["H3CompactMultimodalEditDirector"].execute(**args)
            payload = scope["last_payload"]
            system = payload["messages"][0]["content"]
            self.assertIn(ceiling, system)
            self.assertIn("causally connected reactions", system)
            self.assertIn("named location", system)
            self.assertIn("distinct beats", system)
            self.assertEqual(payload["temperature"], 0.2)
            self.assertEqual(payload["max_tokens"], 2 * tokens_per_scene)
            self.assertEqual(calls, ["external"])


if __name__ == "__main__":
    unittest.main()
