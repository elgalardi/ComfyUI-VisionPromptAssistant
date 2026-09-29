"""CPU-only tests for sheet compilation, routing and input isolation."""
import ast
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


class Frames:
    def __init__(self, count=1):
        self.shape = (count, 64, 64, 3)

    def __getitem__(self, key):
        return self


def load_sheet():
    source = Path(__file__).resolve().parents[1] / "visual_sheet_director.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [item for item in tree.body if not isinstance(item, (ast.Import, ast.ImportFrom))]
    calls = []
    held = {}
    def respond(llm, payload):
        calls.append(payload)
        if payload["response_format"]["json_schema"]["name"] == "qwen_visual_director":
            return {"choices": [{"message": {"content": json.dumps({
                "rewritten_prompt": "Create the requested image, preserving reference identity.",
                "wh_ratio": "16:9", "ratio_follow": "", "warnings": [],
            })}}]}
        if payload["response_format"]["json_schema"]["name"] == "visual_sheet_edit":
            return {"choices": [{"message": {"content": json.dumps({
                "subject_definitions": "<Subject 1> is the person in <Picture 1>.",
                "retention_analysis": "Keep identity and layout; change jacket only.",
                "edit_description": "Give <Subject 1> a blue jacket in <Picture 1>.",
                "warnings": [],
            })}}]}
        count = payload["response_format"]["json_schema"]["schema"]["properties"]["panels"]["minItems"]
        plan = {"visual_bible": "One character with curly hair, red jacket, consistent identity.",
                "subject_definitions": "<Subject 1> is an original curly-haired character in a red jacket.",
                "retention_analysis": "<Subject 1>: preserve identity and red jacket across panels.",
                "panels": [{"visual": f"<Subject 1> performs action {i}.", "camera": "Low-angle medium shot.",
                            "annotation": f"Exact label {i}"} for i in range(count)], "warnings": []}
        return {"choices": [{"message": {"content": json.dumps(plan)}, "finish_reason": "stop"}]}
    scope = dict(json=json, math=math, re=re, _external_llm_request=respond,
                 _get_held_director_plan=lambda key: held.get(key),
                 _set_held_director_plan=lambda key, value: held.update({key: value}),
                 _image_data_url=lambda *args: "data:image/jpeg;base64,TEST",
                 io=SimpleNamespace(ComfyNode=object, NodeOutput=lambda *v, **kw: v),
                 ui=SimpleNamespace(PreviewText=lambda value: value))
    exec(compile(tree, str(source), "exec"), scope)
    return scope, calls


class VisualSheetTests(unittest.TestCase):
    def test_qwen_modes_and_hold_target_isolation(self):
        for mode in ("Storyboard", "Character Sheet", "Custom Sheet", "Edit"):
            scope, calls = load_sheet()
            director = scope["VisualSheetDirector"]
            result = director.execute(SimpleNamespace(model="test"), request="test", sheet_type=mode,
                                      reference_image_1=Frames(), target_model="Qwen", unique_id="q")
            self.assertNotIn("<Picture", result[0])
            self.assertEqual(len(calls), 1)
            self.assertIn("MODE:", calls[0]["messages"][0]["content"])
            self.assertIn("the input image", str(calls[0]["messages"][1]["content"]))
            self.assertIn("wh_ratio", json.loads(result[1]))
            with self.assertRaisesRegex(ValueError, "Target model changed"):
                director.execute(None, hold_prompt=True, unique_id="q", target_model="MiniMax H3")

    def test_qwen_multiple_refs_and_invalid_sizing(self):
        scope, calls = load_sheet()
        scope["VisualSheetDirector"].execute(SimpleNamespace(model="test"), request="test", target_model="Qwen",
            reference_image_2=Frames(), reference_image_4=Frames())
        content = str(calls[0]["messages"][1]["content"])
        self.assertIn("<image1> (reference_image_2)", content)
        self.assertIn("<image2> (reference_image_4)", content)
        for prompt, ratio, follow in (("Use <image3>", "", ""), ("Edit", "16:9", "<image1>")):
            response = {"choices": [{"message": {"content": json.dumps({
                "rewritten_prompt": prompt, "wh_ratio": ratio, "ratio_follow": follow, "warnings": []})}}]}
            with self.assertRaises(ValueError):
                scope["render_qwen"](response, 2)

    def test_edit_missing_source_tag_is_added_without_retry(self):
        scope, calls = load_sheet()
        plan = {"subject_definitions": "The woman in the source image.",
                "retention_analysis": "Preserve everything except jacket color.",
                "edit_description": "Make the jacket blue.", "warnings": []}
        response = {"choices": [{"message": {"content": json.dumps(plan)}}]}
        prompt, _ = scope["render_edit"](response, 1)
        self.assertIn("Edit <Picture 1>", prompt)
        self.assertEqual(calls, [])

    def test_edit_normalizes_tags_but_preserves_quoted_lettering(self):
        scope, _ = load_sheet()
        plan = {"subject_definitions": "subject 1 is the woman in <picture 1>.",
                "retention_analysis": "Keep Subject 1 unchanged.",
                "edit_description": 'Add the exact label "Picture 1" beside <subject 1>.', "warnings": []}
        response = {"choices": [{"message": {"content": json.dumps(plan)}}]}
        prompt, _ = scope["render_edit"](response, 1)
        self.assertIn("<Subject 1> is the woman in <Picture 1>", prompt)
        self.assertIn('"Picture 1"', prompt)
        plan["edit_description"] = "Copy outfit from picture 2."
        response["choices"][0]["message"]["content"] = json.dumps(plan)
        with self.assertRaisesRegex(ValueError, "not connected"):
            scope["render_edit"](response, 1)

    def test_edit_preserves_source_contract_and_ignores_layout_controls(self):
        scope, calls = load_sheet()
        result = scope["VisualSheetDirector"].execute(SimpleNamespace(model="test"),
            request="Change jacket to blue", sheet_type="Edit", reference_image_1=Frames(),
            scene_count=24, annotations="production_notes", layout="vertical")
        self.assertIn("Edit <Picture 1>", result[0])
        self.assertNotIn("PANEL 1", result[0])
        self.assertNotIn("caption band", result[0])
        brief = json.loads(calls[0]["messages"][1]["content"][0]["text"])
        self.assertNotIn("panel_count", brief)
        self.assertNotIn("annotations", brief)
        self.assertEqual(len(calls), 1)

    def test_edit_requires_source_slot_one_and_request(self):
        scope, calls = load_sheet()
        for args in ({"request": "edit", "reference_image_2": Frames()},
                     {"request": "", "reference_image_1": Frames()},
                     {"request": "edit"}):
            with self.assertRaisesRegex(ValueError, "Edit requires"):
                scope["VisualSheetDirector"].execute(SimpleNamespace(model="test"), sheet_type="Edit", **args)
        self.assertEqual(calls, [])

    def test_hold_reuses_and_isolates_nodes_without_calls(self):
        scope, calls = load_sheet()
        director = scope["VisualSheetDirector"]
        with self.assertRaisesRegex(ValueError, "No saved sheet"):
            director.execute(None, hold_prompt=True, unique_id="a")
        first = director.execute(SimpleNamespace(model="test"), request="city", unique_id="a")
        second = director.execute(None, request="changed", hold_prompt=True, unique_id="a")
        self.assertEqual(first[:2], second[:2])
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            director.execute(None, hold_prompt=True, unique_id="b")

    def test_text_only_types_counts_and_no_visible_annotations(self):
        for kind in ("Storyboard", "Character Sheet", "Custom Sheet"):
            for count in (1, 6, 9, 24):
                scope, calls = load_sheet()
                result = scope["VisualSheetDirector"].execute(SimpleNamespace(model="test"),
                    request="A character explores a city", sheet_type=kind, scene_count=count)
                self.assertEqual(len(calls), 1)
                self.assertIn(f"exactly {count} distinct panels", result[0])
                self.assertEqual(len(json.loads(result[1])["panels"]), count)
                self.assertNotIn("EXACT VISIBLE ANNOTATION", result[0])
                self.assertTrue(all(p["annotation"] == "" for p in json.loads(result[1])["panels"]))

    def test_four_images_video_controls_and_seed(self):
        scope, calls = load_sheet()
        result = scope["VisualSheetDirector"].execute(SimpleNamespace(model="vision"),
            reference_image_1=Frames(), reference_image_2=Frames(), reference_image_3=Frames(),
            reference_image_4=Frames(), reference_video=Frames(40), video_samples="5",
            direction_context="Genre: Fantasy", seed=123, annotations="production_notes")
        payload = calls[0]
        self.assertEqual(payload["seed"], 123)
        content = payload["messages"][1]["content"]
        self.assertEqual(sum(v["type"] == "image_url" for v in content), 9)
        self.assertIn("Fantasy", content[0]["text"])
        self.assertIn("frame 40/40", str(content))
        self.assertIn("Experimental video", result[2])
        self.assertIn("Dense generated lettering", result[2])

    def test_sparse_slots_and_one_frame_video(self):
        scope, calls = load_sheet()
        result = scope["VisualSheetDirector"].execute(SimpleNamespace(model="test"),
            reference_image_4=Frames(), reference_video=Frames())
        self.assertIn("<Picture 1> (reference_image_4)", result[0])
        self.assertIn("1 sampled frames", result[2])

    def test_missing_input_or_provider_no_calls(self):
        scope, calls = load_sheet()
        for kwargs in ({"llm": None, "request": "test"}, {"llm": SimpleNamespace(model="test")}):
            with self.assertRaises(ValueError):
                scope["VisualSheetDirector"].execute(**kwargs)
        self.assertEqual(calls, [])

    def test_layout_and_annotations(self):
        for layout, expected in (("grid", "2 row(s), up to 3"), ("horizontal", "1 row(s), up to 5"), ("vertical", "5 row(s), up to 1")):
            scope, _ = load_sheet()
            result = scope["VisualSheetDirector"].execute(SimpleNamespace(model="test"), request="test",
                scene_count=5, layout=layout, annotations="brief_labels")
            self.assertIn(expected, result[0])
            self.assertIn('EXACT VISIBLE ANNOTATION: "Exact label 0"', result[0])

    def test_truncated_invalid_json_and_count(self):
        scope, _ = load_sheet()
        for choice in ({"finish_reason": "length", "message": {"content": "{}"}},
                       {"message": {"content": "not json"}},
                       {"message": {"content": json.dumps({"visual_bible": "Test", "panels": [], "warnings": []})}}):
            with self.assertRaises(ValueError):
                scope["render_sheet"]({"choices": [choice]}, 3, "Storyboard", "grid", "panels_only", "16:9", [])


if __name__ == "__main__":
    unittest.main()
