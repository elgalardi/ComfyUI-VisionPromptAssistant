"""Storyboard routing contracts without models, network calls or generation."""
import json
from types import SimpleNamespace
import unittest
from test_compact_provider_routing import Frame, RoutingTests, load_director


class StoryboardTests(unittest.TestCase):
    def setup_director(self, count=3, warnings=None):
        scope, calls = load_director()
        args = RoutingTests().kwargs()
        args.update(edit_mode="Storyboard", edit_request="", reference_image_1=None,
                    storyboard_image=Frame(), continuous_scene_count=count,
                    llm=SimpleNamespace(model="vision-test"))
        def reply(provider, payload):
            calls.append(payload)
            return {"choices": [{"message": {"content": json.dumps({
                "scene_prompts": [{"shots": [
                    {"start_seconds": 0, "description": f"The hero enters location {i}.", "panel_id": i * 2 + 1},
                    {"start_seconds": 2.5, "description": "The camera cuts to a close-up.", "panel_id": i * 2 + 2},
                ], "overall_soundscape": "Footsteps.", "non_diegetic_music": "N/A"} for i in range(count)],
                "panels": [{"action": "Walk.", "camera": "Medium tracking shot.",
                            "audio_text": "Hola." if i % 2 else "", "speaker": "Hero", "language": "Spanish"}
                           for i in range(count * 2)],
                "planning_warnings": warnings or [],
            })}}]}
        scope["_external_llm_request"] = reply
        return scope["H3CompactMultimodalEditDirector"], args, calls

    def test_scene_counts_and_separate_planning_image(self):
        for count in (1, 2, 3, 12):
            director, args, calls = self.setup_director(count)
            result = director.execute(**args)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(json.loads(result[0])["scene_prompts"]), count)
            self.assertNotIn("<Picture 1>", result[0])
            self.assertIn("[Shot 2] At 00:02.500,", result[0])
            self.assertIn("overall_soundscape:", result[0])
            system = calls[0]["messages"][0]["content"]
            self.assertIn(f"exactly {count} scene_prompts", system)
            self.assertIn("5 seconds", system)
            self.assertNotIn("ONE-PASS LOGICAL CONTINUITY", system)
            self.assertEqual(sum(x["type"] == "image_url" for x in calls[0]["messages"][1]["content"]), 1)

    def test_missing_sheet_fails_before_call(self):
        director, args, calls = self.setup_director()
        args["storyboard_image"] = None
        with self.assertRaisesRegex(ValueError, "requires storyboard_image"):
            director.execute(**args)
        self.assertEqual(calls, [])

    def test_ledger_preserves_lyrics_and_reports_missing_and_short_timing(self):
        director, _, _ = self.setup_director()
        lyric = "Yo controlo el ritmo, la hago a mi manera"
        panel = {"action": "She dances.", "camera": "Low rear shot.",
                 "audio_text": lyric, "speaker": "Lead", "language": "Spanish"}
        parsed = {"panels": [panel, dict(panel)], "planning_warnings": [],
                  "scene_prompts": [{"shots": [{"panel_id": 1, "start_seconds": 0,
                                                "description": "She dances."}]}]}
        director._prepare_storyboard(parsed, 1.5)
        shot = parsed["scene_prompts"][0]["shots"][0]
        self.assertIn(lyric, shot["dialogue"])
        self.assertNotIn("She dances", shot["dialogue"])
        self.assertIn("Low rear shot", shot["description"])
        self.assertTrue(any("Uncovered storyboard panels: 2" in w for w in parsed["planning_warnings"]))
        self.assertTrue(any("likely too short" in w for w in parsed["planning_warnings"]))

    def test_rejects_unstructured_and_invalid_timeline(self):
        director, _, _ = self.setup_director()
        with self.assertRaisesRegex(ValueError, "structured shots"):
            director._render_storyboard_scene("[Shot 1] all actions merged", 10)
        for times in ((1, 2), (0, 0), (0, 10), (0, -1)):
            scene = {"shots": [{"start_seconds": t, "description": "Action.", "dialogue": ""} for t in times],
                     "overall_soundscape": "", "non_diegetic_music": ""}
            with self.assertRaises(ValueError):
                director._render_storyboard_scene(scene, 10)

    def test_warnings_visible_in_validation(self):
        director, args, calls = self.setup_director(warnings=["Not enough time for all nine panels."])
        result = director.execute(**args)
        self.assertIn("Not enough time", result[1])

    def test_other_modes_ignore_sheet(self):
        scope, calls = load_director()
        args = RoutingTests().kwargs()
        args.update(storyboard_image=Frame(), llm=SimpleNamespace(model="test"))
        scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(sum(x["type"] == "image_url" for x in scope["last_payload"]["messages"][1]["content"]), 1)


if __name__ == "__main__":
    unittest.main()
