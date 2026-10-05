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
            self.assertIn("CINEMATIC CRAFT", system)
            self.assertIn("storyboard setups outrank", system)
            self.assertIn("without inventing cuts in a continuous take or dropping required shots/panels", system)
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

    def test_rejects_unstructured_shots(self):
        director, _, _ = self.setup_director()
        with self.assertRaisesRegex(ValueError, "structured shots"):
            director._render_storyboard_scene("[Shot 1] all actions merged", 10)

    def test_repairs_invalid_timestamps_without_dropping_shots(self):
        director, _, _ = self.setup_director()
        for times in ((1, 2), (0, 0), (0, 10), (0, -1), (10, 15),
                      (0, float('nan')), (0, float('inf')), (0, None),
                      (0, '00:04.000'), (0, 0.0001), (0, 9.9999), (0, 4, 2)):
            scene = {"shots": [{"start_seconds": t, "description": "Action.", "dialogue": ""} for t in times],
                     "overall_soundscape": "", "non_diegetic_music": ""}
            rendered = director._render_storyboard_scene(scene, 10)
            self.assertEqual(rendered.count('[Shot '), len(times))
            starts = [s['start_seconds'] for s in scene['shots']]
            self.assertEqual(starts[0], 0)
            self.assertTrue(all(0 <= t < 10 for t in starts))
            self.assertTrue(all(a < b for a, b in zip(starts, starts[1:])))

    def test_valid_timeline_is_preserved(self):
        director, _, _ = self.setup_director()
        scene = {'shots': [{'start_seconds': t} for t in (0, 1.5, 4.999)]}
        self.assertFalse(director._normalize_storyboard_times(scene, 5))
        self.assertEqual([s['start_seconds'] for s in scene['shots']], [0, 1.5, 4.999])

    def test_execute_repairs_each_scene_locally_with_one_llm_call(self):
        director, args, calls = self.setup_director(count=2)
        # The fixture returns a second shot at 2.5s, outside this 1s scene.
        args['seconds_per_scene'] = 1
        result = director.execute(**args)
        prompts = json.loads(result[0])['scene_prompts']
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn('[Shot 2] At 00:00.500,', prompt)
            self.assertIn('Hola.', prompt)
        self.assertIn('automatically redistributed', result[1])

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
