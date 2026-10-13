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
            "Fast Motion": next(ast.literal_eval(node.value)['Fast Motion'] for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == 'MOTION_STYLES' for target in node.targets)),
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
        if payload['response_format']['json_schema']['name'] == 'h3_driven_visual_blueprint':
            content = dict(activity='Cheerleading performance beside a conversation partner.',
                setting='Football sideline', lighting='Daylight', visual_style='Natural sports coverage',
                mood='Energetic', camera='Medium-wide side view', props='Pom-poms',
                roles=[dict(role='cheerleader', wardrobe='Team uniform and trainers', hair='Long blonde hair in a high ponytail',
                    body_proportions='Athletic build with long limbs', pose='Weight on left leg, arms raised with pom-poms',
                    performance='A short cheer with raised pom-poms', spatial_relationships='Beside the partner')])
            return {'choices': [{'message': {'content': json.dumps(content)}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 80}}
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
        properties = payload['response_format']['json_schema']['schema']['properties']
        if 'cut_scene_plan' in properties:
            content['cut_scene_plan'] = [dict(event=f'New activity {index}',
                setting='Courtyard', framing=f'Angle {index}', change_from_previous=f'New result and coverage {index}')
                for index in range(properties['cut_scene_plan']['minItems'])]
        if 'driven_cast_mapping' in properties:
            mapping = properties['driven_cast_mapping']['items']['properties']
            content['driven_cast_mapping'] = [dict(driver_role=role,
                identity_source=mapping['identity_source']['enum'][0])
                for role in mapping['driver_role']['enum']]
        return {"choices": [{"message": {"content": json.dumps(content)}}], "usage": {}}
    def internal(key, payload, timeout):
        calls.append("internal")
        return reply(payload)
    def external(provider, payload):
        calls.append("external")
        scope["last_payload"] = payload
        scope.setdefault('payloads', []).append(payload)
        return reply(payload)
    scope.update(_openrouter_request=internal, _external_llm_request=external)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_openrouter_error_details', '_completion_content')]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "completion", "exec"), scope)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "director", "exec"), scope)
    return scope, calls


class Frame:
    shape = (1, 16, 16, 3)
    def __getitem__(self, key):
        return self


class RoutingTests(unittest.TestCase):
    def test_completion_errors_are_actionable_and_redacted(self):
        scope, _ = load_director()
        extract = scope['_completion_content']
        provider = SimpleNamespace(model='vision-test', api_key='private-test-key')
        payload = {'messages': [{'role': 'user', 'content': 'private prompt contents'}]}
        for result in (
            {'error': {'code': 429, 'message': 'Provider busy private-test-key private prompt contents'}},
            {'message': 'Upstream unavailable private-test-key', 'status': 503},
            {'choices': []}, {'choices': [None]}, {'choices': [{'message': None}]},
            {'choices': [{'message': {'content': None, 'refusal': 'Request declined'}, 'finish_reason': 'stop'}]},
            {'choices': [{'message': {'content': ' '}, 'finish_reason': 'length'}]},
            [],
        ):
            with self.subTest(result=result), self.assertRaises(RuntimeError) as error:
                extract(result, payload, provider, 'Driven visual analysis')
            text = str(error.exception)
            self.assertIn('Driven visual analysis', text)
            self.assertIn('vision-test', text)
            self.assertNotIn('private-test-key', text)
            self.assertNotIn('private prompt contents', text)
        with self.assertRaisesRegex(RuntimeError, '429.*Provider busy'):
            extract({'error': {'code': 429, 'message': 'Provider busy'}}, payload, provider, 'Director generation')
        for content in ('{"roles": []}', {'roles': []}):
            self.assertEqual(extract({'choices': [{'message': {'content': content}}]},
                payload, provider, 'Driven visual analysis'), content)

    def test_driven_provider_error_stops_before_main_request(self):
        scope, calls = load_director()
        def failed_request(provider, payload):
            calls.append('failed')
            return {'error': {'code': 503, 'message': 'Upstream unavailable'}}
        scope['_external_llm_request'] = failed_request
        args = self.kwargs()
        args.update(llm=SimpleNamespace(model='vision-test'), driven_image=Frame())
        with self.assertRaisesRegex(RuntimeError, 'Driven visual analysis.*503.*Upstream unavailable'):
            scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, ['failed'])

    def test_schema_removes_storyboard_and_moods_and_exposes_driven_inputs(self):
        scope, _ = load_director()
        def input_field(name, **kwargs):
            return SimpleNamespace(id=name, **kwargs)
        port = SimpleNamespace(Input=input_field, Output=input_field)
        scope['io'] = SimpleNamespace(Schema=lambda **kwargs: SimpleNamespace(**kwargs),
            String=port, Combo=port, Image=port, Int=port, Float=port, Boolean=port,
            Custom=lambda name: port, Hidden=SimpleNamespace(unique_id='unique_id'))
        schema = scope['H3CompactMultimodalEditDirector'].define_schema()
        names = {field.id for field in schema.inputs}
        self.assertTrue({'driven_image', 'driven_video', 'cut'} <= names)
        self.assertFalse({'mood_image', 'mood_video', 'storyboard_image', 'shots'} & names)
        modes = next(field.options for field in schema.inputs if field.id == 'edit_mode')
        self.assertNotIn('Storyboard', modes)
        self.assertIn('Transfer', modes)
        strength = schema.inputs[-1]
        self.assertEqual(strength.id, 'driven_strength')
        self.assertEqual((strength.default, strength.min, strength.max), (3, 1, 3))

    def test_driven_scenario_transfers_activity_to_cast_in_all_modes(self):
        for mode in ('compact', 'edit', 'Elaborate', 'Enhance', 'Continuous Edit', 'Continuous Elaborate'):
            for driven_kind in ('image', 'video', 'both'):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, edit_request='The woman and man play together.',
                    reference_image_2=Frame(), llm=SimpleNamespace(model='vision-test'),
                    driven_image=Frame() if driven_kind != 'video' else None,
                    driven_video=Frame() if driven_kind != 'image' else None)
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                content = scope['last_payload']['messages'][1]['content']
                labels = ' '.join(item.get('text', '') for item in content)
                self.assertIn('DRIVEN SCENARIO — ACTIVITY AND WORLD TRANSFER', system)
                self.assertIn('those two characters\nplaying football', system)
                self.assertIn('not watching or posing beside it', system)
                self.assertIn('Explicit user instructions take priority', system)
                self.assertIn('Do not import new principal identities, swap faces', system)
                self.assertIn('wardrobe and accessories appropriate to each role', system)
                self.assertIn('a character reference alone does not lock clothing', system)
                self.assertIn('emotional tone, visual style and energy', system)
                self.assertIn('Once the adapted body/wardrobe/hair/world is established', system)
                self.assertGreater(system.index('DRIVEN SCENARIO —'), system.index('CINEMATIC CRAFT'))
                self.assertIn('including wardrobe and mood', system)
                self.assertIn('numbered person references define the complete principal cast', system)
                self.assertIn('woman becomes the cheerleader', system)
                self.assertIn('two principal people, never those two plus a third cheerleader', system)
                self.assertIn('never solve ambiguity by adding a person', system)
                self.assertIn('definitions AND the action description', system)
                self.assertIn('<Picture 1>', labels)
                self.assertIn('<Picture 2>', labels)
                self.assertNotIn('<Picture 3>', labels)
                analysis = scope['payloads'][0]
                analysis_content = analysis['messages'][1]['content']
                driver_labels = ' '.join(item.get('text', '') for item in analysis_content)
                self.assertEqual('<Driven Image 1>' in driver_labels, driven_kind != 'video')
                self.assertEqual('<Driven Video 1>' in driver_labels, driven_kind != 'image')
                self.assertNotIn('LITERAL VISUAL REFERENCE', driver_labels)
                self.assertIn('DRIVEN VISUAL BLUEPRINT', labels)
                self.assertIn('Team uniform and trainers', labels)
                self.assertIn('Daylight', labels)
                self.assertIn('Energetic', labels)
                self.assertIn('DRIVEN CAST COMPILATION', system)
                mapping = scope['last_payload']['response_format']['json_schema']['schema']['properties']['driven_cast_mapping']
                self.assertEqual(mapping['items']['properties']['identity_source']['enum'], ['<Picture 1>', '<Picture 2>'])
                self.assertNotIn('Mood Image', labels)
                image_count = sum(item['type'] == 'image_url' for item in content)
                self.assertEqual(image_count, 2)
                self.assertEqual(sum(item['type'] == 'image_url' for item in analysis_content),
                                 2 if driven_kind == 'both' else 1)
                self.assertEqual(calls, ['external', 'external'])
                self.assertIn('driven analysis input: 100 / output: 80', result[2])
                if mode in ('Enhance', 'Continuous Edit', 'Continuous Elaborate'):
                    prompts = json.loads(result[0])['scene_prompts']
                    self.assertTrue(all('Keep all visual assignments' not in prompt for prompt in prompts))
                    self.assertTrue(all('<Driven ' not in prompt for prompt in prompts))
                    self.assertIn('<Picture 2>', system)

    def test_driven_strength_selects_noncontradictory_policy_in_all_modes(self):
        for strength in (1, 2, 3):
            for mode in ('compact', 'edit', 'Elaborate', 'Enhance', 'Continuous Edit', 'Continuous Elaborate'):
                for kind in ('image', 'video', 'both'):
                    with self.subTest(strength=strength, mode=mode, kind=kind):
                        scope, calls = load_director()
                        args = self.kwargs()
                        args.update(edit_mode=mode, driven_strength=strength,
                            llm=SimpleNamespace(model='test'), reference_image_2=Frame(),
                            driven_image=Frame() if kind != 'video' else None,
                            driven_video=Frame() if kind != 'image' else None)
                        scope['H3CompactMultimodalEditDirector'].execute(**args)
                        system = scope['last_payload']['messages'][0]['content']
                        user = ' '.join(item.get('text', '') for item in scope['last_payload']['messages'][1]['content'])
                        self.assertIn(f'DRIVEN STRENGTH {strength}', system)
                        self.assertEqual(calls, ['external', 'external'])
                        self.assertIn('driven_cast_mapping', scope['last_payload']['response_format']['json_schema']['schema']['properties'])
                        if strength < 3:
                            self.assertNotIn('A person reference alone does not lock body proportions', system)
                            self.assertNotIn('Do not retain reference hair merely', system)
                            self.assertNotIn('Driven connection authorizes full-scene transfer', system)
                            self.assertNotIn('Transfer the exact visible garment design, hair color/length/style, body proportions', user)
                            self.assertIn('selected Driven Strength appearance policy', user)
                        if strength == 1:
                            self.assertIn('Driven does NOT authorize changing', system)
                            self.assertIn('COMPILE THE REFERENCE APPEARANCE', system)
                            self.assertNotIn('COMPILE THE COMPLETE OUTFIT', system)
                        elif strength == 2:
                            self.assertIn('reference body build/proportions', system)
                            self.assertIn("do not copy the driver's bust/waist/hip ratios", system)
                            self.assertIn('COMPILE THE COMPLETE OUTFIT', system)
                        else:
                            self.assertIn('BODY AND POSE FUSION', system)

    def test_transfer_scopes_creativity_and_keeps_sequence_contract(self):
        for strength in (1, 2, 3):
            for kind in ('image', 'video', 'both'):
                for count in (1, 3):
                    with self.subTest(strength=strength, kind=kind, count=count):
                        scope, calls = load_director()
                        args = self.kwargs()
                        args.update(edit_mode='Transfer', driven_strength=strength, cut=True,
                            continuous_scene_count=count, seconds_per_scene=10,
                            direction_context='Genre: Adventure\nDialogue: Spanish',
                            edit_request='The referenced woman runs as in the video, wearing office clothes.',
                            system_prompt='Invent a wild new story and add dialogue.',
                            llm=SimpleNamespace(model='test'),
                            driven_image=Frame() if kind != 'video' else None,
                            driven_video=Frame() if kind != 'image' else None)
                        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                        system = scope['last_payload']['messages'][0]['content']
                        self.assertIn('TRANSFER — FAITHFUL ADAPTATION', system)
                        self.assertIn(f'DRIVEN STRENGTH {strength}', system)
                        self.assertIn('Change only what the user asks', system)
                        self.assertIn('No dialogue, lyrics, narration or score unless', system)
                        self.assertIn('follow THAT block only', system)
                        self.assertIn('Dialogue language alone does NOT authorize speech', system)
                        self.assertIn('office clothes', system)
                        self.assertGreaterEqual(scope['last_payload']['max_tokens'], count * 500)
                        for forbidden in ('CREATIVE TREATMENT SELECTION', 'CUT SEQUENCE —',
                            'CINEMATIC CRAFT:', 'SCENARIO CONTINUATION:',
                            'Genre `Adventure` governs the premise', 'Invent a wild new story'):
                            self.assertNotIn(forbidden, system)
                        parsed = json.loads(result[0])
                        self.assertEqual(len(parsed['scene_prompts']), count)
                        self.assertEqual(result[3], count)
                        schema = scope['last_payload']['response_format']['json_schema']['schema']
                        self.assertNotIn('cut_scene_plan', schema['properties'])
                        self.assertEqual(calls, ['external', 'external'])
                        self.assertIn('Transfer transfer_sequence ready', result[1])

    def test_transfer_without_driver_or_references_has_no_extra_calls(self):
        for mode in ('Transfer', 'transfer'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, reference_image_1=None,
                edit_request='A woman runs in office clothes. Add one short greeting in Spanish.',
                continuous_scene_count=1, llm=SimpleNamespace(model='test'))
            result = scope['H3CompactMultimodalEditDirector'].execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            self.assertIn('TEXT-ONLY GENERATION', system)
            self.assertIn('invent only that dialogue', system)
            self.assertEqual(calls, ['external'])
            self.assertEqual(len(json.loads(result[0])['scene_prompts']), 1)

    def test_transfer_hold_cannot_reuse_enhance_or_other_scene_count(self):
        scope, calls = load_director()
        held = {}
        scope['_set_held_director_plan'] = lambda key, record: held.update(record)
        scope['_get_held_director_plan'] = lambda key: held
        args = self.kwargs()
        args.update(edit_mode='Enhance', llm=SimpleNamespace(model='test'))
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        calls.clear()
        args.update(edit_mode='Transfer', hold_prompt=True)
        with self.assertRaisesRegex(RuntimeError, 'different director mode'):
            scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, [])
        args['hold_prompt'] = False
        first = scope['H3CompactMultimodalEditDirector'].execute(**args)
        calls.clear()
        args['hold_prompt'] = True
        second = scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(first[0], second[0])
        args['continuous_scene_count'] = 3
        with self.assertRaisesRegex(RuntimeError, 'different number'):
            scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, [])

    def test_driven_strength_default_and_disconnected_compatibility(self):
        payloads = []
        for strength in (None, 3):
            scope, _ = load_director()
            args = self.kwargs()
            args.update(driven_image=Frame(), llm=SimpleNamespace(model='test'))
            if strength is not None:
                args['driven_strength'] = strength
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            payloads.append(scope['payloads'])
        self.assertEqual(payloads[0], payloads[1])
        payloads.clear()
        for strength in (1, 2, 3):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(driven_strength=strength, llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            self.assertEqual(calls, ['external'])
            payloads.append(scope['last_payload'])
        self.assertTrue(all(payload == payloads[0] for payload in payloads))

    def test_driven_strength_hold_rejects_change_without_calls_and_accepts_legacy_three(self):
        for strength in (1, 2, 3):
            scope, calls = load_director()
            held = {}
            scope['_set_held_director_plan'] = lambda key, record: held.update(record)
            scope['_get_held_director_plan'] = lambda key: held
            args = self.kwargs()
            args.update(driven_video=Frame(), driven_strength=strength, llm=SimpleNamespace(model='test'))
            first = scope['H3CompactMultimodalEditDirector'].execute(**args)
            self.assertEqual(held['driven_strength'], strength)
            calls.clear()
            args['hold_prompt'] = True
            if strength == 3:
                held.pop('driven_strength')
            second = scope['H3CompactMultimodalEditDirector'].execute(**args)
            self.assertEqual(first[0], second[0])
            args['driven_strength'] = 1 if strength != 1 else 2
            with self.assertRaisesRegex(RuntimeError, 'Driven Strength changed'):
                scope['H3CompactMultimodalEditDirector'].execute(**args)
            self.assertEqual(calls, [])

    def test_driven_rules_are_absent_when_inputs_are_disconnected(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Enhance', llm=SimpleNamespace(model='test'))
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertNotIn('DRIVEN SCENARIO — ACTIVITY AND WORLD TRANSFER',
                         scope['last_payload']['messages'][0]['content'])
        self.assertEqual(calls, ['external'])

    def test_driven_hair_and_garment_design_transfer_preserves_identity_and_locks(self):
        for mode in ('compact', 'edit', 'Elaborate', 'Enhance', 'Continuous Edit', 'Continuous Elaborate'):
            for i2v in (False, True):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, driven_image=Frame(), i2v_mode=i2v,
                            llm=SimpleNamespace(model='vision-test'))
                scope['H3CompactMultimodalEditDirector'].execute(**args)
                analysis = scope['payloads'][0]
                analysis_system = analysis['messages'][0]['content']
                roles = analysis['response_format']['json_schema']['schema']['properties']['roles']['items']
                self.assertIn('hair', roles['properties'])
                self.assertIn('hair', roles['required'])
                self.assertIn('hair is transferable styling, not identity', analysis_system)
                self.assertNotIn('Exclude face, hair', analysis_system)
                self.assertIn('cutout shapes and body-relative positions', analysis_system)
                self.assertIn('invent unseen garment details', analysis_system)
                system = scope['last_payload']['messages'][0]['content']
                self.assertIn('without transferring faces or age', system)
                self.assertIn('Do not retain reference hair merely because', system)
                self.assertIn('user explicitly locks it', system)
                self.assertIn('do not replace its hair or clothes', system)
                self.assertIn('Adapt fit to the fused target body, not the garment', system)
                self.assertIn('Bind facial identity to <Picture N> separately', system)
                labels = ' '.join(item.get('text', '') for item in scope['last_payload']['messages'][1]['content'])
                self.assertIn('Long blonde hair in a high ponytail', labels)
                self.assertEqual(calls, ['external', 'external'])

    def test_hold_rejects_old_driven_analysis_without_hair(self):
        scope, calls = load_director()
        held = {}
        scope['_set_held_director_plan'] = lambda key, record: held.update(record)
        scope['_get_held_director_plan'] = lambda key: held
        args = self.kwargs()
        args.update(edit_mode='Enhance', driven_image=Frame(), llm=SimpleNamespace(model='test'))
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(held['driven_pipeline'], 'visual-blueprint-v3')
        held['driven_pipeline'] = 'visual-blueprint-v1'
        calls.clear()
        args['hold_prompt'] = True
        with self.assertRaisesRegex(RuntimeError, 'Disable Hold once'):
            scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, [])

    def test_driven_body_pose_and_continuation_are_compiled_from_anonymous_roles(self):
        for mode in ('compact', 'edit', 'Elaborate', 'Enhance', 'Continuous Edit', 'Continuous Elaborate'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, driven_image=Frame(), reference_image_2=Frame(),
                        cut=True, llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            analysis = scope['payloads'][0]
            roles = analysis['response_format']['json_schema']['schema']['properties']['roles']['items']
            for field in ('body_proportions', 'pose'):
                self.assertIn(field, roles['required'])
            self.assertIn('foreshortening and garment volume', analysis['messages'][0]['content'])
            system = scope['last_payload']['messages'][0]['content']
            self.assertIn('BODY AND POSE FUSION', system)
            self.assertIn('retaining the numbered reference person', system)
            self.assertIn('frame overrides body/pose/hair/outfit', system)
            self.assertIn('translate stylized proportions into coherent anatomy', system)
            self.assertIn('SCENARIO CONTINUATION', system)
            self.assertIn('blueprint records visible evidence', system)
            self.assertIn('not a generic greeting or an', system)
            labels = ' '.join(item.get('text', '') for item in scope['last_payload']['messages'][1]['content'])
            self.assertIn('Athletic build with long limbs', labels)
            self.assertIn('Weight on left leg', labels)
            self.assertEqual(calls, ['external', 'external'])

    def test_cut_and_driven_priorities_have_no_plot_or_camera_conflict(self):
        for mode in ('Enhance', 'Continuous Elaborate'):
            for cut in (False, True):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, cut=cut, driven_image=Frame(),
                    llm=SimpleNamespace(model='test'),
                    direction_context='Genre: Adventure + Auto\nMotion style: Fast Motion + Auto')
                scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                self.assertEqual('do not invent a new plot' in system, not cut)
                self.assertNotIn('Genre `Auto` governs', system)
                self.assertIn('Genre `Adventure` governs', system)
                self.assertNotIn('energetic rapid action and responsive camera work', system)
                self.assertIn('motion speed does not require camera movement', system)
                self.assertIn('posed portrait does not lock the plot', system)
                self.assertIn('COMPILE THE COMPLETE OUTFIT', system)
                self.assertIn('requested cap replaces the beanie only', system)
                self.assertIn('does not remove the sweatshirt, vest', system)
                self.assertNotIn('No extra fields/workflows/references', system)
                if cut:
                    self.assertEqual(system.count('CUT SEQUENCE — ORIGINAL STORY AND COVERAGE DESIGN'), 1)
                    self.assertNotIn('CUT — SCENE-BOUNDARY EDITING OVERRIDE', system)
                    self.assertIn('if scene 1\nidentifies the route, scene 2 acts', system)
                    schema = scope['last_payload']['response_format']['json_schema']['schema']
                    self.assertEqual(list(schema['properties'])[0], 'driven_cast_mapping')
                    self.assertEqual(schema['required'][0], 'driven_cast_mapping')
                self.assertEqual(calls, ['external', 'external'])

    def test_driven_cast_mapping_rejects_a_third_driver_identity(self):
        scope, calls = load_director()
        original_request = scope['_external_llm_request']

        def invalid_mapping(provider, payload):
            result = original_request(provider, payload)
            if payload['response_format']['json_schema']['name'] != 'h3_driven_visual_blueprint':
                parsed = json.loads(result['choices'][0]['message']['content'])
                parsed['driven_cast_mapping'][0]['identity_source'] = '<Driven Image 1>'
                result['choices'][0]['message']['content'] = json.dumps(parsed)
            return result

        scope['_external_llm_request'] = invalid_mapping
        args = self.kwargs()
        args.update(edit_mode='Enhance', driven_image=Frame(), reference_image_2=Frame(),
                    llm=SimpleNamespace(model='test'))
        with self.assertRaisesRegex(ValueError, 'unreferenced identity'):
            scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, ['external', 'external'])

    def test_driven_hold_reuses_both_stages_without_calls(self):
        scope, calls = load_director()
        held = {}
        scope['_set_held_director_plan'] = lambda key, record: held.update(record)
        scope['_get_held_director_plan'] = lambda key: held
        args = self.kwargs()
        args.update(edit_mode='Enhance', driven_image=Frame(), llm=SimpleNamespace(model='test'))
        first = scope['H3CompactMultimodalEditDirector'].execute(**args)
        calls.clear()
        args['hold_prompt'] = True
        second = scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(first[0], second[0])
        self.assertEqual(calls, [])
        held.pop('driven_pipeline')
        with self.assertRaisesRegex(RuntimeError, 'Driven analysis changed'):
            scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, [])

    def test_cinematic_craft_preserves_contract_and_one_call(self):
        for mode in ("compact", "Elaborate", "Elaborate Continuous"):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(llm=SimpleNamespace(model="test"), edit_mode=mode)
            scope["H3CompactMultimodalEditDirector"].execute(**args)
            self.assertEqual(calls, ["external"])
            payload = scope["last_payload"]
            system = payload["messages"][0]["content"]
            self.assertIn("CINEMATIC CRAFT", system)
            self.assertIn("Carry forward positions", system)
            self.assertIn("Allow requested montage", system)
            self.assertIn("without generated-tail review", system)
            self.assertIn("locked camera/performance", system)
            self.assertIn("sparse video frames do not", system)
            self.assertEqual(payload["response_format"]["type"], "json_schema")

    def test_execution_rules_budget_action_camera_and_adapt_lighting_in_all_modes(self):
        for mode in ('compact', 'edit', 'Elaborate', 'Enhance', 'Continuous Edit', 'Continuous Elaborate'):
            for seconds in (4, 10, 30):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, seconds_per_scene=seconds,
                    llm=SimpleNamespace(model='test'), cut=True,
                    direction_context='Motion Style: Fast Motion\nCamera: Locked camera',
                    max_tokens=2500, temperature=0.35, seed=123)
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                payload = scope['last_payload']
                system = payload['messages'][0]['content']
                self.assertIn(f'EXECUTABLE FOOTAGE — {seconds} SECONDS PER CLIP', system)
                self.assertIn('budget the actual movement', system)
                self.assertIn('Fast Motion means brisk, legible performance', system)
                self.assertIn('reduce optional embellishment first', system)
                self.assertIn('one dominant behavior', system)
                self.assertIn('Keep a selected locked camera locked', system)
                self.assertIn('reinterpret reference/Driven light quality', system)
                self.assertIn('broad window daylight in a warehouse', system)
                self.assertIn('do not force\nphotorealism', system)
                self.assertEqual(payload['max_tokens'], 2500)
                self.assertEqual(payload['temperature'], 0.35)
                self.assertEqual(payload['seed'], 123)
                self.assertNotIn('EXECUTABLE FOOTAGE', result[0])
                self.assertEqual(calls, ['external'])

    def test_driven_execution_adapts_scenario_without_extra_analysis_calls(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Enhance', seconds_per_scene=10, cut=True,
                    driven_image=Frame(), llm=SimpleNamespace(model='test'))
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        system = scope['last_payload']['messages'][0]['content']
        self.assertGreater(system.index('EXECUTABLE FOOTAGE'), system.index('DRIVEN SCENARIO —'))
        self.assertIn('Plan events to fit EXECUTABLE FOOTAGE', system)
        self.assertIn('Preserve a\ndeliberately staged studio', system)
        self.assertEqual(calls, ['external', 'external'])

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
        args.update(edit_mode="Elaborate", seconds_per_scene=30.0,
                    llm=SimpleNamespace(model="local-test"))
        scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(calls, ["external"])
        self.assertIn("30.00 SECONDS PER SCENE", scope["last_payload"]["messages"][0]["content"])

        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode="Elaborate", seconds_per_scene=30.5,
                    llm=SimpleNamespace(model="local-test"))
        with self.assertRaisesRegex(ValueError, "between 1 and 30"):
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

    def test_sequence_compilation_uses_local_shots_and_removes_internal_prose(self):
        scope, calls = load_director()
        prompt = (
            'Keep all visual assignments from <Driven Image 1> active in this scene. '
            'subject_definitions:\n<Subject 1> the woman from <Picture 1>, black jacket from the Driven role.\n\n'
            'integrated_multimodal_description:\n[Shot 2] At 00:10.000, <Subject 1> enters the courtyard.\n\n'
            '[Shot 3] At 00:13.250, She finds the key and opens the gate.\n\n'
            'overall_soundscape:\nFootsteps.\n\nnon_diegetic_music:\nN/A')
        original_request = scope['_external_llm_request']

        def reply(provider, payload):
            result = original_request(provider, payload)
            parsed = json.loads(result['choices'][0]['message']['content'])
            parsed['scene_prompts'] = [prompt] * len(parsed['scene_prompts'])
            result['choices'][0]['message']['content'] = json.dumps(parsed)
            return result

        scope['_external_llm_request'] = reply
        held = {}
        scope['_set_held_director_plan'] = lambda key, record: held.update(record)
        scope['_get_held_director_plan'] = lambda key: held
        args = self.kwargs()
        args.update(edit_mode='Enhance', llm=SimpleNamespace(model='test'))
        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
        for item in json.loads(result[0])['scene_prompts']:
            self.assertNotIn('Keep all visual assignments', item)
            self.assertNotIn('from the Driven role', item)
            self.assertIn('black jacket.', item)
            self.assertIn('[Shot 1] <Subject 1> enters', item)
            self.assertIn('[Shot 2] At 00:03.250,', item)
            self.assertNotIn('00:10.000', item)
            self.assertIn('opens the gate', item)
        self.assertEqual(calls, ['external'])
        held['edit_prompt'] = json.dumps({'scene_prompts': ['[reference generation]\n\n' + prompt] * args['continuous_scene_count']})
        calls.clear()
        args['hold_prompt'] = True
        self.assertEqual(scope['H3CompactMultimodalEditDirector'].execute(**args)[0], result[0])
        self.assertEqual(calls, [])

    def test_frame_zero_alignment_is_only_allowed_with_i2v_enabled(self):
        alignment = ('For the target video, at 0.00 seconds into the target video, '
                     '<Picture 1> (from [Shot 1]) is fully referenced.')
        binding = '<Subject 1> the woman fully referenced from <Picture 1>.'
        for mode in ('compact', 'Elaborate', 'Enhance', 'Continuous Edit', 'Continuous Elaborate'):
            for enabled in (False, True):
                scope, calls = load_director()
                original_request = scope['_external_llm_request']

                def response_with_alignment(provider, payload):
                    result = original_request(provider, payload)
                    parsed = json.loads(result['choices'][0]['message']['content'])
                    if 'scene_prompts' in parsed:
                        parsed['scene_prompts'] = [alignment + '\n\n' + binding
                                                  for _ in parsed['scene_prompts']]
                    else:
                        parsed['edit_prompt'] = alignment + '\n\n' + binding
                    result['choices'][0]['message']['content'] = json.dumps(parsed)
                    return result

                scope['_external_llm_request'] = response_with_alignment
                args = self.kwargs()
                args.update(edit_mode=mode, i2v_mode=enabled, llm=SimpleNamespace(model='test'))
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                self.assertEqual(alignment in system, enabled)
                self.assertEqual(alignment in result[0], enabled)
                self.assertIn(binding, result[0])
                self.assertEqual(calls, ['external'])

    def test_i2v_off_cleans_alignment_from_old_held_prompt_without_calls(self):
        for mode in ('Elaborate', 'Enhance'):
            scope, calls = load_director()
            held = {}
            scope['_set_held_director_plan'] = lambda key, record: held.update(record)
            scope['_get_held_director_plan'] = lambda key: held
            args = self.kwargs()
            args.update(edit_mode=mode, llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            alignment = ('For the target video, at 0.00 seconds into the target video, '
                         '<Picture 1> (from [Shot 1]) is fully referenced.')
            held['edit_prompt'] = held['edit_prompt'].replace('<Picture 1>', alignment + ' <Picture 1>')
            held['preview_prompt'] = alignment + '\n' + held['preview_prompt']
            calls.clear()
            args['hold_prompt'] = True
            result = scope['H3CompactMultimodalEditDirector'].execute(**args)
            self.assertNotIn(alignment, result[0])
            self.assertIn('<Picture 1>', result[0])
            if mode == 'Enhance':
                self.assertEqual(len(json.loads(result[0])['scene_prompts']), args['continuous_scene_count'])
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
                    self.assertIn(scope["H3CompactMultimodalEditDirector"].H3_NATIVE_PROMPT_RULES, system)
                    self.assertNotIn(scope["H3CompactMultimodalEditDirector"].ELABORATE_RULES, system)
                self.assertEqual(calls, ["external"])

    def test_sequence_detail_budget_and_causality_without_temperature_change(self):
        for mode, ceiling, tokens_per_scene in (
            ("Enhance", "up to 260", 350),
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

    def test_creative_exploration_is_scoped_and_does_not_add_calls(self):
        for mode, candidates in (("Elaborate", 3), ("Continuous Elaborate", 3),
                                 ("Enhance", 3), ("compact", None),
                                 ("edit", None), ("Continuous Edit", None)):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, llm=SimpleNamespace(model="test"))
            scope["H3CompactMultimodalEditDirector"].execute(**args)
            system = scope["last_payload"]["messages"][0]["content"]
            self.assertEqual("CREATIVE TREATMENT SELECTION" in system, candidates is not None)
            if candidates:
                self.assertIn(f"consider {candidates} brief", system)
                self.assertIn("locked camera/performance/audio", system)
                self.assertIn("without frame review", system)
                self.assertIn("never add a major event", system)
                self.assertIn("purposeful held beat", system)
                self.assertIn("return alternatives, scores, reasoning or extra JSON fields", system)
            self.assertEqual(calls, ["external"])

    def test_enhance_duration_budget_preserves_schema_and_user_settings(self):
        for seconds, words, allowance in ((4, "90–140", 350), (10, "120–200", 500),
                                         (30, "160–240", 650)):
            for count in (1, 3):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode="Enhance", llm=SimpleNamespace(model="test"),
                            seconds_per_scene=seconds, continuous_scene_count=count,
                            max_tokens=256, temperature=0.35, seed=123)
                result = scope["H3CompactMultimodalEditDirector"].execute(**args)
                payload = scope["last_payload"]
                system = payload["messages"][0]["content"]
                self.assertIn("Think like Elaborate; write economically", system)
                self.assertIn("Compress the treatment, not the imagination", system)
                self.assertIn("two or three complementary choices", system)
                self.assertIn("Delete filler before creative decisions", system)
                self.assertIn("when motivated and unlocked", system)
                self.assertIn("roughly half the developed prose", system)
                self.assertIn("same four H3 sections as Elaborate", system)
                self.assertIn("subject_definitions:", system)
                self.assertIn("integrated_multimodal_description:", system)
                self.assertIn("overall_soundscape:", system)
                self.assertIn("non_diegetic_music:", system)
                self.assertIn("Define <Subject N> through", system)
                self.assertNotIn('never substitute an untagged "same person" or invent <Subject N>', system)
                self.assertIn(words, system)
                self.assertIn(f"Each scene lasts {seconds:g} seconds", system)
                self.assertIn("Preserve supplied", system)
                self.assertNotIn("CREATIVE DIRECTOR MANDATE", system)
                self.assertEqual(payload["max_tokens"], count * allowance)
                self.assertEqual(payload["temperature"], 0.35)
                self.assertEqual(payload["seed"], 123)
                self.assertEqual(len(json.loads(result[0])["scene_prompts"]), count)
                self.assertEqual(calls, ["external"])

    def test_enhance_does_not_reduce_explicit_max_tokens(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode="Enhance", llm=SimpleNamespace(model="test"), max_tokens=2500)
        scope["H3CompactMultimodalEditDirector"].execute(**args)
        self.assertEqual(scope["last_payload"]["max_tokens"], 2500)
        self.assertEqual(calls, ["external"])

    def test_enhance_preserves_native_sections_without_extra_calls_or_truncation(self):
        scope, calls = load_director()
        prompt = (
            'subject_definitions:\n<Subject 1> the woman from <Picture 1>, blue jacket.\n\n'
            'integrated_multimodal_description:\n[Shot 1] <Subject 1> crosses a wet courtyard; '
            'a low tracking frame follows her reflection into warm doorway light.\n\n'
            'overall_soundscape:\nFootsteps and falling rain.\n\n'
            'non_diegetic_music:\nN/A')

        def reply(provider, payload):
            calls.append('external')
            return {'choices': [{'message': {'content': json.dumps({'scene_prompts': [prompt]})}}]}

        scope['_external_llm_request'] = reply
        args = self.kwargs()
        args.update(edit_mode='Enhance', llm=SimpleNamespace(model='test'), continuous_scene_count=1)
        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(json.loads(result[0])['scene_prompts'], ['[reference generation]\n\n' + prompt])
        self.assertNotIn('missing H3 sections', result[1])
        self.assertEqual(calls, ['external'])

    def test_cut_is_scoped_and_preserves_string_outputs(self):
        for mode in ('Enhance', 'Continuous Elaborate', 'Elaborate', 'compact', 'edit', 'Continuous Edit'):
            for cut in (False, True):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, cut=cut, llm=SimpleNamespace(model='test'))
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                enabled = cut and mode in ('Enhance', 'Continuous Elaborate')
                properties = scope['last_payload']['response_format']['json_schema']['schema']['properties']
                self.assertEqual('cut_scene_plan' in properties, enabled)
                self.assertNotIn('cut_scene_plan', result[0])
                self.assertEqual('CUT PRIORITY:' in system, enabled)
                if enabled:
                    plan = properties['cut_scene_plan']['items']
                    self.assertIn('change_from_previous', plan['required'])
                    self.assertIn('CUT CONTRAST', system)
                    self.assertIn('unmistakable coverage contrast', system)
                    self.assertIn('swapped with its neighbor', system)
                    self.assertIn('substantive change in BOTH activity/story purpose and coverage', system)
                    self.assertIn('outline must actually be realized', system)
                    self.assertIn('Never replay a completed', system)
                    self.assertIn('PLAN CAUSAL RESULTS, NOT EXCHANGES', system)
                    self.assertIn('CUT SEQUENCE — ORIGINAL STORY AND COVERAGE DESIGN', system)
                    self.assertIn('invent relevant encounters, discoveries, decisions', system)
                    self.assertNotIn(scope['H3CompactMultimodalEditDirector'].CONTINUOUS_RULES, system)
                    self.assertNotIn(scope['H3CompactMultimodalEditDirector'].CONTINUOUS_ELABORATE_RULES, system)
                    self.assertNotIn(scope['H3CompactMultimodalEditDirector'].CONTINUOUS_EDIT_RULES, system)
                    self.assertNotIn(scope['H3CompactMultimodalEditDirector'].CONTEXTUAL_REACTION_RULES, system)
                    self.assertIn('its consequence or preparation', system)
                    self.assertIn('Locked stories remain intact', system)
                    self.assertGreater(system.index('CUT PRIORITY:'), system.index('CINEMATIC CRAFT'))
                    self.assertIn('only scene 1 aligns', system)
                    self.assertIn('Masked Cut', system)
                    self.assertNotIn('ONE-PASS LOGICAL CONTINUITY FOR H3 CHAINS', system)
                    self.assertIn('Cut on', result[1])
                if mode == 'Continuous Elaborate' and not cut:
                    self.assertIn('ONE-PASS LOGICAL CONTINUITY FOR H3 CHAINS', system)
                props = scope['last_payload']['response_format']['json_schema']['schema']['properties']
                item = props['scene_prompts']['items'] if 'scene_prompts' in props else props['edit_prompt']
                self.assertEqual(item['type'], 'string')
                self.assertNotIn('SHOT COUNT OVERRIDE', system)
                self.assertEqual(calls, ['external'])

    def test_hold_rejects_changed_cut_or_retired_shots(self):
        scope, calls = load_director()
        held = {}
        scope['_set_held_director_plan'] = lambda key, record: held.update(record)
        scope['_get_held_director_plan'] = lambda key: held
        args = self.kwargs()
        args.update(edit_mode='Enhance', llm=SimpleNamespace(model='test'))
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        calls.clear()
        args.update(hold_prompt=True, cut=True)
        with self.assertRaisesRegex(RuntimeError, 'Cut settings changed'):
            scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, [])
        args['cut'] = False
        held['shots'] = 3
        with self.assertRaisesRegex(RuntimeError, 'Cut settings changed'):
            scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, [])

    def test_enhance_uses_less_input_and_output_budget_than_elaborate_sequence(self):
        payloads = {}
        for mode in ("Enhance", "Continuous Elaborate"):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, llm=SimpleNamespace(model="test"),
                        max_tokens=256, seconds_per_scene=10,
                        direction_context="Genre: Gothic Horror")
            scope["H3CompactMultimodalEditDirector"].execute(**args)
            payloads[mode] = scope["last_payload"]
            self.assertEqual(calls, ["external"])
        self.assertLess(len(payloads["Enhance"]["messages"][0]["content"]),
                        len(payloads["Continuous Elaborate"]["messages"][0]["content"]))
        self.assertLess(payloads["Enhance"]["max_tokens"],
                        payloads["Continuous Elaborate"]["max_tokens"])


if __name__ == "__main__":
    unittest.main()
