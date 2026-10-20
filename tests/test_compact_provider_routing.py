"""Offline contract tests: no model loads, HTTP requests or paid generations."""
import ast
import inspect
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
        MOTION_STYLES=next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == 'MOTION_STYLES' for target in node.targets)),
        VISUAL_LOOKS=next(ast.literal_eval(node.value)
                for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'VISUAL_LOOKS'
                        for target in node.targets)),
        _get_held_director_plan=lambda key: None,
        _set_held_director_plan=lambda key, record: None,
        _image_data_url=lambda *args: "data:image/png;base64,TEST",
        _credits=lambda *args: calls.append("credits") or "balance",
    )
    def reply(payload):
        if payload['response_format']['json_schema']['name'] == 'h3_creative_plus_treatment':
            count = payload['response_format']['json_schema']['schema']['properties']['scene_beats']['minItems']
            content = dict(premise='A hesitant reunion becomes a shared decision.',
                cast_and_relationships=['<Picture 1>: wants reconciliation; measured voice and restrained gestures.'],
                production_design='Warm practical light, coherent venue and permitted wardrobe design.',
                scene_beats=[f'Beat {i + 1}: a small choice changes their distance.' for i in range(count)],
                continuity_locks=['Keep reference ownership and completed actions.'])
            return {'choices': [{'message': {'content': json.dumps(content)}}],
                    'usage': {'prompt_tokens': 250, 'completion_tokens': 150, 'total_tokens': 400}}
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
        if 'transfer_cast' in properties:
            cast = properties['transfer_cast']['items']['properties']
            performance = cast['performance_source']['enum'][0]
            content['transfer_cast'] = [dict(subject_tag='<Subject 1>',
                identity_source=cast['identity_source']['enum'][0], identity_details='the woman',
                performance_source=performance,
                performance_role='cheerleader' if performance.startswith('<Driven ') else 'performer',
                reference_wardrobe='Blue jacket', performance_wardrobe='Team uniform and trainers',
                wardrobe_source='performance', requested_wardrobe='N/A')]
        transfer_item = properties.get('scene_prompts', {}).get('items', {})
        if 'source_video' in transfer_item.get('properties', {}):
            content['scene_prompts'] = [dict(
                source_video=transfer_item['properties']['source_video']['enum'][0],
                edit_instruction='replace the performer with the woman from <Picture 1>. Keep the observed motion.',
                subject_definitions='<Subject 1> the woman whose identity comes from <Picture 1>.',
                retention_analysis='<Subject 1>: attribute_transfer - appearance transfers from <Picture 1>; source performance is retained.',
                detailed_description='[Shot 1] She follows the observed turning and reaching action.',
                overall_soundscape='N/A', non_diegetic_music='N/A') for _ in content['scene_prompts']]
        if 'target_design' in properties:
            content['target_design'] = dict(
                subjects=[dict(subject_tag='<Subject 1>',
                    identity='the woman fully referenced from <Picture 1>.',
                    target_appearance='wearing a blue jacket.')],
                visual_style='Soft directional light and restrained blue color.')
            subject_fields = properties['target_design']['properties']['subjects']['items']['properties']
            if 'wardrobe_contrast' in subject_fields:
                content['target_design']['subjects'][0]['wardrobe_contrast'] = 'Reference top -> blue tailored jacket.'
            if 'target_body' in subject_fields:
                content['target_design']['subjects'][0]['target_body'] = 'an athletic build with balanced limb proportions.'
            def scene_body(text):
                body = dict(opening_framing='An eye-level wide shot of the courtyard.',
                            integrated_multimodal_description=text,
                            visible_result='She reaches the far doorway.',
                            overall_soundscape='Quiet footsteps.', non_diegetic_music='N/A')
                body_schema = properties.get('scene_prompts', {}).get('items', properties.get('edit_prompt', {}))
                if 'environment' in body_schema.get('properties', {}):
                    body['environment'] = 'A courtyard with pale stone arches and a shaded fountain.'
                return body
            if 'scene_prompts' in content:
                content['scene_prompts'] = [scene_body(text) for text in content['scene_prompts']]
            else:
                content['edit_prompt'] = scene_body(content['edit_prompt'])
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
    def test_context_controls_shape_concrete_design_in_standard_and_plus(self):
        for mode in ('Enhance', 'Elaborate', 'Elaborate Plus'):
            for profile in ('Standard', 'Creative Plus'):
                with self.subTest(mode=mode, profile=profile):
                    scope, calls = load_director()
                    cls = scope['H3CompactMultimodalEditDirector']
                    args = self.kwargs()
                    args.update(edit_mode=mode, creative_profile=profile,
                        edit_request='Two friends meet on a rooftop and decide where to go next.',
                        direction_context='Genre: Romance\nMotion style: Fast Motion + Locked-Off Camera\nVisual look: 2000s Y2K Digital Camera\nWardrobe: Create New\nEnvironment: Create New',
                        llm=SimpleNamespace(model='test'), max_tokens=2400, seed=42)
                    cls.execute(**args)
                    payload = scope['last_payload']
                    system = payload['messages'][0]['content']
                    cue = cls.PERIOD_DESIGN_CUES['2000s Y2K Digital Camera']
                    self.assertIn(cue, system)
                    self.assertIn('Carry those decisions through every block', system)
                    for other, other_cue in cls.PERIOD_DESIGN_CUES.items():
                        if other != '2000s Y2K Digital Camera':
                            self.assertNotIn(other_cue, system)
                    props = payload['response_format']['json_schema']['schema']['properties']
                    appearance = props['target_design']['properties']['subjects']['items']['properties']['target_appearance']
                    self.assertIn('silhouette, materials', appearance['description'])
                    self.assertIn('not an era label', appearance['description'])
                    scene = props['edit_prompt'] if mode == 'Elaborate' else props['scene_prompts']['items']
                    self.assertIn('active period/world controls', scene['properties']['environment']['description'])
                    self.assertIn('physical cadence and camera behavior', scene['properties']['integrated_multimodal_description']['description'])
                    self.assertEqual(payload['max_tokens'], 2400)
                    self.assertEqual(payload['seed'], 42)
                    self.assertEqual(calls, ['external'] * (2 if profile == 'Creative Plus' else 1))
                    if profile == 'Creative Plus':
                        brief = json.loads(scope['payloads'][0]['messages'][1]['content'])
                        self.assertIn(cue, '\n'.join(brief['resolved_direction_controls']))
                        self.assertIn('ART DIRECTION IS A DECISION', system)
                        self.assertIn('revisable beats do not make selected controls optional', system)

    def test_period_design_cues_are_selected_only_and_preserve_locks(self):
        scope, _ = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        self.assertEqual(set(cls.PERIOD_DESIGN_CUES), set(cls.PERIOD_LOOKS))
        for profile in ('Standard', 'Creative Plus'):
            for mode, look in (('Enhance', 'Auto + none'), ('Enhance', 'VHS / Analog Tape'),
                               ('Transfer', '2000s Y2K Digital Camera')):
                args = self.kwargs()
                args.update(edit_mode=mode, creative_profile=profile,
                            direction_context='Visual look: ' + look,
                            llm=SimpleNamespace(model='test'))
                cls.execute(**args)
                for payload in scope['payloads']:
                    text = json.dumps(payload['messages'])
                    for cue in cls.PERIOD_DESIGN_CUES.values():
                        self.assertNotIn(cue, text)
                scope['payloads'] = []
            args.update(edit_mode='Enhance', i2v_mode=True,
                direction_context='Visual look: 2000s Y2K Digital Camera\nWardrobe: Keep Reference\nEnvironment: Keep Reference')
            cls.execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            self.assertIn('explicit Keep Reference, user locks, source responsibilities and I2V opening facts win', system)
            self.assertNotIn('WARDROBE DESIGN CHECK', system)
            self.assertNotIn('ENVIRONMENT DESIGN CHECK', system)
            scope['payloads'] = []

    def test_plus_preserves_standard_scene_schema_craft_and_output_budget(self):
        for mode in ('Enhance', 'Elaborate', 'Elaborate Plus'):
            schemas, budgets, system_sizes = [], [], []
            for profile in ('Standard', 'Creative Plus'):
                scope, calls = load_director()
                cls = scope['H3CompactMultimodalEditDirector']
                args = self.kwargs()
                args.update(edit_mode=mode, creative_profile=profile,
                    edit_request='A student leaves university, takes the bus and reaches home.',
                    direction_context='Wardrobe: Create New\nEnvironment: Create New\nMotion style: Fast Motion',
                    llm=SimpleNamespace(model='test'))
                cls.execute(**args)
                payload = scope['last_payload']
                system = payload['messages'][0]['content']
                schemas.append(payload['response_format'])
                budgets.append(payload['max_tokens'])
                system_sizes.append(len(system.split()))
                for rule in (cls.CREATIVE_TREATMENT_RULES.format(candidates=3),
                             cls.WRITER_DIRECTOR_RULES, cls.CINEMATIC_CRAFT,
                             cls.H3_NATIVE_PROMPT_RULES):
                    self.assertEqual(system.count(rule), 1)
                self.assertEqual(calls, ['external'] * (2 if profile == 'Creative Plus' else 1))
            self.assertEqual(schemas[0], schemas[1])
            self.assertEqual(budgets[0], budgets[1])
            self.assertLess(system_sizes[1] - system_sizes[0], 800)

    def test_plus_drops_blanket_edit_and_continuous_camera_rules_not_user_locks(self):
        for mode in ('Enhance', 'Elaborate', 'Elaborate Plus'):
            scope, _ = load_director()
            cls = scope['H3CompactMultimodalEditDirector']
            args = self.kwargs()
            args.update(edit_mode=mode, creative_profile='Creative Plus',
                system_prompt=cls.DEFAULT_SYSTEM_PROMPT,
                direction_context='Motion style: Fast Motion + Orbit Camera',
                llm=SimpleNamespace(model='test'))
            cls.execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            for obsolete in (cls.DEFAULT_SYSTEM_PROMPT, cls.DEEP_EDIT_RULES,
                             cls.CONTINUOUS_EDIT_RULES, cls.CONTINUOUS_ELABORATE_RULES,
                             cls.CONTINUOUS_H3_HANDOFF_RULES, cls.CONTROL_CONTINUOUS_OUTPUT_RULES,
                             'within the continuous take', 'without implied time jumps',
                             'PRIVATE SHOT SCHEMA', 'start_seconds'):
                self.assertNotIn(obsolete, system)
            for protected in ('An explicit single take/no cuts, locked source coverage',
                              'Inherited AV/prefix footage owns the opening',
                              'after the protected prefix, never at frame zero',
                              scope['MOTION_STYLES']['Fast Motion'], scope['MOTION_STYLES']['Orbit Camera']):
                self.assertIn(protected, system)
            for payload in scope['payloads']:
                self.assertNotIn('Do not add cuts, new output sections', payload['messages'][0]['content'])
            args['system_prompt'] = 'One continuous take, no cuts. Keep the exact supplied dialogue.'
            cls.execute(**args)
            self.assertIn(args['system_prompt'], scope['last_payload']['messages'][0]['content'])

    def test_plus_supports_creative_decisions_without_inflating_the_premise(self):
        scope, calls = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        args = self.kwargs()
        args.update(edit_mode='Elaborate Plus', creative_profile='Creative Plus',
                    edit_request='A quiet reunion over coffee, with a warm resolved ending.',
                    llm=SimpleNamespace(model='test'))
        cls.execute(**args)
        planner, writer = scope['payloads']
        self.assertLess(len(planner['messages'][0]['content'].split()), 1200)
        system = writer['messages'][0]['content']
        for phrase in ('optional creative support', 'not a mandatory screenplay',
                       'not a replacement premise', 'may improve its dialogue, acting, beats',
                       'A simple\npremise can stay simple', 'not exaggeration',
                       'do not manufacture a threat', 'each block change something concrete'):
            self.assertIn(phrase, system)
        self.assertEqual(calls, ['external', 'external'])

    def test_elaborate_plus_renames_the_mode_without_changing_legacy_behavior(self):
        for profile in ('Standard', 'Creative Plus'):
            results = []
            schemas = []
            for mode in ('Elaborate Plus', 'Continuous Elaborate', 'Elaborate Continuo', 'continuous_elaborate'):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, creative_profile=profile,
                            llm=SimpleNamespace(model='test'))
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                self.assertTrue(result[1].startswith('Elaborate Plus '))
                self.assertEqual(calls, ['external'] * (2 if profile == 'Creative Plus' else 1))
                results.append(result[0])
                schemas.append(scope['last_payload']['response_format'])
            self.assertTrue(all(value == results[0] for value in results))
            self.assertTrue(all(value == schemas[0] for value in schemas))

    def test_reference_subject_order_renumbers_actions_and_speakers_simultaneously(self):
        scope, _ = load_director()
        order = scope['H3CompactMultimodalEditDirector']._order_reference_subjects
        prompt = ('subject_definitions:\n'
                  '<Subject 1> the man from <Picture 2>, wearing a coat.\n'
                  '<Subject 2> the woman from <Picture 1>, wearing a jacket.\n\n'
                  'integrated_multimodal_description:\n'
                  '[Shot 1] <Subject 2> Woman (S2) says: <d>[Spanish] Hola.</d> '
                  '<Subject 1> Man (S1) listens.\n'
                  '[Shot 2] <Subject 1> Man (S1) says: <d>[Spanish] Vamos.</d>\n\n'
                  'overall_soundscape:\nFootsteps.\n\nnon_diegetic_music:\nN/A')
        result = order(prompt)
        self.assertLess(result.index('<Subject 1> the woman'), result.index('<Subject 2> the man'))
        self.assertIn('<Subject 1> Woman (S1) says: <d>[Spanish] Hola.</d>', result)
        self.assertIn('<Subject 2> Man (S2) says: <d>[Spanish] Vamos.</d>', result)
        self.assertEqual(result.count('<d>'), 2)
        self.assertEqual(order(result), result)

    def test_reference_subject_order_avoids_extra_collisions_and_preserves_unknown_bindings(self):
        scope, _ = load_director()
        order = scope['H3CompactMultimodalEditDirector']._order_reference_subjects
        prompt = ('subject_definitions:\n<Subject 1> an independent waiter.\n'
                  '<Subject 3> the woman from <Picture 1>.\n\n'
                  'integrated_multimodal_description:\n<Subject 1> (S1) serves <Subject 3> (S3).')
        result = order(prompt)
        self.assertIn('<Subject 1> the woman from <Picture 1>', result)
        self.assertIn('<Subject 4> an independent waiter', result)
        self.assertIn('<Subject 4> (S4) serves <Subject 1> (S1)', result)
        for original in ('A plain prompt.', 'subject_definitions:\nN/A\n\nsummary:\nA landscape.',
                         'subject_definitions:\n<Subject 1> a blend of <Picture 1> and <Picture 2>.\n\nsummary:\nA portrait.'):
            self.assertEqual(order(original), original)

    def test_creative_arc_scales_to_scene_budget_without_more_calls(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
            for count in (1, 2, 3, 12):
                with self.subTest(mode=mode, count=count):
                    scope, calls = load_director()
                    args = self.kwargs()
                    args.update(edit_mode=mode, continuous_scene_count=count,
                                seconds_per_scene=5, llm=SimpleNamespace(model='test'))
                    scope['H3CompactMultimodalEditDirector'].execute(**args)
                    system = scope['last_payload']['messages'][0]['content']
                    actual_count = 1 if mode == 'Elaborate' else count
                    self.assertIn(f'Budget: {actual_count} scene(s), 5s each, {actual_count * 5}s total.', system)
                    self.assertIn('One scene contains setup/turn/payoff', system)
                    self.assertIn('two establish/develop then resolve', system)
                    self.assertIn('Reserve time INSIDE the last scene', system)
                    self.assertIn('close only at the global end', system)
                    self.assertEqual(calls, ['external'])

    def test_plus_planner_shares_arc_budget_but_faithful_modes_do_not(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Enhance', continuous_scene_count=3, seconds_per_scene=10,
                    creative_profile='Creative Plus', llm=SimpleNamespace(model='test'))
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        for payload in scope['payloads']:
            system = payload['messages'][0]['content']
            self.assertIn('Budget: 3 scene(s), 10s each, 30s total.', system)
            self.assertIn('Source-locked performance is not rewritten', system)
            self.assertIn('Trailers, loops, deliberate open', system)
        self.assertEqual(calls, ['external', 'external'])
        for mode in ('Transfer', 'edit', 'Continuous Edit', 'compact'):
            scope, calls = load_director()
            args.update(edit_mode=mode)
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            self.assertNotIn('DRAMATIZE, DO NOT DEFER — FINITE SEQUENCE',
                             scope['last_payload']['messages'][0]['content'])
            self.assertEqual(calls, ['external'])

    def test_reference_identity_is_exclusive_without_extra_requests(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate', 'Transfer',
                     'compact', 'edit', 'Continuous Edit'):
            with self.subTest(mode=mode):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, llm=SimpleNamespace(model='test'),
                            edit_request='The referenced couple have dinner; a waiter serves them.')
                scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                for rule in ('CAST IDENTITY — EXCLUSIVE REFERENCE OWNERSHIP',
                             'One reference identity, one character across blocks',
                             'not merely different uniforms',
                             'Bind Picture tags locally, never to groups/extras',
                             'silhouette/uniform/activity, not facial detail',
                             'preserve source population/role tracking',
                             'replace only assigned performers',
                             'duplicates require explicit requests'):
                    self.assertIn(rule, system)
                self.assertEqual(calls, ['external'])

    def test_creative_plus_plans_independent_supporting_cast(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Enhance', creative_profile='Creative Plus',
                    llm=SimpleNamespace(model='test'),
                    edit_request='The referenced couple have dinner; a waiter serves them.')
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        for payload in scope['payloads']:
            self.assertIn('CAST IDENTITY — EXCLUSIVE REFERENCE OWNERSHIP',
                          payload['messages'][0]['content'])
            self.assertIn('independent\nfaces/hair/silhouettes',
                          payload['messages'][0]['content'])
        self.assertEqual(calls, ['external', 'external'])

    def test_creative_plus_adds_one_text_only_treatment_in_creative_modes(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate', 'Transfer',
                     'compact', 'edit', 'Continuous Edit'):
            with self.subTest(mode=mode):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, creative_profile='Creative Plus',
                            llm=SimpleNamespace(model='test'), temperature=0.35, seed=42)
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                active = mode in ('Enhance', 'Elaborate', 'Continuous Elaborate')
                self.assertEqual(calls, ['external'] * (2 if active else 1))
                if active:
                    planner, director = scope['payloads']
                    self.assertEqual(planner['response_format']['json_schema']['name'], 'h3_creative_plus_treatment')
                    self.assertIn('Return a compact JSON treatment matching the schema', planner['messages'][0]['content'])
                    embedded_schema = planner['messages'][0]['content'].split('JSON Schema:\n', 1)[1]
                    self.assertEqual(json.loads(embedded_schema), planner['response_format']['json_schema']['schema'])
                    self.assertIsInstance(planner['messages'][1]['content'], str)
                    self.assertNotIn('base64', json.dumps(planner))
                    self.assertLessEqual(planner['max_tokens'], 1600)
                    self.assertEqual(planner['temperature'], 0.35)
                    self.assertEqual(planner['seed'], 42)
                    self.assertFalse(planner['reasoning']['enabled'])
                    self.assertEqual(director['max_tokens'], 1200)
                    self.assertEqual(director['messages'][0]['content'].count('CREATIVE TREATMENT SELECTION'), 1)
                    self.assertIn('CREATIVE PLUS EXECUTION', director['messages'][0]['content'])
                    self.assertIn('Creative Plus planning input: 250 / output: 150', result[2])
                    self.assertIn('2 stages', result[2])
                    self.assertNotIn('cast_and_relationships', result[0])
                else:
                    self.assertIn('Creative Plus inactive', result[1])

    def test_creative_plus_unifies_authorship_without_new_fields_calls_or_output_budget(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
            for profile in ('Standard', 'Creative Plus'):
                with self.subTest(mode=mode, profile=profile):
                    scope, calls = load_director()
                    args = self.kwargs()
                    args.update(edit_mode=mode, creative_profile=profile,
                                edit_request='Create a light comedy about two travelers at a rural station.',
                                direction_context='Wardrobe: Keep Reference\nEnvironment: Create New\nDialogue: Spanish',
                                llm=SimpleNamespace(model='test'), max_tokens=1200)
                    scope['H3CompactMultimodalEditDirector'].execute(**args)
                    director = scope['last_payload']
                    active = profile == 'Creative Plus'
                    self.assertEqual('CREATIVE PLUS REALIZATION' in director['messages'][0]['content'], active)
                    self.assertEqual(calls, ['external'] * (2 if active else 1))
                    self.assertEqual(director['max_tokens'], 1200)
                    self.assertIn('Keep Reference', director['messages'][0]['content'])
                    if not active:
                        continue
                    planner = scope['payloads'][0]
                    system = planner['messages'][0]['content']
                    for instruction in ('one central idea worth filming',
                                        'without returning alternatives',
                                        'Match the requested form', 'no forced palette',
                                        'Keep requested uniforms/signatures',
                                        'Details must be resolvable',
                                        'Honor deliberate wardrobe/location changes',
                                        'silence and source-audio locks remain protected'):
                        self.assertIn(instruction, system)
                    fields = planner['response_format']['json_schema']['schema']['properties']
                    self.assertEqual(set(fields), {'premise', 'cast_and_relationships',
                        'production_design', 'scene_beats', 'continuity_locks'})
                    self.assertTrue(all('description' in field for field in fields.values()))
                    self.assertLessEqual(planner['max_tokens'], 1600)
                    self.assertNotIn('image_url', json.dumps(planner))
                    for instruction in ('no replay or rewind', 'Resolve incompatible plan suggestions',
                                        'retaining required events and the existing output budget',
                                        'optional creative support', 'not a mandatory screenplay',
                                        'not transient acting poses'):
                        self.assertIn(instruction, director['messages'][0]['content'])

    def test_creative_plus_authorship_does_not_leak_into_faithful_modes(self):
        for mode in ('Transfer', 'edit', 'Continuous Edit', 'compact'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, creative_profile='Creative Plus',
                        llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            self.assertEqual(calls, ['external'])
            self.assertNotIn('CREATIVE PLUS REALIZATION', scope['last_payload']['messages'][0]['content'])
            self.assertNotIn('CREATIVE PLUS AUTHORSHIP', scope['last_payload']['messages'][0]['content'])

    def test_general_genres_have_specific_mechanisms_not_only_generic_fallback(self):
        scope, _ = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        tree = ast.parse((ROOT / 'story_director.py').read_text(encoding='utf-8'))
        genres = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                      and target.id == 'GENRES' for target in node.targets))
        general = genres[1:genres.index('Sensual Romance (Adults 18+)')]
        self.assertFalse(set(general) - set(cls.GENRE_MECHANISMS))
        self.assertEqual(len(set(cls.GENRE_MECHANISMS[name] for name in general)), len(general))
        for name, phrase in (('Crime', 'evidence'), ('Heist', 'security constraint'),
                             ('Science Fiction', 'speculative rule'), ('Anime', 'rendering grammar'),
                             ('Documentary', 'Do not invent factual events'),
                             ('Product Showcase', 'no fabricated specifications')):
            self.assertIn(phrase, cls.GENRE_MECHANISMS[name])

    def test_genre_look_motion_combine_in_one_treatment_and_reach_plus_planner(self):
        for profile in ('Standard', 'Creative Plus'):
            scope, calls = load_director()
            cls = scope['H3CompactMultimodalEditDirector']
            args = self.kwargs()
            args.update(edit_mode='Enhance', creative_profile=profile,
                        edit_request='The two travelers retrieve a missing suitcase. Keep their clothes.',
                        direction_context='Genre: Science Fiction + Heist\nVisual look: Cinematic\nMotion style: Fast Motion + Locked-Off Camera',
                        llm=SimpleNamespace(model='test'), max_tokens=1200)
            cls.execute(**args)
            director = scope['last_payload']
            system = director['messages'][0]['content']
            for genre in ('Science Fiction', 'Heist'):
                self.assertIn(cls.GENRE_MECHANISMS[genre], system)
            self.assertIn(scope['VISUAL_LOOKS']['Cinematic'], system)
            self.assertIn(scope['MOTION_STYLES']['Fast Motion'], system)
            for phrase in ('one causal situation', 'not alternating unrelated scenes',
                           'not story or wardrobe', "override a look's default camera",
                           'first-selection priority', 'Keep explicit reference/design locks'):
                self.assertIn(phrase.lower(), system.lower())
            self.assertEqual(director['max_tokens'], 1200)
            self.assertEqual(calls, ['external'] * (2 if profile == 'Creative Plus' else 1))
            if profile == 'Creative Plus':
                brief = json.loads(scope['payloads'][0]['messages'][1]['content'])
                resolved = '\n'.join(brief['resolved_direction_controls'])
                self.assertIn(cls.GENRE_MECHANISMS['Heist'], resolved)
                self.assertIn(scope['VISUAL_LOOKS']['Cinematic'], resolved)

    def test_period_looks_and_capture_grades_have_different_design_authority(self):
        for look, period in (('1940s Black-and-White Studio', True),
                             ('1960s Mod / Pop Cinema', True),
                             ('2000s Y2K Digital Camera', True),
                             ('VHS / Analog Tape', False),
                             ('Teal and Orange Blockbuster', False),
                             ('Dreamlike / Soft Focus', False)):
            with self.subTest(look=look):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode='Enhance', llm=SimpleNamespace(model='test'),
                            direction_context=f'Visual look: {look}\nWardrobe: Keep Reference')
                scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                self.assertIn(scope['VISUAL_LOOKS'][look], system)
                self.assertEqual('For an open period treatment, realize the era' in system, period)
                self.assertEqual('This is a capture/grade treatment' in system, not period)
                self.assertIn('An explicit Keep Reference beats genre restyling', system)
                self.assertEqual(calls, ['external'])

    def test_genre_story_mechanism_does_not_rewrite_faithful_transfer(self):
        scope, calls = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        args = self.kwargs()
        args.update(edit_mode='Transfer', llm=SimpleNamespace(model='test'), source_video_1=Frame(),
                    direction_context='Genre: Crime\nVisual look: Film Print Emulation')
        cls.execute(**args)
        system = scope['last_payload']['messages'][0]['content']
        self.assertNotIn(cls.GENRE_MECHANISMS['Crime'], system)
        self.assertIn('around the observed activity, not a new plot', system)
        self.assertIn('faithful source modes preserve choreography', system)
        self.assertEqual(calls, ['external'])

    def test_motion_controls_describe_distinct_body_and_camera_mechanics(self):
        scope, _ = load_director()
        motion = scope['MOTION_STYLES']
        for name, evidence in (
                ('Fast Motion', ('visibly faster than everyday pace', 'No slow motion',
                                 'not the event count, dialogue or playback', 'Respect a locked camera')),
                ('Super Fast Motion', ('explosive high-speed bursts', 'near-immediate launch',
                                       'add neither events nor cuts')),
                ('Slow Motion', ('visibly expanded time', 'not merely slow', 'do not stretch spoken words')),
                ('Fluid Motion', ('not necessarily slow', 'selected fast cadence')),
                ('Gimbal Camera', ('smooth does not mean slow', 'natural parallax')),
                ('Locked-Off Camera', ('position, orientation and focal length fixed', 'no pan, tilt')),
                ('Orbit Camera', ('named pivot', 'radius', 'background parallax', 'full 360 unless requested')),
                ('Crash Zoom Energy', ('optical focal length', 'not a dolly translation')),
                ('Dolly / Tracking Shot', ('Translate the camera', 'do not substitute optical zoom')),
                ('Whip Pan Energy', ('not a new shot/cut', 'readable landing'))):
            with self.subTest(name=name):
                for phrase in evidence:
                    self.assertIn(phrase, motion[name])

    def test_active_controls_reach_execution_without_canceling_independent_axes(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate', 'Transfer', 'edit'):
            for choices in ('Fast Motion + Locked-Off Camera', 'Locked-Off Camera + Fast Motion'):
                with self.subTest(mode=mode, choices=choices):
                    scope, calls = load_director()
                    args = self.kwargs()
                    args.update(edit_mode=mode, llm=SimpleNamespace(model='test'),
                                edit_request='The traveler crosses the station platform.',
                                direction_context='Motion style: ' + choices,
                                max_tokens=2500, temperature=0.35, seed=42)
                    scope['H3CompactMultimodalEditDirector'].execute(**args)
                    payload = scope['last_payload']
                    system = payload['messages'][0]['content']
                    rules = scope['H3CompactMultimodalEditDirector'].CONTROL_EXECUTION_RULES
                    self.assertEqual(system.count(rules), 1)
                    self.assertIn(scope['MOTION_STYLES']['Fast Motion'], system)
                    self.assertIn(scope['MOTION_STYLES']['Locked-Off Camera'], system)
                    for phrase in ('SAME axis', 'fixed or slow camera',
                                   'faithful source modes preserve choreography',
                                   'audio timing unless retiming is explicitly requested',
                                   'not labels', 'Do not add cuts'):
                        self.assertIn(phrase, system)
                    self.assertEqual(calls, ['external'])
                    self.assertEqual(payload['max_tokens'], 2500)
                    self.assertEqual(payload['temperature'], 0.35)
                    self.assertEqual(payload['seed'], 42)

    def test_plus_planner_gets_resolved_motion_specs_without_more_stages(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Enhance', creative_profile='Creative Plus',
                    direction_context='Motion style: Fast Motion + Orbit Camera',
                    llm=SimpleNamespace(model='test'))
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        planner, director = scope['payloads']
        brief = json.loads(planner['messages'][1]['content'])
        self.assertEqual(len(brief['resolved_direction_controls']), 2)
        for name in ('Fast Motion', 'Orbit Camera'):
            self.assertIn(scope['MOTION_STYLES'][name], '\n'.join(brief['resolved_direction_controls']))
            self.assertIn(scope['MOTION_STYLES'][name], director['messages'][0]['content'])
        self.assertIn('resolved_direction_controls defines the selected behavior',
                      planner['messages'][0]['content'])
        self.assertNotIn('Do not add cuts, new output sections', planner['messages'][0]['content'])
        self.assertNotIn('image_url', json.dumps(planner))
        self.assertEqual(calls, ['external', 'external'])
        self.assertEqual(director['max_tokens'], 1200)

    def test_inactive_motion_controls_add_no_execution_contract(self):
        for profile in ('Standard', 'Creative Plus'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode='Enhance', creative_profile=profile,
                        direction_context='Motion style: Auto + none', llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            for payload in scope['payloads']:
                self.assertNotIn('CONTROL EXECUTION — OBSERVABLE', payload['messages'][0]['content'])
            if profile == 'Creative Plus':
                self.assertNotIn('resolved_direction_controls',
                                 json.loads(scope['payloads'][0]['messages'][1]['content']))
            self.assertEqual(calls, ['external'] * (2 if profile == 'Creative Plus' else 1))

    def test_plus_story_development_is_shared_by_planning_and_realization_only(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
            for profile in ('Standard', 'Creative Plus'):
                with self.subTest(mode=mode, profile=profile):
                    scope, calls = load_director()
                    cls = scope['H3CompactMultimodalEditDirector']
                    args = self.kwargs()
                    args.update(edit_mode=mode, creative_profile=profile,
                                edit_request='Create a story: an office worker finishes her task, gets angry and leaves.',
                                llm=SimpleNamespace(model='test'))
                    cls.execute(**args)
                    active = profile == 'Creative Plus'
                    for payload in scope['payloads']:
                        self.assertEqual(payload['messages'][0]['content'].count(cls.CREATIVE_PLUS_STORY_RULES), int(active))
                    self.assertEqual(calls, ['external'] * (2 if active else 1))
                    self.assertEqual(scope['last_payload']['max_tokens'], 1200)
                    if active:
                        system = scope['last_payload']['messages'][0]['content']
                        self.assertGreater(system.index(cls.CREATIVE_PLUS_STORY_RULES),
                                           system.index('DRAMATIZE, DO NOT DEFER'))
                        for phrase in ('not a replacement premise', 'fulfill\nthe brief core early',
                                       'BECAUSE/THEREFORE', 'readable result',
                                       'not only preparation', 'not a script',
                                       'do not manufacture a threat', 'faithful edits',
                                       'Honor loops, supplied endings and source/audio locks'):
                            self.assertIn(phrase, cls.CREATIVE_PLUS_STORY_RULES)

    def test_plus_story_structure_keeps_scene_count_fields_and_budgets(self):
        for count in (1, 2, 3, 6):
            budgets = []
            for profile in ('Standard', 'Creative Plus'):
                scope, calls = load_director()
                cls = scope['H3CompactMultimodalEditDirector']
                args = self.kwargs()
                args.update(edit_mode='Continuous Elaborate', continuous_scene_count=count,
                            creative_profile=profile, llm=SimpleNamespace(model='test'))
                result = cls.execute(**args)
                self.assertEqual(len(json.loads(result[0])['scene_prompts']), count)
                budgets.append(scope['last_payload']['max_tokens'])
                if profile == 'Creative Plus':
                    planner = scope['payloads'][0]
                    fields = planner['response_format']['json_schema']['schema']['properties']
                    self.assertEqual(set(fields), {'premise', 'cast_and_relationships',
                        'production_design', 'scene_beats', 'continuity_locks'})
                    self.assertEqual(planner['max_tokens'], min(1600, 700 + 70 * count))
                    self.assertIn('Fulfill a brief open premise early', fields['scene_beats']['description'])
                    self.assertNotIn('image_url', json.dumps(planner))
            self.assertEqual(budgets[0], budgets[1])

    def test_plus_story_development_does_not_override_closed_script_or_source_evidence(self):
        scope, calls = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        request = 'Exact script: she enters, closes the door and sits. End there; no aftermath or extra dialogue.'
        args = self.kwargs()
        args.update(edit_mode='Enhance', creative_profile='Creative Plus', source_video_1=Frame(),
                    edit_request=request, llm=SimpleNamespace(model='test'))
        cls.execute(**args)
        planner, director = scope['payloads']
        brief = json.loads(planner['messages'][1]['content'])
        self.assertEqual(brief['request'], request)
        self.assertTrue(brief['source_performance_locked'])
        self.assertIn('Exact scripts, faithful edits and source chronology stay faithful',
                      director['messages'][0]['content'])
        self.assertIn('preserving the user brief, accepted canon', director['messages'][0]['content'])
        self.assertEqual(calls, ['external', 'external'])
        scope, calls = load_director()
        args['edit_mode'] = 'Transfer'
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertNotIn(cls.CREATIVE_PLUS_STORY_RULES, scope['last_payload']['messages'][0]['content'])
        self.assertEqual(calls, ['external'])

    def test_creative_plus_reuses_driven_evidence_and_reports_three_stages(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Enhance', creative_profile='Creative Plus', driven_video=Frame(),
                    llm=SimpleNamespace(model='test'), debug_request=True)
        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, ['external'] * 3)
        planner = scope['payloads'][1]
        brief = json.loads(planner['messages'][1]['content'])
        self.assertEqual(brief['observed_driven_blueprint']['setting'], 'Football sideline')
        self.assertNotIn('image_url', json.dumps(planner))
        self.assertIn('3 stages', result[2])
        self.assertIn('CREATIVE PLUS PLANNING:', result[4])

    def test_creative_plus_continuation_does_not_replan_and_survives_cold_resume(self):
        scope, calls = load_director()
        records = []
        scope['_set_held_director_plan'] = lambda key, record: records.append(record)
        args = self.kwargs()
        args.update(edit_mode='Enhance', creative_profile='Creative Plus', continuous_scene_count=1,
                    llm=SimpleNamespace(model='test'), edit_request='A reunion.\nSOURCE BLOCK 1/3: 0..10s.')
        cls = scope['H3CompactMultimodalEditDirector']
        first = cls.execute(**args)
        anchor = json.loads(first[0])['scene_prompts'][0]
        self.assertEqual(records[-1]['creative_anchor'], anchor)
        scope['_get_held_director_plan'] = lambda key: records[-1]
        args['edit_request'] = 'A reunion.\nSOURCE BLOCK 2/3: 10..20s.\nFIRST ACCEPTED BLOCK:\n' + anchor
        second = cls.execute(**args)
        self.assertEqual(calls, ['external'] * 3)
        self.assertIn('treatment reused', second[1])
        self.assertEqual(scope['last_payload']['messages'][0]['content'].count(cls.CREATIVE_PLUS_STORY_RULES), 1)
        self.assertIn('no planning call', second[2])
        self.assertIn('A hesitant reunion', json.dumps(scope['last_payload']))
        scope['_get_held_director_plan'] = lambda key: None
        args['edit_request'] = ('A reunion.\nSOURCE BLOCK 3/3: 20..30s.\nFIRST ACCEPTED BLOCK:\n'
                                + anchor + '\nPREVIOUS ACCEPTED BLOCK:\nContinued conversation.')
        third = cls.execute(**args)
        self.assertEqual(calls, ['external'] * 4)
        self.assertIn('no planning call', third[2])
        self.assertIn('accepted_first_block', json.dumps(scope['last_payload']))

    def test_creative_profile_hold_and_bypass_never_add_paid_calls(self):
        scope, calls = load_director()
        records = []
        scope['_set_held_director_plan'] = lambda key, record: records.append(record)
        args = self.kwargs()
        args.update(edit_mode='Enhance', creative_profile='Creative Plus', llm=SimpleNamespace(model='test'))
        cls = scope['H3CompactMultimodalEditDirector']
        cls.execute(**args)
        scope['_get_held_director_plan'] = lambda key: records[-1]
        args['hold_prompt'] = True
        cls.execute(**args)
        self.assertEqual(len(calls), 2)
        args['creative_profile'] = 'Standard'
        with self.assertRaisesRegex(RuntimeError, 'Creative Profile changed'):
            cls.execute(**args)
        self.assertEqual(len(calls), 2)
        args.update(bypass=True, llm=None)
        cls.execute(**args)
        self.assertEqual(len(calls), 2)

    def test_creative_plus_invalid_json_continues_with_standard_craft_without_planning_retry(self):
        scope, calls = load_director()
        original_request = scope['_external_llm_request']
        logs = []
        scope['print'] = logs.append
        def broken(provider, payload):
            if payload['response_format']['json_schema']['name'] == 'h3_creative_plus_treatment':
                calls.append('external')
                return {'choices': [{'message': {'content': '{"premise":"private unfinished'},
                                     'finish_reason': 'length'}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 50}}
            return original_request(provider, payload)
        scope['_external_llm_request'] = broken
        args = self.kwargs()
        args.update(edit_mode='Enhance', creative_profile='Creative Plus', llm=SimpleNamespace(model='test'))
        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, ['external', 'external'])
        self.assertIn('treatment fallback', result[1])
        self.assertIn('planning input: 100 / output: 50', result[2])
        self.assertIn('2 stages', result[2])
        self.assertNotIn('no planning call', result[2])
        system = scope['last_payload']['messages'][0]['content']
        self.assertIn(scope['H3CompactMultimodalEditDirector'].CREATIVE_TREATMENT_RULES.format(candidates=3), system)
        self.assertNotIn('CREATIVE PLUS EXECUTION', system)
        self.assertIn(scope['H3CompactMultimodalEditDirector'].CREATIVE_PLUS_STORY_RULES, system)
        self.assertIn('finish_reason=length', logs[0])
        self.assertIn('line 1, column', logs[0])
        self.assertNotIn('private', logs[0])

    def test_creative_plus_invalid_treatment_reports_fields_without_content_or_retry(self):
        valid = dict(premise='private scene text', production_design='private design',
                     cast_and_relationships=['lead'], scene_beats=['one', 'two'], continuity_locks=[])
        for changes, expected in (
                ({'premise': None}, 'premise: expected string, got NoneType'),
                ({'cast_and_relationships': [{'name': 'private identity'}]},
                 'cast_and_relationships: expected string items'),
                ({'continuity_locks': 'private lock'}, 'continuity_locks: expected array, got str'),
                ({}, 'production_design: missing')):
            with self.subTest(expected=expected):
                scope, calls = load_director()
                logs = []
                scope['print'] = logs.append
                treatment = dict(valid, **changes)
                if not changes:
                    treatment.pop('production_design')
                def broken(provider, payload):
                    calls.append('external')
                    return {'choices': [{'message': {'content': json.dumps(treatment)}}]}
                scope['_external_llm_request'] = broken
                plan, _, _ = scope['H3CompactMultimodalEditDirector']._plan_creative_plus(
                    SimpleNamespace(model='test'), {}, 2, 10, 0.35, 42)
                self.assertEqual(plan, {})
                self.assertIn(expected, logs[0])
                self.assertNotIn('private', logs[0])
                self.assertEqual(calls, ['external'])

    def test_creative_plus_repairs_only_local_json_syntax_without_changing_strings(self):
        valid = dict(premise='Literal punctuation ,] and ,} with "quotes" and \\slashes.',
                     production_design='Warm light.\nRainy street.', cast_and_relationships=['lead'],
                     scene_beats=['setup', 'resolution'], continuity_locks=[])
        encoded = json.dumps(valid)
        for raw in (encoded.replace('"lead"]', '"lead",]')[:-1] + ',}',
                    encoded.replace('Warm light.\\nRainy street.', 'Warm light.\nRainy street.')):
            scope, calls = load_director()
            scope['print'] = lambda text: None
            def reply(provider, payload):
                calls.append('external')
                return {'choices': [{'message': {'content': raw}, 'finish_reason': 'stop'}]}
            scope['_external_llm_request'] = reply
            plan, _, _ = scope['H3CompactMultimodalEditDirector']._plan_creative_plus(
                SimpleNamespace(model='test'), {}, 2, 10, 0.35, 42)
            self.assertEqual(plan, valid)
            self.assertEqual(calls, ['external'])

    def test_creative_plus_does_not_hide_provider_failures(self):
        scope, calls = load_director()
        def unavailable(provider, payload):
            calls.append('external')
            raise RuntimeError('HTTP 402: insufficient balance')
        scope['_external_llm_request'] = unavailable
        with self.assertRaisesRegex(RuntimeError, 'HTTP 402'):
            scope['H3CompactMultimodalEditDirector']._plan_creative_plus(
                SimpleNamespace(model='test'), {}, 2, 10, 0.35, 42)
        self.assertEqual(calls, ['external'])

    def test_creative_plus_beat_count_does_not_override_requested_scene_count(self):
        for beats in ([], ['setup', 'resolution'], ['one', 'two', 'three', 'four']):
            with self.subTest(beats=beats):
                scope, calls = load_director()
                original_request = scope['_external_llm_request']
                def reply(provider, payload):
                    result = original_request(provider, payload)
                    if payload['response_format']['json_schema']['name'] == 'h3_creative_plus_treatment':
                        treatment = json.loads(result['choices'][0]['message']['content'])
                        treatment['scene_beats'] = beats
                        result['choices'][0]['message']['content'] = json.dumps(treatment)
                    return result
                scope['_external_llm_request'] = reply
                args = self.kwargs()
                args.update(edit_mode='Continuous Elaborate', creative_profile='Creative Plus',
                            continuous_scene_count=3, llm=SimpleNamespace(model='test'))
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                self.assertEqual(len(json.loads(result[0])['scene_prompts']), 3)
                self.assertEqual(calls, ['external', 'external'])
                planner, director = scope['payloads']
                self.assertIn('redistribute beats', director['messages'][0]['content'])
                self.assertIn('Earn the final', director['messages'][0]['content'])
                notes = next(m['text'] for m in director['messages'][1]['content']
                             if m.get('type') == 'text' and m['text'].startswith('CREATIVE PLUS TREATMENT'))
                self.assertEqual(json.loads(notes.split('\n', 1)[1])['scene_beats'], beats)

    def test_body_controls_are_scoped_independently_in_all_modes(self):
        for mode in ('compact', 'edit', 'Transfer', 'Enhance', 'Elaborate',
                     'Continuous Edit', 'Continuous Elaborate'):
            for selection in ('Auto', 'Keep Reference', 'Creative Adaptation'):
                with self.subTest(mode=mode, selection=selection):
                    scope, calls = load_director()
                    args = self.kwargs()
                    args.update(edit_mode=mode, llm=SimpleNamespace(model='test'),
                                edit_request='The character walks across a courtyard.',
                                direction_context=f'Body proportions: {selection}')
                    result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                    payload = scope['last_payload']
                    system = payload['messages'][0]['content']
                    active = selection != 'Auto'
                    self.assertEqual('BODY PROPORTIONS — INDEPENDENT DESIGN SCOPE' in system, active)
                    self.assertNotIn('WARDROBE / ENVIRONMENT — EXPLICIT DESIGN SCOPE', system)
                    props = payload['response_format']['json_schema']['schema']['properties']
                    if mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
                        subject = props['target_design']['properties']['subjects']['items']
                        self.assertEqual('target_body' in subject['properties'], active)
                        self.assertNotIn('wardrobe_contrast', subject['properties'])
                        if active:
                            self.assertIn('Body proportions: an athletic build', result[0])
                    if active:
                        for contract in ('I2V frame-zero facts win', 'not automatic exaggeration',
                                         'Keep pose, action timing, hand contacts', 'across scenes/blocks',
                                         'not contradictory source proportions'):
                            self.assertIn(contract, system)
                    self.assertEqual(calls, ['external'])
                    self.assertEqual(payload['max_tokens'], 1800 if mode == 'edit' else 1200)

    def test_body_and_new_outfit_compile_together_without_new_analysis_stage(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Enhance', llm=SimpleNamespace(model='test'), driven_video=Frame(),
                    direction_context='Wardrobe: Create New\nBody proportions: Creative Adaptation')
        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
        props = scope['last_payload']['response_format']['json_schema']['schema']['properties']
        subject = props['target_design']['properties']['subjects']['items']
        self.assertIn('target_body', subject['properties'])
        self.assertIn('target_body', subject['required'])
        self.assertIn('wardrobe_contrast', subject['properties'])
        self.assertIn('Body proportions: an athletic build', result[0])
        self.assertIn('Target appearance: wearing a blue jacket', result[0])
        self.assertEqual(calls, ['external', 'external'])
        self.assertNotIn('BODY PROPORTIONS', scope['payloads'][0]['messages'][0]['content'])

    def test_new_design_checks_are_conditional_and_reuse_the_current_call(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
            for wardrobe, environment in (('Auto', 'Auto'), ('Create New', 'Keep Reference'),
                                         ('Keep Reference', 'Create New'), ('Create New', 'Create New')):
                with self.subTest(mode=mode, wardrobe=wardrobe, environment=environment):
                    scope, calls = load_director()
                    args = self.kwargs()
                    args.update(edit_mode=mode, llm=SimpleNamespace(model='test'),
                                edit_request='A couple talks over drinks in a bar. Keep her white scarf.',
                                direction_context=f'Wardrobe: {wardrobe}\nEnvironment: {environment}')
                    scope['H3CompactMultimodalEditDirector'].execute(**args)
                    payload = scope['last_payload']
                    props = payload['response_format']['json_schema']['schema']['properties']
                    subject = props['target_design']['properties']['subjects']['items']
                    scene = props['edit_prompt'] if mode == 'Elaborate' else props['scene_prompts']['items']
                    self.assertEqual('wardrobe_contrast' in subject['properties'], wardrobe == 'Create New')
                    self.assertEqual('environment' in scene['properties'], environment == 'Create New')
                    if wardrobe == 'Create New':
                        self.assertLess(list(subject['properties']).index('wardrobe_contrast'),
                                        list(subject['properties']).index('target_appearance'))
                        self.assertIn('Explicit user garment locks/I2V opening facts win', payload['messages'][0]['content'])
                    if environment == 'Create New':
                        self.assertEqual(list(scene['properties'])[0], 'environment')
                        self.assertIn('within the user\'s venue category', payload['messages'][0]['content'])
                    self.assertEqual(calls, ['external'])
                    self.assertEqual(payload['max_tokens'], 1200)

    def test_redesigned_wardrobe_and_environment_compile_without_the_source_outfit_or_audit(self):
        scope, calls = load_director()
        environment = 'An Art Deco cocktail bar: curved walnut booths flank a brass-edged table; amber glass lamps light smoked-mirror shelves behind.'
        def reply(provider, payload):
            calls.append('external')
            design = dict(subjects=[
                dict(subject_tag='<Subject 1>', identity='the woman fully referenced from <Picture 1>.',
                     wardrobe_contrast='Striped black top -> rust velvet blazer and cream blouse.',
                     target_appearance='rust velvet blazer over a cream blouse, charcoal pleated trousers, ankle boots.'),
                dict(subject_tag='<Subject 2>', identity='the man fully referenced from <Picture 2>.',
                     wardrobe_contrast='Black T-shirt -> sage knit polo and brown wool trousers.',
                     target_appearance='sage knit polo, brown wool trousers, leather loafers.')],
                visual_style='Warm directional light and restrained amber highlights.')
            scene = dict(environment=environment, opening_framing='An eye-level two-shot across the booth.',
                         integrated_multimodal_description='The couple share a smile and raise their glasses.',
                         visible_result='Their glasses meet and return to the table.',
                         overall_soundscape='Glasses clink softly.', non_diegetic_music='N/A')
            return {'choices': [{'message': {'content': json.dumps(dict(target_design=design, scene_prompts=[scene, scene]))}}]}
        scope['_external_llm_request'] = reply
        args = self.kwargs()
        args.update(edit_mode='Enhance', llm=SimpleNamespace(model='test'), reference_image_2=Frame(),
                    edit_request='A couple chats over drinks in a bar.',
                    direction_context='Wardrobe: Create New\nEnvironment: Create New')
        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
        for prompt in json.loads(result[0])['scene_prompts']:
            self.assertIn('rust velvet blazer', prompt)
            self.assertIn('sage knit polo', prompt)
            self.assertIn('identity referenced from <Picture 1>', prompt)
            self.assertIn(environment, prompt)
            self.assertNotIn('Striped black top', prompt)
            self.assertNotIn('Black T-shirt', prompt)
            self.assertNotIn('wardrobe_contrast', prompt)
            for heading in ('subject_definitions:', 'integrated_multimodal_description:',
                            'overall_soundscape:', 'non_diegetic_music:'):
                self.assertEqual(prompt.count(heading), 1)
        self.assertEqual(calls, ['external'])

    def test_redesign_renderer_preserves_explicit_locks_and_old_designs(self):
        scope, _ = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        scene = dict(integrated_multimodal_description='She walks into the bar.')
        person = dict(subject_tag='<Subject 1>', identity='the woman fully referenced from <Picture 1>.',
                      target_appearance='her original blue coat.')
        original = cls._render_creative_scene(scene, dict(subjects=[person]))
        self.assertIn('fully referenced from <Picture 1>', original)
        person['wardrobe_contrast'] = 'N/A'
        self.assertEqual(cls._render_creative_scene(scene, dict(subjects=[person])), original)

    def test_design_control_widgets_append_without_shifting_existing_widgets(self):
        tree = ast.parse((ROOT / 'story_director.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'H3CompactDirectionControls')
        scope = dict(io=SimpleNamespace(
            ComfyNode=object, NodeOutput=lambda *v: v, Schema=lambda **kw: kw,
            Combo=SimpleNamespace(Input=lambda name, **kw: dict(name=name, **kw)),
            String=SimpleNamespace(Output=lambda name: name)),
            GENRES=[], MOTION_STYLES={}, VISUAL_LOOKS={}, DIALOGUE_OPTIONS=[])
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'controls', 'exec'), scope)
        controls = scope['H3CompactDirectionControls']
        fields = controls.define_schema()['inputs']
        self.assertEqual([f['name'] for f in fields], [
            'genre_1', 'genre_2', 'motion_style_1', 'motion_style_2',
            'look_1', 'look_2', 'dialogue', 'wardrobe', 'environment', 'body_proportions'])
        for field in fields[7:9]:
            self.assertEqual(field['options'], ['Auto', 'Keep Reference', 'Create New'])
            self.assertEqual(field['default'], 'Auto')
        self.assertEqual(fields[-1]['options'], ['Auto', 'Keep Reference', 'Creative Adaptation'])
        self.assertEqual(fields[-1]['default'], 'Auto')
        self.assertEqual(controls.execute()[0], '')
        self.assertEqual(controls.execute(wardrobe='Auto', environment='Auto')[0], '')
        self.assertEqual(controls.execute(wardrobe='Create New', environment='Keep Reference')[0],
                         'Wardrobe: Create New\nEnvironment: Keep Reference')
        self.assertEqual(controls.execute(body_proportions='Auto')[0], '')
        self.assertEqual(controls.execute(body_proportions='Creative Adaptation')[0],
                         'Body proportions: Creative Adaptation')

    def test_written_shot_requests_reach_creative_modes_without_automatic_split(self):
        request = 'Create a reunion. Cut to a reverse angle at five seconds; speak each line only once.'
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
            for profile in ('Standard', 'Creative Plus'):
                with self.subTest(mode=mode, profile=profile):
                    scope, calls = load_director()
                    args = self.kwargs()
                    args.update(edit_mode=mode, creative_profile=profile, edit_request=request,
                                llm=SimpleNamespace(model='test'))
                    scope['H3CompactMultimodalEditDirector'].execute(**args)
                    director = scope['last_payload']
                    self.assertIn(request, json.dumps(director['messages'], ensure_ascii=False))
                    props = director['response_format']['json_schema']['schema']['properties']
                    scene = props['edit_prompt'] if mode == 'Elaborate' else props['scene_prompts']['items']
                    self.assertNotIn('second_shot', scene['properties'])
                    self.assertNotIn('TWO-SHOT COMPILATION', director['messages'][0]['content'])
                    self.assertNotIn('MID-SCENE SHOT CHANGE', director['messages'][0]['content'])
                    if profile == 'Creative Plus':
                        brief = json.loads(scope['payloads'][0]['messages'][1]['content'])
                        self.assertEqual(brief['request'], request)
                        self.assertNotIn('scene_coverage', brief)
                    self.assertEqual(calls, ['external'] * (2 if profile == 'Creative Plus' else 1))

    def test_renderer_preserves_user_directed_shots_without_inserting_midpoint(self):
        scope, _ = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        scene = dict(opening_framing='[Shot 1] Wide view.',
                     integrated_multimodal_description=(
                         '[Shot 1] She approaches. [Shot 2] At 00:03.250, '
                         '<Subject 1> (S1) says: <d>[Spanish] Hola.</d> '
                         '[Shot 3] At 00:07.000, He answers with a smile.'),
                     visible_result='They sit down.', overall_soundscape='Room tone.',
                     non_diegetic_music='N/A')
        for sequence in (False, True):
            result = cls._compile_scene_prompt(cls._render_creative_scene(scene, {}), sequence=sequence)
            self.assertEqual(re.findall(r'\[Shot\s+\d+\]', result),
                             ['[Shot 1]', '[Shot 2]', '[Shot 3]'])
            self.assertIn('[Shot 2] At 00:03.250,', result)
            self.assertIn('[Shot 3] At 00:07.000,', result)
            self.assertNotIn('00:05.000', result)
            self.assertEqual(result.count('<d>[Spanish] Hola.</d>'), 1)
        scene['integrated_multimodal_description'] = 'They talk without cutting.'
        result = cls._render_creative_scene(scene, {})
        self.assertNotIn('[Shot 2]', result)

    def test_plus_editorial_contract_reaches_both_stages_only_in_creative_plus(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate', 'Transfer', 'Continuous Edit'):
            for profile in ('Standard', 'Creative Plus'):
                with self.subTest(mode=mode, profile=profile):
                    scope, calls = load_director()
                    cls = scope['H3CompactMultimodalEditDirector']
                    args = self.kwargs()
                    args.update(edit_mode=mode, creative_profile=profile,
                        edit_request='Tell the journey of a student from university to her home.',
                        llm=SimpleNamespace(model='test'))
                    result = cls.execute(**args)
                    active = profile == 'Creative Plus' and mode in ('Enhance', 'Elaborate', 'Continuous Elaborate')
                    for payload in scope['payloads']:
                        self.assertEqual(payload['messages'][0]['content'].count(cls.CREATIVE_PLUS_EDITORIAL_RULES), int(active))
                    props = scope['last_payload']['response_format']['json_schema']['schema']['properties']
                    scene = props['edit_prompt'] if mode == 'Elaborate' else props['scene_prompts']['items']
                    self.assertNotIn('shots', scene.get('properties', {}))
                    if active:
                        self.assertIn('opening_framing', scene['properties'])
                        self.assertIn('integrated_multimodal_description', scene['properties'])
                        self.assertEqual(set(scene['required']), {'opening_framing',
                            'integrated_multimodal_description', 'visible_result',
                            'overall_soundscape', 'non_diegetic_music'})
                        self.assertNotIn('ONE-PASS LOGICAL CONTINUITY FOR H3 CHAINS',
                                         scope['last_payload']['messages'][0]['content'])
                        self.assertNotIn('start_seconds', result[0])
                    self.assertEqual(calls, ['external'] * (2 if active else 1))

    def test_plus_shot_renderer_separates_locations_and_does_not_repeat_dialogue(self):
        scope, _ = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        scene = dict(opening_framing='Wide tracking view along the university corridor.',
            integrated_multimodal_description=(
                '[Shot 1] <Subject 1> closes her notebook and approaches the exit.\n'
                '[Shot 2] At 00:03.275, hard cut. Medium side view inside a moving bus, with seat rows and passing street light. '
                '<Subject 1> sits by the window, notebook on her lap. <Subject 1> (S1) says: <d>[Spanish] Ya voy.</d>\n'
                '[Shot 3] At 00:07.125, hard cut. Wide view of her home entrance at dusk, brick walls and a narrow doorstep. '
                '<Subject 1> unlocks the front door, carrying the same notebook.'),
            visible_result='She steps into the house.', overall_soundscape='Footsteps, then bus hum, then her key turning.',
            non_diegetic_music='One soft score bridges the cuts.')
        design = dict(subjects=[dict(subject_tag='<Subject 1>', identity='the woman from <Picture 1>.',
                                    target_appearance='White blouse and fitted blue trousers.')])
        original = json.dumps(scene)
        for sequence in (False, True):
            prompt = cls._compile_scene_prompt(cls._render_creative_scene(scene, design), sequence=sequence)
            self.assertEqual(re.findall(r'\[Shot \d+\]', prompt), ['[Shot 1]', '[Shot 2]', '[Shot 3]'])
            self.assertIn('[Shot 2] At 00:03.275, hard cut.', prompt)
            self.assertIn('[Shot 3] At 00:07.125, hard cut.', prompt)
            self.assertEqual(prompt.count('Ya voy.'), 1)
            self.assertEqual(prompt.count('White blouse and fitted blue trousers.'), 1)
            self.assertLess(prompt.index('university corridor'), prompt.index('[Shot 2]'))
            self.assertLess(prompt.index('[Shot 2]'), prompt.index('inside a moving bus'))
            self.assertLess(prompt.index('[Shot 3]'), prompt.index('home entrance'))
            self.assertEqual(prompt.count('She steps into the house.'), 1)
            self.assertNotIn('00:05.000', prompt)
        self.assertEqual(json.dumps(scene), original)

    def test_plus_one_shot_and_i2v_alignment_do_not_create_a_cut(self):
        scope, _ = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        scene = dict(opening_framing='[Shot 1] A continuous close-up.',
                     integrated_multimodal_description='[Shot 1] She considers the letter, then smiles.',
                     visible_result='She folds the letter.', overall_soundscape='Paper rustle.', non_diegetic_music='N/A')
        prompt = cls._render_creative_scene(scene, {}, opening_frame=True)
        self.assertIn('at 0.00 seconds into the target video, <Picture 1>', prompt)
        self.assertEqual(prompt.count('[Shot 1]'), 2)  # Alignment binding plus actual footage.
        self.assertNotIn('[Shot 2]', prompt)
        self.assertNotIn('hard cut', prompt)

    def test_plus_edited_response_compiles_through_execute_without_extra_requests(self):
        scope, calls = load_director()
        original = scope['_external_llm_request']
        def edited(provider, payload):
            response = original(provider, payload)
            if payload['response_format']['json_schema']['name'] == 'h3_creative_plus_treatment':
                return response
            content = json.loads(response['choices'][0]['message']['content'])
            for scene in content['scene_prompts']:
                scene['opening_framing'] = 'Wide view of the university exit.'
                scene['integrated_multimodal_description'] = (
                    '<Subject 1> walks out holding her notebook.\n'
                    '[Shot 2] At 00:03.400, hard cut. Medium view inside a bus with blue seats. '
                    '<Subject 1> sits down, keeping the notebook on her lap.')
                scene['visible_result'] = 'She watches the street pass the bus window.'
            response['choices'][0]['message']['content'] = json.dumps(content)
            return response
        scope['_external_llm_request'] = edited
        args = self.kwargs()
        args.update(edit_mode='Continuous Elaborate', creative_profile='Creative Plus',
                    continuous_scene_count=1, seconds_per_scene=10, llm=SimpleNamespace(model='test'))
        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
        prompts = json.loads(result[0])['scene_prompts']
        self.assertEqual(len(prompts), 1)
        self.assertEqual(re.findall(r'\[Shot \d+\]', prompts[0]), ['[Shot 1]', '[Shot 2]'])
        self.assertIn('[Shot 2] At 00:03.400, hard cut. Medium view inside a bus', prompts[0])
        self.assertNotIn('start_seconds', prompts[0])
        self.assertEqual(calls, ['external', 'external'])

    def test_plus_editing_protects_continuous_source_and_inherited_prefix(self):
        scope, _ = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        for request in ('One continuous take, no cuts. She walks home.',
                        'Preserve the source video choreography, camera, dialogue and timing.'):
            args = self.kwargs()
            args.update(edit_mode='Continuous Elaborate', creative_profile='Creative Plus',
                        edit_request=request, llm=SimpleNamespace(model='test'))
            cls.execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            self.assertIn('single take/no cuts, locked source coverage', system)
            self.assertIn('after the protected prefix, never at frame zero', system)
            self.assertIn('Hold a take when uninterrupted acting', system)
            self.assertIn('Cut between lines/breaths rather than splitting a word', system)
            self.assertIn('Inherited AV/prefix footage owns the opening', system)
            self.assertIn(request, json.dumps(scope['last_payload']['messages']))

    def test_design_choices_reach_all_modes_without_extra_calls(self):
        for mode in ('compact', 'edit', 'Transfer', 'Enhance', 'Elaborate',
                     'Continuous Edit', 'Continuous Elaborate'):
            with self.subTest(mode=mode):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, llm=SimpleNamespace(model='test'),
                            direction_context='Wardrobe: Create New\nEnvironment: Keep Reference')
                scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                self.assertIn('ACTIVE DESIGN CHOICES:\nWardrobe: Create New\nEnvironment: Keep Reference', system)
                self.assertIn('unselected choices are definitions, not instructions', system)
                self.assertIn('Written user assignments and I2V frame-zero facts win', system)
                self.assertIn('An explicit Keep Reference beats genre restyling', system)
                self.assertIn('direction_controls with the complete resulting requested_wardrobe', system)
                self.assertIn('NOT new clothes/locations at every block', system)
                self.assertEqual(calls, ['external'])
                self.assertEqual(scope['last_payload']['max_tokens'], 1800 if mode == 'edit' else 1200)

    def test_design_choices_do_not_change_auto_or_driven_analysis(self):
        scope, _ = load_director()
        args = self.kwargs()
        args.update(edit_mode='Transfer', driven_video=Frame(), llm=SimpleNamespace(model='test'))
        cls = scope['H3CompactMultimodalEditDirector']
        cls.execute(**args)
        analysis = scope['payloads'][0]
        self.assertNotIn('EXPLICIT DESIGN SCOPE', scope['last_payload']['messages'][0]['content'])
        scope['payloads'] = []
        args['direction_context'] = 'Wardrobe: Create New\nEnvironment: Create New'
        cls.execute(**args)
        self.assertEqual(analysis, scope['payloads'][0])
        self.assertEqual(len(scope['payloads']), 2)
        self.assertIn('EXPLICIT DESIGN SCOPE', scope['last_payload']['messages'][0]['content'])

    def test_create_new_supports_controls_only_elaborate_and_hold_is_invalidated(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Elaborate', edit_request='', llm=SimpleNamespace(model='test'),
                    direction_context='Wardrobe: Create New\nEnvironment: Create New')
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(calls, ['external'])
        self.assertIn('ACTIVE DESIGN CHOICES', scope['last_payload']['messages'][0]['content'])
        saved = []
        scope['_set_held_director_plan'] = lambda key, record: saved.append(record)
        args.update(edit_request='Create a scene.', edit_mode='Enhance')
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        scope['_get_held_director_plan'] = lambda key: saved[-1]
        args.update(hold_prompt=True, direction_context='Wardrobe: Keep Reference')
        before = len(calls)
        with self.assertRaisesRegex(RuntimeError, 'Direction Controls changed'):
            scope['H3CompactMultimodalEditDirector'].execute(**args)
        self.assertEqual(len(calls), before)

    def test_transfer_ownership_separates_motion_camera_and_appearance_without_new_calls(self):
        for source in ('source_video_1', 'driven_video'):
            with self.subTest(source=source):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode='Transfer', llm=SimpleNamespace(model='test'),
                            edit_request='Transfer her movement, keep the image outfit, and change the setting.')
                args[source] = Frame()
                scope['H3CompactMultimodalEditDirector'].execute(**args)
                payload = scope['last_payload']
                system = payload['messages'][0]['content']
                self.assertIn('TRANSFER OWNERSHIP — INDEPENDENT AXES', system)
                for instruction in ('initial position/visible role', 'through crossings or occlusion',
                                    'Camera-only transfer', 'Body-only transfer',
                                    'not automatic beautification', 'never a guess from sampled lips',
                                    'Do not infer a new camera', 'active Direction Controls within their scope'):
                    self.assertIn(instruction, system)
                self.assertEqual(calls, ['external'] * (2 if source == 'driven_video' else 1))
                self.assertEqual(payload['max_tokens'], 1200)
                if source == 'driven_video':
                    self.assertNotIn('TRANSFER OWNERSHIP',
                                     scope['payloads'][0]['messages'][0]['content'])

    def test_story_canon_and_performance_craft_stay_in_creative_modes(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate',
                     'compact', 'edit', 'Continuous Edit', 'Transfer'):
            with self.subTest(mode=mode):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, llm=SimpleNamespace(model='test'),
                            edit_request='Create a quiet comic scene with this woman.')
                scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                creative = mode in ('Enhance', 'Elaborate', 'Continuous Elaborate')
                self.assertEqual('tiny narrative canon' in system, creative)
                self.assertEqual('TRANSFER OWNERSHIP' in system, mode == 'Transfer')
                if creative:
                    for instruction in ('never infer a real person', 'update the emotional state',
                                        'Keep exact supplied words separate from delivery',
                                        'outside <d> tags', 'without mouthing the other line',
                                        'details resolvable at this framing',
                                        'not a physics checklist', 'not a posing hold',
                                        'no forced conflict'):
                        self.assertIn(instruction, system)
                    self.assertIn('No additional bible, interview or output fields', system)
                self.assertEqual(calls, ['external'])
                self.assertEqual(scope['last_payload']['max_tokens'], 1800 if mode == 'edit' else 1200)

    def test_craft_rules_keep_existing_detail_budgets_and_no_platform_templates(self):
        for mode, ceiling in (('Transfer', 1200), ('Enhance', 3500),
                              ('Elaborate', 4800), ('Continuous Elaborate', 5600)):
            with self.subTest(mode=mode):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, llm=SimpleNamespace(model='test'))
                scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                cast_rules = scope['H3CompactMultimodalEditDirector'].CAST_IDENTITY_RULES
                self.assertLess(len(cast_rules.split()), 100)
                self.assertEqual(system.count(cast_rules), 1)
                self.assertLess(len(system.replace(cast_rules, '').split()), ceiling)
                for platform_rule in ('<<<image_', '@image1', 'PHOTOREAL FLAT CLOSE',
                                      'THE SPINE (LOCKED ORDER)', 'Age-blind'):
                    self.assertNotIn(platform_rule, system)
                self.assertEqual(calls, ['external'])
                self.assertEqual(scope['last_payload']['max_tokens'], 1200)

    def test_transfer_cast_compiles_only_the_selected_outfit(self):
        scope, _ = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        person = dict(subject_tag='<Subject 1>', identity_source='<Picture 1>',
            identity_details='the woman with dark hair', performance_source='<Video 1>',
            performance_role='woman seated on the bed', reference_wardrobe='Black strapless top and trousers',
            performance_wardrobe='White long-sleeve crop top and white socks',
            wardrobe_source='performance', requested_wardrobe='Incorrect black outfit')
        for owner, expected in (
            ('performance', person['performance_wardrobe']),
            ('reference', person['reference_wardrobe']),
            ('user_request', 'Red office suit'),
            ('direction_controls', 'Cyberpunk jacket'),
            ('established_design', 'Established blue suit'),
        ):
            person['wardrobe_source'] = owner
            if owner not in ('reference', 'performance'):
                person['requested_wardrobe'] = expected
            definition = cls._render_transfer_cast([person], {})
            self.assertIn('Wardrobe: ' + expected, definition)
            self.assertIn('Identity referenced from <Picture 1>', definition)
            if owner == 'performance':
                self.assertNotIn('Black strapless', definition)
                self.assertNotIn('Incorrect black', definition)
        person.update(wardrobe_source='performance', performance_source='<Driven Video 1>',
            performance_wardrobe='Wrongly copied reference clothing')
        definition = cls._render_transfer_cast([person], {'roles': [dict(
            role=person['performance_role'], wardrobe='Observed white video outfit')]})
        self.assertIn('Wardrobe: Observed white video outfit', definition)
        self.assertNotIn('Wrongly copied', definition)

    def test_transfer_cast_contract_covers_video_and_driven_without_extra_calls(self):
        for source in ('source_video_1', 'driven_video'):
            for strength in (1, 2, 3):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode='Transfer', driven_strength=strength, llm=SimpleNamespace(model='test'),
                    edit_request='transfiere a la mujer del video por la mujer de la imagen, se mantiene vestuario y acciones del video')
                args[source] = Frame()
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                payload = scope['last_payload']
                props = payload['response_format']['json_schema']['schema']['properties']
                self.assertEqual(next(iter(props)), 'transfer_cast')
                system = payload['messages'][0]['content']
                self.assertIn('se mantiene vestuario y acciones del video', system)
                self.assertIn('even when image identity predominates or strength 1', system)
                self.assertNotIn('COMPILE THE REFERENCE APPEARANCE', system)
                if source == 'source_video_1':
                    self.assertNotIn('subject_definitions', props['scene_prompts']['items']['properties'])
                for prompt in json.loads(result[0])['scene_prompts']:
                    self.assertEqual(prompt.count('subject_definitions:'), 1)
                    self.assertIn('Wardrobe: Team uniform and trainers', prompt)
                    self.assertNotIn('transfer_cast', prompt)
                    self.assertNotIn('Blue jacket', prompt)
                self.assertEqual(calls, ['external'] if source == 'source_video_1' else ['external', 'external'])

    def test_transfer_replaces_conflicting_driven_subject_definition(self):
        scope, _ = load_director()
        original = scope['_external_llm_request']
        def reply(provider, payload):
            result = original(provider, payload)
            if 'transfer_cast' in payload['response_format']['json_schema']['schema']['properties']:
                parsed = json.loads(result['choices'][0]['message']['content'])
                parsed['scene_prompts'] = [
                    'subject_definitions:\n<Subject 1> wearing the wrong black photo outfit.\n\n'
                    'integrated_multimodal_description:\n[Shot 1] <Subject 1> follows the seated video action.\n\n'
                    'overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A'
                    for _ in parsed['scene_prompts']]
                result['choices'][0]['message']['content'] = json.dumps(parsed)
            return result
        scope['_external_llm_request'] = reply
        args = self.kwargs()
        args.update(edit_mode='Transfer', driven_video=Frame(), llm=SimpleNamespace(model='test'),
            edit_request='Keep the video clothing and actions, replace the performer identity.')
        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
        for prompt in json.loads(result[0])['scene_prompts']:
            self.assertEqual(prompt.count('subject_definitions:'), 1)
            self.assertNotIn('wrong black photo outfit', prompt)
            self.assertIn('Wardrobe: Team uniform and trainers', prompt)
            self.assertIn('follows the seated video action', prompt)

    def test_transfer_auto_none_controls_are_neutral(self):
        for source in ('source_video_1', 'driven_video'):
            scope, _ = load_director()
            cls = scope['H3CompactMultimodalEditDirector']
            args = self.kwargs()
            args.update(edit_mode='Transfer', llm=SimpleNamespace(model='test'))
            args[source] = Frame()
            cls.execute(**args, direction_context='')
            original = json.loads(json.dumps(scope['payloads']))
            scope['payloads'] = []
            cls.execute(**args, direction_context='Genre: Auto + none\nMotion style: none\nVisual look: Auto\nDialogue: none')
            self.assertEqual(original, scope['payloads'])

    def test_transfer_active_controls_override_only_selected_attributes(self):
        for source in ('source_video_1', 'driven_video'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode='Transfer', llm=SimpleNamespace(model='test'),
                edit_request='Keep her red shirt and identity. Transfer the observed action.',
                direction_context='Genre: Cyberpunk + Auto\nMotion style: Orbit Camera + none\nVisual look: Cinematic\nDialogue: No dialogue')
            args[source] = Frame()
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            self.assertIn('TRANSFER DIRECTION CONTROLS — SCOPED OVERRIDES', system)
            self.assertIn('Genre: Cyberpunk\nMotion style: Orbit Camera', system)
            self.assertNotIn('Cyberpunk + Auto', system)
            self.assertIn(scope['MOTION_STYLES']['Orbit Camera'], system)
            self.assertIn(scope['VISUAL_LOOKS']['Cinematic'], system)
            self.assertIn('Genre `Cyberpunk` governs visible production design', system)
            self.assertIn('not only a neon color grade', system)
            self.assertIn('Priority: written user changes/locks', system)
            self.assertIn('Orbit Camera requires a concrete smooth arc', system)
            self.assertIn('never declare a changed attribute fully preserved', system)
            self.assertNotIn('without inventing events, relocating the scene or changing appearance', system)
            self.assertEqual(calls, ['external'] if source == 'source_video_1' else ['external', 'external'])

    def test_transfer_continuation_uses_first_design_and_current_block_motion(self):
        for mode in ('Transfer', 'Enhance'):
            scope, calls = load_director()
            args = self.kwargs()
            memory = ('Transfer the action.\nSOURCE BLOCK 3/3: 20.000..30.000s.\n'
                'CONTINUITY MEMORY\nFIRST ACCEPTED BLOCK:\nRed-haired woman in a blue cyberpunk suit.\n'
                'PREVIOUS ACCEPTED BLOCK:\nShe reaches the doorway.')
            args.update(edit_mode=mode, source_video_1=Frame(), edit_request=memory,
                direction_context='Genre: Cyberpunk\nMotion style: Orbit Camera',
                llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            if mode == 'Transfer':
                self.assertIn('TRANSFER CONTINUATION:', system)
                self.assertIn('not the action to replay', system)
                self.assertIn('Use PREVIOUS ACCEPTED BLOCK only for the last physical state', system)
                self.assertIn('Current source frames supply this block', system)
                self.assertIn('do not freeze genuine action/location changes', system)
                self.assertIn('The first generated scene establishes the target design', system)
            else:
                self.assertNotIn('TRANSFER CONTINUATION:', system)
            self.assertIn(memory, json.dumps(scope['last_payload']['messages'][1]['content']).replace('\\n', '\n'))
            self.assertEqual(calls, ['external'])

    def test_reference_transfer_renderer_uses_roles_and_six_sections(self):
        scope, _ = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        scene = dict(source_video='<Video 1>', edit_instruction='Change the environment to a dungeon.',
            subject_definitions='<Subject 1> the observed performer whose movement comes from <Video 1>.',
            retention_analysis='<Video 1>: partially_preserved - performance retained; environment changed.',
            detailed_description='[Shot 4] At 00:10.000, The performer reaches toward the dungeon door.',
            overall_soundscape='N/A', non_diegetic_music='N/A')
        rendered = cls._compile_scene_prompt(cls._render_transfer_scene(scene, ['<Video 1>']), sequence=True)
        self.assertIn('summary:\n[video editing] The target video is an edited version of <Video 1>', rendered)
        self.assertIn('[Shot 1] The performer', rendered)
        self.assertNotIn('At 00:10.000', rendered)
        self.assertNotIn('original background remains unchanged', rendered)
        scene['subject_definitions'] = '<Subject 1> identity/appearance from <Picture 1>, performance from <Video 1>.'
        self.assertIn('[video editing + reference generation]', cls._render_transfer_scene(scene, ['<Video 1>']))

    def test_transfer_numbered_video_route_is_distinct_from_driven(self):
        for socket, tag in (('source_video_1', '<Video 1>'), ('source_video_2', '<Video 2>')):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode='Transfer', llm=SimpleNamespace(model='test'),
                edit_request='Replace the protagonist with the woman in the reference image.')
            args[socket] = Frame()
            result = scope['H3CompactMultimodalEditDirector'].execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            self.assertIn('TRANSFER NUMBERED VIDEO ROUTE', system)
            self.assertIn('actual canonical tags: ' + tag, system)
            self.assertIn('despite generic finished-footage-only rules', system)
            self.assertIn('not the static image pose', system)
            self.assertEqual(calls, ['external'])
            self.assertEqual(len(json.loads(result[0])['scene_prompts']), 2)
            for prompt in json.loads(result[0])['scene_prompts']:
                self.assertIn('The target video is an edited version of ' + tag, prompt)
                self.assertIn('replace the performer', prompt)
                self.assertTrue(prompt.startswith('subject_definitions:'))
                headings = ('subject_definitions:', 'summary:', 'retention_analysis:',
                    'detailed_description:', 'overall_soundscape:', 'non_diegetic_music:')
                for heading in headings:
                    self.assertEqual(prompt.count(heading), 1)
                self.assertNotIn('integrated_multimodal_description:', prompt)
                self.assertIn('<Subject 1>', prompt)
                self.assertIn('observed turning and reaching', prompt)
                self.assertNotIn('source_video', prompt)
        for mode, source in (('Transfer', 'driven_video'), ('Enhance', 'source_video_1')):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, llm=SimpleNamespace(model='test'))
            args[source] = Frame()
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            self.assertNotIn('TRANSFER NUMBERED VIDEO ROUTE',
                             scope['last_payload']['messages'][0]['content'])
            self.assertEqual(calls, ['external', 'external'] if source == 'driven_video' else ['external'])

    def test_source_transfer_owns_appearance_independently_of_unused_driven_strength(self):
        payloads = []
        for strength in (1, 2, 3):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode='Transfer', source_video_1=Frame(), driven_strength=strength,
                edit_request='Transfer the video actions to the reference woman; her image predominates.',
                llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            payloads.append(scope['last_payload'])
            system = scope['last_payload']['messages'][0]['content']
            self.assertIn(scope['H3CompactMultimodalEditDirector'].SOURCE_TRANSFER_RULES, system)
            self.assertNotIn(scope['H3CompactMultimodalEditDirector'].TRANSFER_RULES, system)
            self.assertNotIn('existing Driven Strength appearance policy', system)
            self.assertIn('Do not mix the source performer', system)
            self.assertIn('incidental handheld objects', system)
            self.assertIn('Mouth movement alone', system)
            self.assertEqual(calls, ['external'])
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[1], payloads[2])

    def test_source_transfer_supports_scoped_edits_with_or_without_character_image(self):
        for request, has_picture in (
            ('Transfer actions to the reference woman wearing office clothes in a dungeon.', True),
            ('Change only the environment to a dungeon.', False),
            ('Change the outfit to office clothes and add a guard.', False),
            ('Swap only the face; retain the original clothes and body.', True),
        ):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode='Transfer', source_video_1=Frame(), edit_request=request,
                llm=SimpleNamespace(model='test'))
            if not has_picture:
                args['reference_image_1'] = None
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            payload = scope['last_payload']
            system = payload['messages'][0]['content']
            self.assertIn(request, json.dumps(payload['messages'][1]['content']))
            self.assertIn('face-only swap instead preserves the source body/outfit', system)
            self.assertIn('Without an assigned replacement image', system)
            self.assertIn('Never also lock the old background', system)
            self.assertIn('remain additions, not automatic replacements', system)
            self.assertEqual(calls, ['external'])

    def test_source_transfer_excludes_selfie_props_and_requests_moving_footage(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Transfer', source_video_1=Frame(),
            edit_request='Transfer the video actions to the reference woman and add reggaeton music.',
            llm=SimpleNamespace(model='test'))
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        system = scope['last_payload']['messages'][0]['content']
        self.assertIn('worn accessories such as jewelry, not every object', system)
        self.assertIn('Exclude incidental handheld objects from subject_definitions,', system)
        self.assertIn('observed video action uses them', system)
        self.assertIn('do not invent a phone interaction or put-down event', system)
        self.assertIn('track hand occupancy and contact chronologically', system)
        self.assertIn("not 'video stills'", system)
        self.assertIn('unless the user explicitly requests that treatment', system)
        self.assertEqual(calls, ['external'])

    def test_source_transfer_rules_do_not_leak_into_driven_or_enhance(self):
        for mode, source in (('Transfer', 'driven_video'), ('Enhance', 'source_video_1')):
            scope, _ = load_director()
            cls = scope['H3CompactMultimodalEditDirector']
            args = self.kwargs()
            args.update(edit_mode=mode, llm=SimpleNamespace(model='test'))
            args[source] = Frame()
            cls.execute(**args)
            original = json.loads(json.dumps(scope['payloads']))
            cls.SOURCE_TRANSFER_RULES += '\nSOURCE-TRANSFER-ONLY TEST MARKER'
            scope['payloads'] = []
            cls.execute(**args)
            self.assertEqual(original, scope['payloads'])

    def test_transfer_assigns_image_appearance_and_video_performance_at_all_strengths(self):
        for strength in (1, 2, 3):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode='Transfer', driven_video=Frame(), driven_strength=strength,
                edit_request='The reference woman performs the action of the video.',
                llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            self.assertIn('SOURCE RESPONSIBILITIES', system)
            self.assertIn("never the photograph's posture", system)
            self.assertIn('motion, including weight shifts, hand releases and reactions', system)
            self.assertIn('at EVERY strength', system)
            self.assertIn('an active I2V frame', system)
            self.assertIn('Actual numbered source-video', system)
            self.assertEqual(calls, ['external', 'external'])

    def test_transfer_responsibilities_do_not_change_enhance_or_driven_analysis(self):
        scope, _ = load_director()
        args = self.kwargs()
        args.update(edit_mode='Enhance', driven_video=Frame(), llm=SimpleNamespace(model='test'))
        cls = scope['H3CompactMultimodalEditDirector']
        cls.execute(**args)
        original = json.loads(json.dumps(scope['payloads']))
        cls.TRANSFER_RULES += '\nTRANSFER-ONLY TEST MARKER'
        scope['payloads'] = []
        cls.execute(**args)
        self.assertEqual(original, scope['payloads'])
        args['edit_mode'] = 'Transfer'
        scope['payloads'] = []
        cls.execute(**args)
        self.assertEqual(original[0], scope['payloads'][0])
        self.assertIn('TRANSFER-ONLY TEST MARKER', scope['last_payload']['messages'][0]['content'])

    def test_cut_removed_and_continuous_rules_are_the_only_sequence_route(self):
        scope, calls = load_director()
        cls = scope['H3CompactMultimodalEditDirector']
        for mode in ('Enhance', 'Continuous Elaborate'):
            args = self.kwargs()
            args.update(edit_mode=mode, llm=SimpleNamespace(model='test'))
            result = cls.execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            props = scope['last_payload']['response_format']['json_schema']['schema']['properties']
            self.assertNotIn('cut_scene_plan', props)
            self.assertNotIn('CUT SEQUENCE', system)
            self.assertNotIn('Cut on', result[1])
            if mode == 'Continuous Elaborate':
                self.assertIn(cls.CONTINUOUS_ELABORATE_RULES, system)
                self.assertIn(cls.CONTINUOUS_H3_HANDOFF_RULES, system)
            else:
                self.assertIn(cls.CONTINUOUS_RULES, system)
        self.assertEqual(calls, ['external', 'external'])
        self.assertNotIn('cut', inspect.signature(cls.execute).parameters)

    def test_direction_controls_auto_does_not_compete_with_explicit_selection(self):
        scope, _ = load_director()
        tree = ast.parse((ROOT / 'story_director.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'H3CompactDirectionControls')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'controls', 'exec'), scope)
        controls = scope['H3CompactDirectionControls']
        self.assertEqual(controls.execute(genre_1='Auto', motion_style_1='Auto', look_1='Auto')[0], '')
        self.assertEqual(controls.execute(genre_1='Auto', genre_2='Dark Comedy',
            look_1='1950s Technicolor Studio', look_2='Auto', dialogue='No dialogue')[0],
            'Genre: Dark Comedy\nVisual look: 1950s Technicolor Studio\nDialogue: No dialogue')

    def test_explicit_controls_drive_decisions_without_added_calls(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, max_tokens=2400,
                edit_request='Create a comic story with these people.', llm=SimpleNamespace(model='test'),
                direction_context='Genre: Dark Comedy + Auto\nMotion style: Fast Motion + Auto\nVisual look: 1950s Technicolor Studio + Auto')
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            payload = scope['last_payload']
            system = payload['messages'][0]['content']
            self.assertIn('CONTROL DECISIONS — RESOLVE BEFORE DESIGN AND STORY', system)
            self.assertIn('Do not require the user to repeat a selected era in text', system)
            self.assertIn('uncomfortable irony, hypocrisy or perverse cost', system)
            self.assertIn('Fast Motion still allows a listening pause', system)
            self.assertNotIn('Genre `Auto` governs', system)
            self.assertNotIn('do not invent a new plot;', system)
            self.assertEqual(calls, ['external'])
            self.assertEqual(payload['max_tokens'], 2400)

    def test_appearance_is_separate_from_identity_and_result_is_rendered(self):
        scope, calls = load_director()
        args = self.kwargs()
        args.update(edit_mode='Continuous Elaborate', max_tokens=2400,
                    direction_context='Visual look: 1950s Technicolor Studio\nGenre: Dark Comedy',
                    llm=SimpleNamespace(model='test'))
        result = scope['H3CompactMultimodalEditDirector'].execute(**args)
        payload = scope['last_payload']
        props = payload['response_format']['json_schema']['schema']['properties']
        subject_schema = props['target_design']['properties']['subjects']['items']
        self.assertEqual(subject_schema['required'], ['subject_tag', 'identity', 'target_appearance'])
        item = props['scene_prompts']['items']
        self.assertIn('opening_framing', item['required'])
        self.assertIn('visible_result', item['required'])
        for scene in json.loads(result[0])['scene_prompts']:
            self.assertIn('Target appearance: wearing a blue jacket.', scene)
            self.assertIn('[Shot 1] An eye-level wide shot', scene)
            self.assertIn('She reaches the far doorway.', scene)
            self.assertNotIn('[Shot 2]', scene)
        system = payload['messages'][0]['content']
        self.assertIn('Do not put source clothing or hairstyle in identity', system)
        self.assertIn('A pan/reframe is NOT a new [Shot n]', system)
        self.assertIn('DRAMATIZE, DO NOT DEFER', system)
        self.assertIn('actually shown, not promised', system)
        self.assertIn('explicit keep-outfit locks, I2V', system)
        self.assertEqual(calls, ['external'])
        self.assertEqual(payload['max_tokens'], 2400)

    def test_structured_design_renders_four_sections_and_keeps_public_strings(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
            with self.subTest(mode=mode):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, llm=SimpleNamespace(model='test'), max_tokens=2400)
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                payload = scope['last_payload']
                schema = payload['response_format']['json_schema']['schema']
                self.assertIn('target_design', schema['required'])
                scenes = json.loads(result[0])['scene_prompts'] if mode != 'Elaborate' else [result[0]]
                for scene in scenes:
                    self.assertIsInstance(scene, str)
                    for heading in ('subject_definitions:', 'integrated_multimodal_description:',
                                    'overall_soundscape:', 'non_diegetic_music:'):
                        self.assertEqual(scene.count(heading), 1)
                    self.assertIn('wearing a blue jacket', scene)
                    self.assertIn('Soft directional light', scene)
                    self.assertIn('non_diegetic_music:\nN/A', scene)
                    self.assertIn('[Shot 1]', scene)
                    self.assertNotIn('target_design', scene)
                self.assertIn('SHORT DIALOGUE, VISIBLE STORY', payload['messages'][0]['content'])
                self.assertEqual(calls, ['external'])
                self.assertEqual(payload['max_tokens'], 2400)

    def test_structured_renderer_handles_absent_audio_without_inventing_it(self):
        scope, _ = load_director()
        director = scope['H3CompactMultimodalEditDirector']
        design = dict(subject_definitions='<Subject 1> the woman from <Picture 1> in a tailored dress.',
                      visual_style='Rich color and sculpted studio light.')
        scene = dict(integrated_multimodal_description='[Shot 1] A wide view of the doorway as she arrives.')
        prompt = director._render_creative_scene(scene, design, opening_frame=True)
        self.assertIn('overall_soundscape:\nN/A', prompt)
        self.assertIn('non_diegetic_music:\nN/A', prompt)
        self.assertTrue(prompt.startswith('For the target video, at 0.00 seconds'))
        self.assertNotIn('For the target video', director._render_creative_scene(scene, design))
        self.assertEqual(director._render_creative_scene('Legacy source-bound prose.', {}),
                         'Legacy source-bound prose.')

    def test_world_design_does_not_rewrite_transfer_or_faithful_modes(self):
        for mode in ('compact', 'edit', 'Continuous Edit', 'Transfer'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, direction_context='Visual look: 1950s Technicolor Studio',
                        llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            self.assertNotIn('LOOK AS WORLD DESIGN — CREATIVE MODE PRIORITY',
                             scope['last_payload']['messages'][0]['content'])
            self.assertEqual(calls, ['external'])

    def test_requested_formats_share_one_creative_route(self):
        for request in ('Create a drama with this woman.', 'Create a superhero trailer.',
                        'Create a comedy.', 'Create an experimental music video.',
                        'Create a commercial for this product.'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode='Enhance', edit_request=request, llm=SimpleNamespace(model='test'))
            result = scope['H3CompactMultimodalEditDirector'].execute(**args)
            payload = scope['last_payload']
            system = payload['messages'][0]['content']
            self.assertIn('FORMAT FIRST', system)
            self.assertIn('not mandatory recipes or new modes', system)
            self.assertIn('Invent the treatment, not unsupported facts', system)
            self.assertIn('not automatically running/fighting', system)
            self.assertEqual(calls, ['external'])
            self.assertEqual(payload['max_tokens'], 1200)
            self.assertEqual(len(json.loads(result[0])['scene_prompts']), 2)

    def test_writer_director_is_scoped_to_creative_modes_without_new_calls(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate', 'compact', 'edit', 'Continuous Edit', 'Transfer'):
            with self.subTest(mode=mode):
                scope, calls = load_director()
                args = self.kwargs()
                args.update(edit_mode=mode, edit_request='Create a heroic fantasy scene with this woman.',
                            llm=SimpleNamespace(model='test'), max_tokens=1200)
                result = scope['H3CompactMultimodalEditDirector'].execute(**args)
                system = scope['last_payload']['messages'][0]['content']
                creative = mode in ('Enhance', 'Elaborate', 'Continuous Elaborate')
                self.assertEqual('WRITER-DIRECTOR — CREATIVE BRIEF AUTHORITY' in system, creative)
                self.assertEqual(calls, ['external'])
                self.assertEqual(scope['last_payload']['temperature'], 0.2)
                self.assertEqual(scope['last_payload']['seed'], 0)
                self.assertEqual(result[3], 2)
                if creative:
                    self.assertIn('original situation', system)
                    self.assertIn('not automatically running/fighting', system)
                    self.assertIn('No dialogue forbids it', system)
                    self.assertIn('A precise action, faithful edit, source performance', system)
                    self.assertIn('existing word/token budget', system)
                    self.assertLess(system.index('EXECUTABLE FOOTAGE'), system.index('WRITER-DIRECTOR'))

    def test_requested_conversations_have_an_explicit_script_contract_without_extra_calls(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
            for profile in ('Standard', 'Creative Plus'):
                with self.subTest(mode=mode, profile=profile):
                    scope, calls = load_director()
                    args = self.kwargs()
                    args.update(edit_mode=mode, creative_profile=profile,
                                edit_request='Two travelers discuss how to catch the last train.',
                                direction_context='Dialogue: Spanish',
                                llm=SimpleNamespace(model='test'), max_tokens=1200)
                    scope['H3CompactMultimodalEditDirector'].execute(**args)
                    payload = scope['last_payload']
                    system = payload['messages'][0]['content']
                    for rule in ('Requested fictional conversation requires actual words',
                                 'Choose a concrete', 'preceding line/action',
                                 '<Subject N> (SN) says: <d>[Language] exact words</d>',
                                 'cannot replace requested dialogue',
                                 'Keep exact supplied words separate from delivery',
                                 'without mouthing the other line', 'not its opening question',
                                 'Silence, source-audio locks', 'never guess source words from frames'):
                        self.assertIn(rule, system)
                    self.assertNotIn('do not assume S1 equals Subject 1', system)
                    self.assertNotIn('chronological order of first vocal events', system)
                    props = payload['response_format']['json_schema']['schema']['properties']
                    scene = props['edit_prompt'] if mode == 'Elaborate' else props['scene_prompts']['items']
                    action_schema = scene['properties']['integrated_multimodal_description']
                    self.assertIn('actual lines once', action_schema['description'])
                    self.assertIn('No dialogue text', scene['properties']['overall_soundscape']['description'])
                    self.assertEqual(calls, ['external'] * (2 if profile == 'Creative Plus' else 1))
                    self.assertEqual(payload['max_tokens'], 1200)
                    if profile == 'Creative Plus':
                        self.assertIn('connected exchange that advances across beats',
                                      scope['payloads'][0]['messages'][0]['content'])

    def test_speech_permission_stays_distinct_from_language_and_source_preservation(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate', 'Transfer'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, edit_request='A silent traveler waits for the train. No dialogue.',
                        direction_context='Dialogue: Spanish', llm=SimpleNamespace(model='test'))
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            if mode == 'Transfer':
                self.assertNotIn('Requested fictional conversation requires actual words', system)
                self.assertIn('Dialogue language alone does NOT authorize speech', system)
            else:
                self.assertIn('Selected language permits speech, not a quota. No dialogue forbids it', system)
                self.assertIn('Silence, source-audio locks', system)
            self.assertEqual(calls, ['external'])

    def test_writer_director_keeps_driven_and_i2v_contracts(self):
        for mode in ('Enhance', 'Elaborate', 'Continuous Elaborate'):
            scope, calls = load_director()
            args = self.kwargs()
            args.update(edit_mode=mode, edit_request='Create a fantasy scene. No dialogue. Keep her outfit.',
                        llm=SimpleNamespace(model='test'), driven_video=Frame(), i2v_mode=True)
            scope['H3CompactMultimodalEditDirector'].execute(**args)
            system = scope['last_payload']['messages'][0]['content']
            self.assertIn('observed Driven Video chronology', system)
            self.assertIn('I2V frame-zero facts win', system)
            self.assertIn('Keep the referenced principal cast', system)
            self.assertIn('DRIVEN CAST COMPILATION', system)
            self.assertIn('I2V', system)
            self.assertEqual(calls, ['external', 'external'])

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
        self.assertTrue({'driven_image', 'driven_video'} <= names)
        self.assertFalse({'mood_image', 'mood_video', 'storyboard_image', 'shots'} & names)
        modes = next(field.options for field in schema.inputs if field.id == 'edit_mode')
        self.assertNotIn('Storyboard', modes)
        self.assertIn('Transfer', modes)
        self.assertIn('Elaborate Plus', modes)
        self.assertNotIn('Continuous Elaborate', modes)
        strength = schema.inputs[-2]
        self.assertEqual(strength.id, 'driven_strength')
        self.assertEqual((strength.default, strength.min, strength.max), (3, 1, 3))
        profile = schema.inputs[-1]
        self.assertEqual(profile.id, 'creative_profile')
        self.assertEqual(profile.options, ['Standard', 'Creative Plus'])
        self.assertEqual(profile.default, 'Standard')

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
                        args.update(edit_mode='Transfer', driven_strength=strength, continuous_scene_count=count, seconds_per_scene=10,
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
                        llm=SimpleNamespace(model='test'))
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
                    llm=SimpleNamespace(model='test'), direction_context='Motion Style: Fast Motion\nCamera: Locked camera',
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
        args.update(edit_mode='Enhance', seconds_per_scene=10, driven_image=Frame(), llm=SimpleNamespace(model='test'))
        scope['H3CompactMultimodalEditDirector'].execute(**args)
        system = scope['last_payload']['messages'][0]['content']
        self.assertGreater(system.index('EXECUTABLE FOOTAGE'), system.index('DRIVEN SCENARIO —'))
        self.assertIn('EXECUTABLE FOOTAGE', system)
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
        self.assertEqual(prompt_schema['type'], 'object')
        self.assertIn('overall_soundscape', prompt_schema['required'])
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
        self.assertTrue(result[1].startswith("Elaborate Plus "))
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
            self.assertEqual(item['type'], 'object')
            self.assertIn('non_diegetic_music', item['required'])
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
            self.assertIn(scope['MOTION_STYLES']['Orbit Camera'], system)
            self.assertIn(scope['VISUAL_LOOKS']['Cinematic'], system)
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
