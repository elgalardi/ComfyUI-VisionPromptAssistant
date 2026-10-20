"""Offline diagnostics tests; no provider calls or model loads."""
import ast
import io
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch
import urllib.error
import urllib.parse
import urllib.request


source = Path(__file__).resolve().parents[1] / "story_director.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_openrouter_error_details")
scope = {"json": json, "re": re}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), scope)
format_error = scope["_openrouter_error_details"]


class APIReasoningTests(unittest.TestCase):
    def test_venice_preserves_schema_envelope_and_adds_explicit_contract_without_mutating_payload(self):
        connection_node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                               and n.name == "H3LLMModelAPIConnection")
        connection_scope = dict(json=json, urllib=urllib,
            _chat_completions_url=lambda base: base.rstrip('/') + '/chat/completions')
        exec(compile(ast.Module(body=[connection_node], type_ignores=[]), str(source), "exec"), connection_scope)
        schema = dict(type='object', properties={'scene_prompts': {
            'type': 'array', 'items': {'type': 'string'}, 'minItems': 3, 'maxItems': 3}},
            required=['scene_prompts'], additionalProperties=False)
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
        for base in ('https://api.venice.ai/api/v1', 'https://example.com/v1',
                     'https://api.venice.ai.example.com/v1'):
            for wrapped in (True, False):
                for embedded in (True, False):
                    with self.subTest(base=base, wrapped=wrapped, embedded=embedded):
                        payload = dict(messages=[{'role': 'system', 'content': 'Director rules.'
                            + ('\nJSON Schema:\n' + schema_text if embedded else '')},
                            {'role': 'user', 'content': [{'type': 'text', 'text': 'Create 3 scenes.'},
                             {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,TEST'}}]}],
                            max_tokens=2400, seed=42, reasoning={'enabled': False},
                            response_format={'type': 'json_schema', 'json_schema':
                                {'name': 'scenes', 'strict': True, 'schema': schema} if wrapped else schema})
                        original = json.dumps(payload, sort_keys=True)
                        connection = connection_scope['H3LLMModelAPIConnection'](base, 'test', '', 30)
                        with patch.object(urllib.request, 'urlopen', return_value=io.BytesIO(b'{}')) as send:
                            connection.h3_chat_completion(payload)
                        send.assert_called_once()
                        sent = json.loads(send.call_args.args[0].data)
                        if base == 'https://api.venice.ai/api/v1':
                            self.assertFalse(sent['venice_parameters']['include_venice_system_prompt'])
                            self.assertEqual(sent['response_format'], payload['response_format'])
                            if wrapped:
                                self.assertEqual(sent['response_format']['json_schema']['schema'], schema)
                            self.assertEqual(sent['messages'][0]['content'].count(schema_text), 1)
                            if not embedded:
                                self.assertIn('do not return a bare array', sent['messages'][0]['content'])
                        else:
                            self.assertNotIn('venice_parameters', sent)
                            self.assertEqual(sent['response_format'], payload['response_format'])
                            self.assertEqual(sent['messages'], payload['messages'])
                        self.assertEqual(sent['messages'][1], payload['messages'][1])
                        self.assertEqual(sent['max_tokens'], 2400)
                        self.assertEqual(sent['seed'], 42)
                        self.assertEqual(json.dumps(payload, sort_keys=True), original)

    def test_venice_translates_reasoning_without_changing_other_providers(self):
        connection_node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                               and n.name == "H3LLMModelAPIConnection")
        connection_scope = dict(json=json, urllib=urllib,
            _chat_completions_url=lambda base: base.rstrip('/') + '/chat/completions')
        exec(compile(ast.Module(body=[connection_node], type_ignores=[]), str(source), "exec"), connection_scope)
        for base, enabled, expected in (
                ('https://api.venice.ai/api/v1', False, True),
                ('https://api.venice.ai/api/v1/', True, False),
                ('https://example.com/v1', False, None),
                ('https://api.venice.ai.example.com/v1', False, None),
                ('https://api.venice.ai/api/v1', None, None)):
            with self.subTest(base=base, enabled=enabled):
                payload = dict(messages=[], provider={'require_parameters': True},
                               venice_parameters={'include_venice_system_prompt': True,
                                                  'enable_web_search': 'off'})
                if enabled is not None:
                    payload['reasoning'] = {'enabled': enabled}
                original = json.dumps(payload, sort_keys=True)
                connection = connection_scope['H3LLMModelAPIConnection'](base, 'test', '', 30)
                with patch.object(urllib.request, 'urlopen', return_value=io.BytesIO(b'{}')) as send:
                    connection.h3_chat_completion(payload)
                sent = json.loads(send.call_args.args[0].data)
                self.assertEqual(sent['venice_parameters'].get('disable_thinking'), expected)
                self.assertEqual(sent['venice_parameters']['include_venice_system_prompt'],
                                 urllib.parse.urlsplit(base).hostname != 'api.venice.ai')
                self.assertEqual(sent['venice_parameters']['enable_web_search'], 'off')
                self.assertEqual(sent['messages'], payload['messages'])
                self.assertNotIn('reasoning', sent)
                self.assertNotIn('provider', sent)
                self.assertEqual(json.dumps(payload, sort_keys=True), original)


class ErrorDetailsTests(unittest.TestCase):
    def test_nested_provider_error_and_redaction(self):
        body = json.dumps({"error": {"message": "Provider returned error", "metadata": {
            "provider_name": "Test Provider", "raw": json.dumps({
                "error": {"code": "permission_denied", "message": "Access denied"},
                "api_key": "private-key", "prompt": "private prompt text"}),
            "request": {"authorization": "private-key"}}}})
        result = format_error(body, "private-key", {})
        self.assertIn("permission_denied", result)
        self.assertIn("Test Provider", result)
        self.assertNotIn("private-key", result)
        self.assertNotIn("private prompt", result)

    def test_plain_error_redacts_credentials_and_prompt(self):
        result = format_error(
            "Denied Bearer abc123 api_key=other-secret sk-or-test private prompt text",
            "abc123", {"messages": [{"content": "private prompt text"}]})
        for secret in ("abc123", "other-secret", "sk-or-test", "private prompt text"):
            self.assertNotIn(secret, result)
        self.assertIn("Denied", result)

    def test_bounded_non_json_response(self):
        self.assertLessEqual(len(format_error("x" * 20000, "", {})), 6000)


if __name__ == "__main__":
    unittest.main()
