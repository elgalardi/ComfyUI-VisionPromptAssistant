"""Offline regressions for local model JSON wrappers and extra output."""
import ast
import json
from pathlib import Path
import re
import unittest


tree = ast.parse((Path(__file__).resolve().parents[1] / "compact_llm_providers.py").read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_local_qwen_json")
scope = {"json": json, "re": re}
exec(compile(ast.Module(body=[function], type_ignores=[]), "local_qwen_json", "exec"), scope)
parse = scope["_local_qwen_json"]


class LocalQwenJSONTests(unittest.TestCase):
    schema = {"type": "object", "required": ["prompt"]}

    def test_complete_json_and_extra_text(self):
        for text in ('{"prompt":"ready"}', '{"prompt":"ready"}<|im_end|>',
                     '```json\n{"prompt":"ready"}\n```\nDone.',
                     'Result follows\n{"prompt":"ready"}'):
            self.assertEqual(json.loads(parse(text, self.schema)), {"prompt": "ready"})

    def test_repeated_identical_answer(self):
        self.assertEqual(json.loads(parse('{"prompt":"ok"}\n{"prompt":"ok"}', self.schema)), {"prompt": "ok"})

    def test_conflicting_answers_rejected(self):
        with self.assertRaisesRegex(ValueError, "Multiple"):
            parse('{"prompt":"first"}\n{"prompt":"second"}', self.schema)

    def test_truncated_answer_rejected(self):
        with self.assertRaisesRegex(ValueError, "No complete"):
            parse('{"prompt":"unfinished', self.schema)

    def test_missing_fields_rejected(self):
        with self.assertRaises(ValueError):
            parse('{"other":1}', self.schema)

    def test_thinking_discarded(self):
        text = '<think>{"prompt":"draft"}</think>{"prompt":"final"}'
        self.assertEqual(json.loads(parse(text, self.schema)), {"prompt": "final"})
        with self.assertRaises(ValueError):
            parse('<think>{"prompt":"draft"}', self.schema)

    def test_nested_data_and_escaped_strings(self):
        value = {"prompt": 'A brace } and a quote "', "nested": [{"value": 2}]}
        self.assertEqual(json.loads(parse(json.dumps(value) + "\nend", self.schema)), value)


if __name__ == "__main__":
    unittest.main()
