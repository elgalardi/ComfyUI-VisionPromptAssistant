"""Offline checks for the interchangeable-provider simple assistant."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

class AssistantTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "llm_vision_prompt.py").read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        self.calls = []
        self.reply = {"prompt": "Ready prompt."}
        def request(llm, payload):
            self.calls.append(payload)
            return {"choices": [{"message": {"content": json.dumps(self.reply)}}]}
        scope = {"json": json, "io": SimpleNamespace(ComfyNode=object, NodeOutput=lambda *v: v),
                 "_external_llm_request": request, "_image_data_url": lambda *a: "data:image/png;base64,test",
                 "_completion_content": lambda result, *a: result["choices"][0]["message"]["content"]}
        execute = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute")
        execute.body = [n for n in execute.body if not isinstance(n, ast.ImportFrom)]
        scope["_local_qwen_json"] = lambda raw, schema: raw
        exec(compile(tree, "llm_vision_prompt.py", "exec"), scope)
        self.node = scope["LLMVisionPromptAssistant"]
        self.llm = SimpleNamespace(model="test-model")

    def test_plain_text_and_sampling(self):
        self.assertEqual(self.node.execute(self.llm, "Action", "System", temperature=0.4, seed=12), ("Ready prompt.",))
        self.assertEqual(self.calls[0]["temperature"], 0.4)
        self.assertEqual(self.calls[0]["seed"], 12)
        self.assertEqual(len(self.calls), 1)

    def test_reference_numbering_with_empty_socket(self):
        class Image:
            shape = (1, 8, 8, 3)
            def __getitem__(self, item): return self
        self.node.execute(self.llm, "Action", "System", image_1=Image())
        content = self.calls[0]["messages"][1]["content"]
        self.assertEqual(content[1]["text"], "<Picture 1> reference:")
        self.assertEqual(content[2]["type"], "image_url")

    def test_empty_prompt_rejected(self):
        self.reply = {"prompt": " "}
        with self.assertRaises(ValueError):
            self.node.execute(self.llm, "Action", "System")
