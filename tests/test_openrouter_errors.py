"""Offline diagnostics tests; no provider calls or model loads."""
import ast
import json
from pathlib import Path
import re
import unittest


source = Path(__file__).resolve().parents[1] / "story_director.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_openrouter_error_details")
scope = {"json": json, "re": re}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), scope)
format_error = scope["_openrouter_error_details"]


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
