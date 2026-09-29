import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


class OptionalResizeTests(unittest.TestCase):
    def test_missing_and_dimensions(self):
        path = Path(__file__).resolve().parents[1] / "optional_reference_resize.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        calls = []
        class Image:
            shape = (1, 1000, 2000, 3)
            def movedim(self, *args):
                return self
        def upscale(image, width, height, method, crop):
            calls.append((width, height, method, crop))
            return image
        scope = dict(math=math, io=SimpleNamespace(ComfyNode=object, NodeOutput=lambda *v: v),
                     comfy=SimpleNamespace(utils=SimpleNamespace(common_upscale=upscale)))
        exec(compile(tree, str(path), "exec"), scope)
        resize = scope["OptionalReferenceResize"]
        self.assertEqual(resize.execute(), (None, 0, 0))
        self.assertEqual(calls, [])
        output = resize.execute(image=Image())
        self.assertEqual(output[1:], (2016, 992))
        self.assertEqual(calls[-1][-1], "disabled")
        output = resize.execute(image=Image(), width=512, height=512, keep_proportion="crop")
        self.assertEqual(output[1:], (512, 512))
        self.assertEqual(calls[-1][-1], "center")


if __name__ == "__main__":
    unittest.main()
