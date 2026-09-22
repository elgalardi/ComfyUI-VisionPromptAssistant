"""CPU-only compatibility checks; no weights or generation requests."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.argv = [sys.argv[0], '--cpu']
spec = importlib.util.spec_from_file_location('local_vision_test', ROOT / 'local_vision_prompt.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LocalVisionTests(unittest.TestCase):
    def test_original_schema(self):
        schema = module.LocalVisionPromptGenerator.define_schema()
        self.assertEqual(schema.node_id, 'LocalVisionPromptGenerator')
        self.assertEqual(schema.display_name, 'Vision Prompt Assistant')
        self.assertTrue({'image_0', 'image_1', 'image_2', 'clip_name', 'seed',
                         'system_prompt', 'user_prompt', 'max_length'}.issubset(
                             {entry.id for entry in schema.inputs}))

    def test_sampling_and_sparse_references(self):
        clip = Mock()
        clip.decode.return_value = 'A cinematic scene.'
        image = module.torch.zeros(1, 8, 8, 3)
        with patch.object(module.LocalVisionPromptGenerator, '_load_clip', return_value=clip):
            module.LocalVisionPromptGenerator.execute(
                'test.safetensors', 'ltxv', 'cpu', 'Describe the scene', 'Animate it',
                256, True, 0.7, 40, 0.9, 0.05, 1.05, 123, image_2=image)
        self.assertIn('image_2 maps to <Picture 1>', clip.tokenize.call_args.args[0])
        self.assertEqual(len(clip.tokenize.call_args.kwargs['images']), 1)
        self.assertEqual(clip.generate.call_args.kwargs['seed'], 123)
        self.assertEqual(clip.generate.call_args.kwargs['max_length'], 256)
        clip.decode.assert_called_once()


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
