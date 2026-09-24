import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('dispatch', Path(__file__).parents[1] / 'util/decode_dispatch_profile.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DispatchProfile = module.DispatchProfile


class Input:
    def __init__(self, shape):
        self.shape = shape


class Block:
    key = 'block'
    def forward(self, x):
        return x


class Model:
    def __init__(self):
        self.block = Block()
    def __iter__(self):
        yield self.block
    def forward_ls(self, x):
        return self.block.forward(x)
    def forward(self, input_ids):
        return self.forward_ls(input_ids)


class TestDispatch(unittest.TestCase):
    def test_decode_only_and_restoration(self):
        model = Model()
        original = model.block.forward
        with DispatchProfile(model) as profiler:
            model.forward(Input((1, 160)))
            for _ in range(3):
                model.forward(input_ids=Input((1, 1)))
            model.forward(Input((2, 1)))
        report = profiler.report()
        self.assertEqual(report['decode_forward_calls'], 3)
        self.assertEqual(len(report['calls']), 3)
        for row in report['calls']:
            self.assertEqual(row['count'], 3)
            self.assertGreaterEqual(row['inclusive_ms'], row['exclusive_ms'])
            self.assertGreaterEqual(row['exclusive_ms'], 0)
        self.assertEqual(model.block.forward, original)
        self.assertNotIn('forward', model.block.__dict__)

    def test_exception_restores_owned_method(self):
        model = Model()
        def fail(x):
            raise RuntimeError('test')
        model.block.forward = fail
        profiler = DispatchProfile(model)
        with self.assertRaises(RuntimeError), profiler:
            model.forward(Input((1, 1)))
        self.assertIs(model.block.forward, fail)
        self.assertEqual(profiler.stack, [])
        self.assertFalse(profiler.active)


if __name__ == '__main__':
    unittest.main()
