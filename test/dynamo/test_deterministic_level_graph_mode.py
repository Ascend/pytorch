import unittest

import torch

import torch_npu


class TestDeterministicLevelGraphMode(unittest.TestCase):
    def tearDown(self):
        torch._dynamo.reset()
        torch_npu.npu.set_deterministic_level(0)
        super().tearDown()

    def test_deterministic_level_guard(self):
        from torch_npu.dynamo._deterministic_guard import install_npu_deterministic_level_guard

        compile_count = 0

        def backend(gm, example_inputs):
            nonlocal compile_count
            self.assertTrue(install_npu_deterministic_level_guard())
            compile_count += 1
            return gm.forward

        def fn(x):
            return x + 1

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
        for level in (1, 2, 1, 2):
            torch_npu.npu.set_deterministic_level(level)
            torch.testing.assert_close(compiled_fn(torch.ones(2)), torch.full((2,), 2.0))

        self.assertEqual(compile_count, 2)


if __name__ == "__main__":
    unittest.main()
