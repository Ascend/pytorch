from unittest import mock

from torch.testing._internal.common_utils import run_tests, TestCase

from torch_npu._inductor.runtime.triton_heuristics import NPUCachingAutotuner


class TestPrecompileWithoutDevice(TestCase):
    def test_parallel_precompile_success(self):
        self._check_parallel_precompile(fails=False)

    def test_parallel_precompile_failure(self):
        self._check_parallel_precompile(fails=True)

    def _check_parallel_precompile(self, fails):
        autotuner = object.__new__(NPUCachingAutotuner)
        autotuner.triton_meta = {"device": 0}

        def compile_config(config):
            if fails and config == 2:
                raise ValueError("invalid candidate")
            return config * 10

        autotuner._precompile_config = compile_config
        with mock.patch.object(
            autotuner, "get_device_interface",
            side_effect=RuntimeError("Cannot re-initialize NPU in forked subprocess"),
        ) as device:
            results, errors, stacks = autotuner._precompile_configs_parallel_once([1, 2])
        device.assert_not_called()
        self.assertEqual(results, [10, None if fails else 20])
        self.assertIsNone(errors[0])
        if fails:
            self.assertIsInstance(errors[1], ValueError)
            self.assertIn("invalid candidate", stacks[1])
        else:
            self.assertEqual(errors, [None, None])
            self.assertEqual(stacks, ["", ""])


if __name__ == "__main__":
    run_tests()
