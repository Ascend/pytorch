import os
import subprocess
import sys
import textwrap
import unittest


class TestImportLazyInit(unittest.TestCase):
    def run_in_fresh_process(self, code):
        env = dict(os.environ, TORCHINDUCTOR_NPU_BACKEND="ascendc")
        # Test PTA imports independently of the separately packaged TorchAir backend.
        setup = """
            import sys
            import types
            sys.modules['torch_npu._inductor.ascendc'] = types.ModuleType(
                'torch_npu._inductor.ascendc'
            )
        """
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(setup) + textwrap.dedent(code)],
            env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pta_import_does_not_initialize_device(self):
        self.run_in_fresh_process("""
            import torch, torch_npu
            from unittest import mock
            assert not torch.npu.is_initialized()
            with mock.patch.object(torch.npu, '_lazy_init',
                                   side_effect=AssertionError('import initialized NPU')):
                import torch_npu._inductor
                from torch_npu._inductor import config
                from torch_npu._inductor.codegen.split_tiling import SplitTiling
            assert not torch.npu.is_initialized()
            assert config._get_core_nums.cache_info().currsize == 0
        """)

    def test_core_counts_are_lazy_and_cached(self):
        self.run_in_fresh_process("""
            import torch, torch_npu
            import torch_npu._inductor
            from torch_npu._inductor import config
            from torch_npu._inductor.codegen.split_tiling import SplitTiling
            from unittest import mock
            assert not torch.npu.is_initialized()
            assert config._get_core_nums.cache_info().currsize == 0
            with mock.patch.object(config, '_obtain_and_limit_cube_vector_core_num',
                                   return_value=(None, 7, 14)) as query:
                assert config.num_cube_core == 7
                assert config.num_vector_core == 14
                assert config.prop is None
                assert SplitTiling._outer_buckets_with_static_axes() == (14, 28, 42, 56, 70)
                query.assert_called_once_with()
            try:
                config.unknown_device_property
            except AttributeError:
                pass
            else:
                raise AssertionError('unknown attribute did not raise')
        """)

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_fork_can_use_npu_after_parent_import(self):
        self.run_in_fresh_process("""
            import multiprocessing as mp
            import torch, torch_npu
            import torch_npu._inductor
            from torch_npu._inductor import config
            from torch_npu._inductor.codegen.split_tiling import SplitTiling
            assert not torch.npu.is_initialized()
            def child():
                x = torch.ones(8, device='npu')
                torch.testing.assert_close((x + 1).cpu(), torch.full((8,), 2.0))
            p = mp.get_context('fork').Process(target=child)
            p.start()
            p.join(60)
            if p.is_alive():
                p.terminate()
                p.join()
                raise AssertionError('fork child timed out')
            assert p.exitcode == 0, p.exitcode
        """)


if __name__ == "__main__":
    unittest.main()
