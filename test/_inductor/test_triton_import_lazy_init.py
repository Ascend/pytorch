import os
import subprocess
import sys
import textwrap
import unittest


class TestTritonImportLazyInit(unittest.TestCase):
    def run_in_fresh_process(self, code):
        env = dict(os.environ)
        env.pop("TORCHINDUCTOR_NPU_BACKEND", None)
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_backend_import_does_not_initialize_device(self):
        self.run_in_fresh_process("""
            import sys
            import torch, torch_npu
            from unittest import mock
            assert not torch.npu.is_initialized()
            with mock.patch.object(torch.npu, '_lazy_init',
                                   side_effect=AssertionError('import initialized NPU')):
                import torch_npu._inductor
            for name in ('mm_common', 'mm', 'bmm'):
                assert 'torch_npu._inductor.kernel.' + name in sys.modules, name
            assert not torch.npu.is_initialized()
            from torch_npu._inductor import config
            from torch_npu._inductor.kernel import mm_common, bmm
            assert config._get_core_nums.cache_info().currsize == 0
            with mock.patch.object(config, '_obtain_and_limit_cube_vector_core_num',
                                   return_value=(None, 7, 14)) as query:
                assert mm_common.get_num_cube_cores() == 7
                assert mm_common.get_num_cube_cores() == 7
                assert bmm._split_groups(14) == (7, 2, False)
                assert bmm._wave_use(14) == 1.0
                query.assert_called_once_with()
        """)

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_fork_can_use_npu_after_default_backend_import(self):
        self.run_in_fresh_process("""
            import multiprocessing as mp
            import sys
            import torch, torch_npu
            import torch_npu._inductor
            assert 'torch_npu._inductor.kernel.mm' in sys.modules
            assert 'torch_npu._inductor.kernel.bmm' in sys.modules
            assert not torch.npu.is_initialized()
            def child():
                from torch_npu._inductor import config
                from torch_npu._inductor.kernel.mm_common import get_num_cube_cores
                assert get_num_cube_cores() == config.num_cube_core > 0
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
