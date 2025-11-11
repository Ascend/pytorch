from unittest.mock import patch

from torch.distributed import run as torch_run
from torch.distributed.argparse_util import check_env, env
from torch.distributed.run import get_args_parser
from torch.distributed.elastic.multiprocessing.errors import record
import torch_npu
from torch_npu.distributed.run import parse_args
from torch_npu.distributed.run import _main
from torch_npu.testing.testcase import TestCase, run_tests


class TestRun(TestCase):
    def test_main_function_default_args(self):
        with patch('torch_npu.distributed.run.torch_run.run') as mock_run:
            _main(["dummy_script.py"])
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            self.assertEqual(call_args.rdzv_backend, "parallel")
            self.assertIsNotNone(call_args.rdzv_endpoint)

    def test_parse_args_default_tiered_parallel_tcpstore(self):
        args = ["--nproc_per_node", "1", "dummy_script.py"]
        parsed_args = parse_args(args)
        self.assertEqual(parsed_args.enable_tiered_parallel_tcpstore, "false")

    def test_main_function_with_existing_rdzv_endpoint(self):
        with patch('torch_npu.distributed.run.torch_run.run') as mock_run:
            import argparse
            args = argparse.Namespace(nproc_per_node=1,
                                      master_addr='localhost',
                                      master_port=12345,
                                      rdzv_backend=None,
                                      rdzv_endpoint="existing_endpoint:54321")

            with patch('torch_npu.distributed.run.parse_args', return_value=args):
                _main(None)

                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                self.assertEqual(call_args.rdzv_backend, 'parallel')
                self.assertEqual(call_args.rdzv_endpoint, "existing_endpoint:54321")


if __name__ == "__main__":
    run_tests()