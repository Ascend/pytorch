import os
import unittest
import torch.distributed.run as launch
from torch_npu.testing._testcase import run_tests, TestCase
from torch_npu.testing._internal.common_distributed import skipIfUnsupportMultiNPU


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


class ElasticLaunchTest(TestCase):
    @skipIfUnsupportMultiNPU(2)
    def test_communicate_npu_watchdog_timeout(self):
        try:
            error = None
            launch.main(
                [
                    "--run-path",
                    "--nnodes=1",
                    "--nproc-per-node=2",
                    "--monitor-interval=1",
                    path("watchdog/watchdog_base.py"),
                ]
            )
        except Exception as e:
            error = e
        finally:
            if error is None:
                raise RuntimeError("Test case fail")


if __name__ == "__main__":
    run_tests()
