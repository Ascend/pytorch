import os
import unittest
import time
import torch.distributed.run as launch
from torch_npu.testing.testcase import run_tests, TestCase
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


class ElasticLaunchTest(TestCase):
    @skipIfUnsupportMultiNPU(2)
    def test_npu_watchdog_timeout(self):
        try:
            error = None
            launch.main(
                [
                    "--nproc-per-node=2",
                    path("watchdog/watchdog_base.py"),
                ]
            )
        except Exception as e:
            error = e
        finally:
            if error is None:
                raise RuntimeError("Test case fail")


    @skipIfUnsupportMultiNPU(2)
    def test_npu_watchdog_quick_exit(self):
        start_time = time.time()
        try:
            launch.main(
                [
                    "--nproc-per-node=2",
                    path("watchdog/watchdog_quick_exit.py"),
                ]
            )
        except Exception:
            print("Program fail and exit")
            
        end_time = time.time()
        excution_time = end_time - start_time
        if excution_time > 120:
            print(f"Excution time using time.time(): {excution_time} seconds")
            raise RuntimeError("Test case fail")
            

if __name__ == "__main__":
    run_tests()
