import os
import subprocess
import sys

from torch_npu.testing.testcase import TestCase, run_tests


class TestWorkerStartMethod(TestCase):
    def test_inductor_uses_spawn(self):
        env = os.environ.copy()
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        env["TORCHINDUCTOR_WORKER_START"] = "fork"
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from torch._inductor import config; "
                "assert config.worker_start_method == 'fork'; "
                "import torch_npu._inductor; "
                "import os; "
                "assert os.environ['TORCHINDUCTOR_WORKER_START'] == 'spawn'; "
                "assert config.worker_start_method == 'spawn'",
            ],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    run_tests()
