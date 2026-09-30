import os
import subprocess
import sys

from torch_npu.testing.testcase import TestCase, run_tests


class TestWorkerStartMethod(TestCase):
    def test_inductor_preserves_worker_start_method(self):
        for start_method in (None, "fork", "spawn"):
            with self.subTest(start_method=start_method):
                env = os.environ.copy()
                env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
                if start_method is None:
                    env.pop("TORCHINDUCTOR_WORKER_START", None)
                else:
                    env["TORCHINDUCTOR_WORKER_START"] = start_method
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from torch._inductor import config; "
                        "import os; "
                        "before = config.worker_start_method; "
                        "before_env = os.environ.get('TORCHINDUCTOR_WORKER_START'); "
                        "import torch_npu._inductor; "
                        "assert config.worker_start_method == before; "
                        "assert os.environ.get('TORCHINDUCTOR_WORKER_START') == before_env",
                    ],
                    check=True,
                    env=env,
                )


if __name__ == "__main__":
    run_tests()
