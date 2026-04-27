import os
import tempfile
import time
from unittest.mock import patch

import torch
import torch_npu
from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion
from torch_npu.testing.testcase import TestCase, run_tests


def wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _resolved_npu_device_index(tensor):
    idx = tensor.device.index
    if idx is None:
        idx = torch.npu.current_device()
    return idx


class TestAclgraphDfx(TestCase):

    @SkipIfNotGteCANNVersion("8.5.0")
    def test_print_npugraph_tensor(self):
        torch.npu.set_device(0)
        g = torch.npu.NPUGraph()
        x = torch.arange(6, dtype=torch.float32, device='npu').reshape(2, 3)

        with patch("builtins.print") as mock_print:
            with torch.npu.graph(g):
                torch.ops.npu.print_npugraph_tensor(x, tensor_name="tensor")
            g.replay()
            torch.npu.synchronize()
            self.assertTrue(wait_until(lambda: mock_print.call_count > 0))

        printed_messages = [call.args[0] for call in mock_print.call_args_list if call.args]
        self.assertTrue(any("tensor=tensor(" in msg for msg in printed_messages))
        self.assertTrue(any("shape=(2, 3)" in msg for msg in printed_messages))
        self.assertTrue(any("dtype=torch.float32" in msg for msg in printed_messages))

    @SkipIfNotGteCANNVersion("8.5.0")
    def test_print_npugraph_tensor_with_default_message(self):
        torch.npu.set_device(0)
        g = torch.npu.NPUGraph()
        x = torch.arange(6, dtype=torch.float32, device='npu').reshape(2, 3)

        with patch("builtins.print") as mock_print:
            with torch.npu.graph(g):
                torch.ops.npu.print_npugraph_tensor(x)
            g.replay()
            torch.npu.synchronize()
            self.assertTrue(wait_until(lambda: mock_print.call_count > 0))

        printed_messages = [call.args[0] for call in mock_print.call_args_list if call.args]
        self.assertTrue(any(msg.startswith("tensor(") for msg in printed_messages))

    @SkipIfNotGteCANNVersion("8.5.0")
    def test_print_npugraph_tensor_with_args(self):
        torch.npu.set_device(0)
        g = torch.npu.NPUGraph()
        x = torch.arange(6, dtype=torch.float32, device='npu').reshape(2, 3)

        with patch("builtins.print") as mock_print:
            with torch.npu.graph(g):
                torch_npu.print_npugraph_tensor(x, tensor_name="x")
            g.replay()
            torch.npu.synchronize()
            self.assertTrue(wait_until(lambda: mock_print.call_count > 0))

        printed_messages = [call.args[0] for call in mock_print.call_args_list if call.args]
        self.assertTrue(any("x=tensor(" in msg for msg in printed_messages))
        self.assertTrue(any("shape=(2, 3)" in msg for msg in printed_messages))
        self.assertTrue(any("dtype=torch.float32" in msg for msg in printed_messages))


if __name__ == '__main__':
    run_tests()