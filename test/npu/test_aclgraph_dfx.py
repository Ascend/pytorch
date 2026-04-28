import os
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch_npu
from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion, SupportedDevices
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
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
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
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
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
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
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

    @unittest.skip("Skip until OpApi impl is removed")
    @SkipIfNotGteCANNVersion("8.5.0")
    def test_save_npugraph_tensor(self):
        torch.npu.set_device(0)
        first_graph = torch.npu.NPUGraph()
        second_graph = torch.npu.NPUGraph()
        x = torch.arange(6, dtype=torch.float32, device='npu').reshape(2, 3)
        device_index = _resolved_npu_device_index(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "tensor.pt")
            expected_counter_path = os.path.join(tmpdir, f"tensor_device_{device_index}_0.pt")
            expected_second_counter_path = os.path.join(tmpdir, f"tensor_device_{device_index}_1.pt")

            with torch.npu.graph(first_graph):
                torch.ops.npu.save_npugraph_tensor(x, save_path=save_path)
            first_graph.replay()
            torch.npu.synchronize()
            self.assertTrue(wait_until(lambda: os.path.exists(expected_counter_path)))
            self.assertEqual(torch.load(expected_counter_path), x.cpu())

            with torch.npu.graph(second_graph):
                torch.ops.npu.save_npugraph_tensor(x, save_path=save_path)
            second_graph.replay()
            torch.npu.synchronize()
            self.assertTrue(wait_until(lambda: os.path.exists(expected_second_counter_path)))
            self.assertEqual(torch.load(expected_second_counter_path), x.cpu())

    @unittest.skip("Skip until OpApi impl is removed")
    @SkipIfNotGteCANNVersion("8.5.0")
    def test_save_npugraph_tensor_overwrite(self):
        torch.npu.set_device(0)
        first_graph = torch.npu.NPUGraph()
        second_graph = torch.npu.NPUGraph()
        x = torch.arange(6, dtype=torch.float32, device='npu').reshape(2, 3)
        y = torch.arange(6, dtype=torch.float32, device='npu').reshape(2, 3) + 1
        device_index = _resolved_npu_device_index(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "tensor.pt")
            expected_overwrite_path = os.path.join(tmpdir, f"tensor_device_{device_index}.pt")
            unexpected_counter_path = os.path.join(tmpdir, f"tensor_device_{device_index}_0.pt")

            with torch.npu.graph(first_graph):
                torch.ops.npu.save_npugraph_tensor(x, save_path=save_path, overwrite=True)
            first_graph.replay()
            torch.npu.synchronize()
            self.assertTrue(wait_until(lambda: os.path.exists(expected_overwrite_path)))
            self.assertFalse(os.path.exists(unexpected_counter_path))
            self.assertEqual(torch.load(expected_overwrite_path), x.cpu())

            with torch.npu.graph(second_graph):
                torch.ops.npu.save_npugraph_tensor(y, save_path=save_path, overwrite=True)
            second_graph.replay()
            torch.npu.synchronize()
            self.assertEqual(torch.load(expected_overwrite_path), y.cpu())

    @unittest.skip("Skip until OpApi impl is removed")
    @SkipIfNotGteCANNVersion("8.5.0")
    def test_save_npugraph_tensor_with_default_save_path(self):
        torch.npu.set_device(0)
        g = torch.npu.NPUGraph()
        x = torch.arange(6, dtype=torch.float32, device='npu').reshape(2, 3)
        device_index = _resolved_npu_device_index(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with torch.npu.graph(g):
                    torch.ops.npu.save_npugraph_tensor(x)
                g.replay()
                torch.npu.synchronize()

                def default_saved_files():
                    return [
                        file_name for file_name in os.listdir(tmpdir)
                        if file_name.startswith("tensor_") and file_name.endswith(".pt")
                    ]

                self.assertTrue(wait_until(lambda: len(default_saved_files()) == 1))
                [saved_file] = default_saved_files()
                self.assertIn(f"_device_{device_index}_0.pt", saved_file)
                self.assertEqual(torch.load(os.path.join(tmpdir, saved_file)), x.cpu())
            finally:
                os.chdir(original_cwd)

    @unittest.skip("Skip until OpApi impl is removed")
    @SkipIfNotGteCANNVersion("8.5.0")
    def test_save_npugraph_tensor_tensor_list(self):
        torch.npu.set_device(0)
        g = torch.npu.NPUGraph()
        x = torch.arange(6, dtype=torch.float32, device='npu').reshape(2, 3)
        y = torch.arange(4, dtype=torch.float16, device='npu').reshape(2, 2)
        device_index = _resolved_npu_device_index(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "tensor_list.pt")
            expected_path = os.path.join(tmpdir, f"tensor_list_device_{device_index}_0.pt")

            with torch.npu.graph(g):
                torch.ops.npu.save_npugraph_tensor.tensor_list([x, y], save_path=save_path)
            g.replay()
            torch.npu.synchronize()
            self.assertTrue(wait_until(lambda: os.path.exists(expected_path)))

            saved = torch.load(expected_path)
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0], x.cpu())
            self.assertEqual(saved[1], y.cpu())


if __name__ == '__main__':
    run_tests()