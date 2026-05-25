import importlib
import os
import unittest


RUN_MANUAL_NATIVE_TESTS = (
    os.getenv("TORCH_NPU_RUN_FSDP_NATIVE_COLLECTIVES_TESTS") == "1"
)
MANUAL_NATIVE_SKIP_REASON = (
    "manual-only FSDP native collectives checks; "
    "set TORCH_NPU_RUN_FSDP_NATIVE_COLLECTIVES_TESTS=1 to run"
)


@unittest.skipUnless(RUN_MANUAL_NATIVE_TESTS, MANUAL_NATIVE_SKIP_REASON)
class TestFSDPNativeCollectives(unittest.TestCase):
    def _require_npu(self):
        import torch

        importlib.import_module("torch_npu")

        if not torch.npu.is_available():
            self.skipTest("NPU is required for FSDP native collectives checks")
        torch.npu.set_device(0)
        return torch, torch.device("npu:0")

    def test_foreach_copy_same_device_npu(self):
        torch, device = self._require_npu()

        src_tensors = [
            torch.tensor([1.0, 2.0], device=device),
            torch.tensor([3.0, 4.0, 5.0], device=device),
        ]
        dst_tensors = [torch.empty_like(tensor) for tensor in src_tensors]

        torch._foreach_copy_(dst_tensors, src_tensors)
        torch.npu.synchronize()

        for actual, expected in zip(dst_tensors, src_tensors):
            torch.testing.assert_close(actual.cpu(), expected.cpu())

    def test_native_all_gather_copy_in_op_copies_rank_input(self):
        torch, device = self._require_npu()
        importlib.import_module("torch.distributed.fsdp._fully_shard._fsdp_collectives")

        all_gather_inputs = [
            torch.tensor([1.0, 2.0], device=device),
            torch.tensor([3.0, 4.0, 5.0], device=device),
        ]
        inp_split_sizes = [tensor.numel() for tensor in all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        rank = 1
        world_size = 2
        all_gather_output = torch.full(
            (all_gather_input_numel * world_size,), -1.0, device=device
        )

        rank_input, gathered_output = torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            all_gather_output,
            inp_split_sizes,
            all_gather_input_numel,
            rank,
        )
        torch.npu.synchronize()

        expected_rank_input = torch.cat(all_gather_inputs).cpu()
        expected_output = torch.cat(
            [
                torch.full((all_gather_input_numel,), -1.0, device=device),
                torch.cat(all_gather_inputs),
            ]
        ).cpu()

        torch.testing.assert_close(rank_input.cpu(), expected_rank_input)
        torch.testing.assert_close(gathered_output.cpu(), expected_output)
        self.assertEqual(gathered_output.data_ptr(), all_gather_output.data_ptr())

    def test_native_get_param_all_gather_inputs_uses_foreach_copy_path(self):
        torch, device = self._require_npu()
        fsdp_collectives = importlib.import_module(
            "torch.distributed.fsdp._fully_shard._fsdp_collectives"
        )
        from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState

        class MockFSDPParam:
            def __init__(self, sharded_state, sharded_data, post_forward_data):
                self.param_dtype = sharded_data.dtype
                self.offload_to_cpu = False
                self._sharded_local_tensor = sharded_data
                self.sharded_state = sharded_state
                self._sharded_param_data = sharded_data
                self._sharded_post_forward_param_data = post_forward_data
                self.device = sharded_data.device

        sharded_data = torch.tensor([1.0, 2.0], device=device)
        post_forward_data = torch.tensor([3.0, 4.0, 5.0], device=device)
        fsdp_params = [
            MockFSDPParam(
                ShardedState.SHARDED,
                sharded_data,
                torch.full_like(sharded_data, -1.0),
            ),
            MockFSDPParam(
                ShardedState.SHARDED_POST_FORWARD,
                torch.full_like(post_forward_data, -1.0),
                post_forward_data,
            ),
        ]

        all_gather_inputs = fsdp_collectives._get_param_all_gather_inputs(fsdp_params)
        torch.npu.synchronize()

        self.assertEqual(len(all_gather_inputs), 2)
        self.assertEqual([len(inputs) for inputs in all_gather_inputs], [1, 1])
        torch.testing.assert_close(all_gather_inputs[0][0].cpu(), sharded_data.cpu())
        torch.testing.assert_close(
            all_gather_inputs[1][0].cpu(), post_forward_data.cpu()
        )
        self.assertNotEqual(all_gather_inputs[0][0].data_ptr(), sharded_data.data_ptr())
        self.assertNotEqual(
            all_gather_inputs[1][0].data_ptr(), post_forward_data.data_ptr()
        )


if __name__ == "__main__":
    unittest.main()
