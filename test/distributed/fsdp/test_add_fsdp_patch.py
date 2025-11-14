import logging
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled, TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup, AllGatherState
from torch.distributed.utils import _to_kwargs
from torch_npu.distributed.fsdp._add_fsdp_patch import _patched_finalize_backward, _get_param_all_gather_inputs, \
    _apply_fsdp_patch
import torch_npu.distributed.fsdp._add_fsdp_patch as add_fsdp_patch
from torch_npu.testing.testcase import TestCase, run_tests


class TestAddFsdpPatch(TestCase):
    def test_get_param_all_gather_inputs_compiled_autograd(self):

        with patch('torch.distributed.fsdp._fully_shard._fsdp_common.compiled_autograd_enabled', return_value=True):
            class MockFSDPParam:
                def __init__(self):
                    self.all_gather_inputs = [torch.tensor([3.0, 4.0])]
                    self.param_dtype = torch.float32
                    self.offload_to_cpu = False
                    self._sharded_local_tensor = torch.tensor([1.0, 2.0])
                    self.sharded_state = ShardedState.SHARDED
                    self._sharded_param_data = torch.tensor([3.0, 4.0])
                    self._sharded_post_forward_param_data = torch.tensor([5.0, 6.0])
                    self.device = torch.device("cpu")

            fsdp_param = MockFSDPParam()
            fsdp_params = [fsdp_param]

            result = _get_param_all_gather_inputs(fsdp_params)
            self.assertEqual(result, [fsdp_param.all_gather_inputs])

    def test_patched_finalize_backward_with_events(self):
        class MockFSDPParamGroup:
            def __init__(self):
                self.fsdp_params = []
                self._all_gather_result = MockAllGatherResult()
                self._post_forward_indices = [1, 2, 3]

            def _wait_for_post_backward(self):
                pass

        class MockAllGatherResult:
            def __init__(self):
                self.all_gather_event = MockEvent()
                self.all_gather_work = MockWork()

        class MockEvent:
            def synchronize(self):
                pass

            def wait(self, *args):
                pass

        class MockWork:
            def wait(self):
                pass

        class MockFSDPParam:
            def __init__(self):
                self.grad_offload_event = MockEvent()

        mock_group = MockFSDPParamGroup()
        mock_group.fsdp_params = [MockFSDPParam()]

        _patched_finalize_backward(mock_group)

        self.assertIsNone(mock_group._all_gather_result)
        self.assertEqual(len(mock_group._post_forward_indices), 0)

    def test_get_param_all_gather_inputs_no_foreach_copy(self):
        with patch('torch.distributed.fsdp._fully_shard._fsdp_common.compiled_autograd_enabled', return_value=False):
            class MockFSDPParam:
                def __init__(self):
                    self.param_dtype = torch.float32
                    self.offload_to_cpu = True
                    self._sharded_local_tensor = torch.tensor([1.0, 2.0])
                    self.sharded_state = ShardedState.SHARDED
                    self._sharded_param_data = torch.tensor([3.0, 4.0])
                    self._sharded_post_forward_param_data = torch.tensor([5.0, 6.0])
                    self.device = torch.device("cpu")
                    self.all_gather_inputs = [torch.tensor([7.0, 8.0])]

            fsdp_param = MockFSDPParam()
            fsdp_params = [fsdp_param]

            result = _get_param_all_gather_inputs(fsdp_params)
            self.assertEqual(result, [fsdp_param.all_gather_inputs])


if __name__ == "__main__":
    run_tests()