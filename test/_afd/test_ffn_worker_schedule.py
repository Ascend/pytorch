import unittest
import torch
import torch.nn as nn
import torchair
from torchair.configs.compiler_config import CompilerConfig

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

window_size = 209715200
ffn_window_tensor = torch.zeros([window_size], dtype=torch.int8).npu()

attn_workers = 2
micro_batch_number = 3
batch_size = 6
top_k = 8
hidden_size = 7168
expert_num = 288
attn_to_ffn_token_size = (7168 + 4 + 511) // 512 * 512
ffn_to_attn_token_size = 7168 * 2
ffn_window = ffn_window_tensor.data_ptr()


def _set_all_flags():
    num_int8 = attn_workers * micro_batch_number * (8 + batch_size * top_k * 4)
    int32_view = ffn_window_tensor[:num_int8].view(torch.int32)
    int32_view[:] = 1


class TestModelInplace(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, schedule_context):
        torch_npu._afd.ffn_worker_scheduler_(schedule_context, sync_group_size=1, execute_mode=0)


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, schedule_context):
        return torch_npu._afd.ffn_worker_scheduler(schedule_context, sync_group_size=1)


class TestFfnWorkerScheduler(TestCase):
    def setUp(self):
        self.context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode=0, session_num=attn_workers,
                                                                            micro_batch_num=micro_batch_number,
                                                                            micro_batch_size=batch_size,
                                                                            selected_expert_num=top_k + 1,
                                                                            expert_num=expert_num,
                                                                            attn_to_ffn_token_size=attn_to_ffn_token_size,
                                                                            ffn_to_attn_token_size=ffn_to_attn_token_size,
                                                                            ffn_window=ffn_window,
                                                                            ffn_window_size=window_size)

        self.schedule_context = self.context_holder.get_schedule_context_tensor()
        _set_all_flags()

    @unittest.skip("skip case until cann supported")
    @SupportedDevices(['Ascend910B'])
    def test_ffn_worker_scheduler_(self):
        _set_all_flags()
        schedule_context1 = self.schedule_context.clone()
        torch_npu._afd.ffn_worker_scheduler_(self.schedule_context, sync_group_size=2)
        self.assertNotEqual(schedule_context1, self.schedule_context)

    @unittest.skip("skip case until cann supported")
    @SupportedDevices(['Ascend910B'])
    def test_ffn_worker_scheduler(self):
        _set_all_flags()
        schedule_context1 = self.schedule_context.clone()
        schedule_context2 = torch_npu._afd.ffn_worker_scheduler(self.schedule_context, sync_group_size=2)
        self.assertEqual(schedule_context1, self.schedule_context)
        self.assertNotEqual(schedule_context2, self.schedule_context)

    @unittest.skip("skip case until cann supported")
    @SupportedDevices(['Ascend910B'])
    def test_ffn_worker_scheduler__graph(self):
        _set_all_flags()
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = TestModelInplace().npu()
        model = torch.compile(model, backend=npu_backend)
        schedule_context1 = self.schedule_context.clone()
        model(self.schedule_context)
        self.assertNotEqual(schedule_context1, self.schedule_context)
        torch._dynamo.reset()

    @unittest.skip("skip case until cann supported")
    @SupportedDevices(['Ascend910B'])
    def test_ffn_worker_scheduler_graph(self):
        _set_all_flags()
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = TestModel().npu()
        model = torch.compile(model, backend=npu_backend)
        schedule_context1 = self.schedule_context.clone()
        schedule_context2 = model(self.schedule_context)
        self.assertEqual(schedule_context1, self.schedule_context)
        self.assertNotEqual(schedule_context2, self.schedule_context)
        torch._dynamo.reset()


if __name__ == '__main__':
    run_tests()
