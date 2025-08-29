import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestScheduleContext(TestCase):
    def setUp(self):
        npu_device = torch._C._get_privateuse1_backend_name()
        self.window_tensor = torch.ones([1 * 1024 * 1024 * 1024], dtype=torch.int8).to(
            npu_device
        )
        self.default_params = {
            "schedule_mode": 0,
            "session_num": 288,
            "micro_batch_num": 3,
            "micro_batch_size": 30,
            "selected_expert_num": 9,
            "expert_num": 288,
            "attn_to_ffn_token_size": 7168 + 512,
            "ffn_to_attn_token_size": 7168 * 2,
            "attention_window": self.window_tensor.data_ptr(),
            "attention_window_size": 1 * 1024 * 1024 * 1024,
            "ffn_window": self.window_tensor.data_ptr(),
            "ffn_window_size": 1 * 1024 * 1024 * 1024,
        }

    def tearDown(self):
        return super().tearDown()

    def test_init_with_invalid_params(self):
        """测试参数校验"""
        invalid_params = [
            ({"session_num": 0}, "session_num=0 should fail"),
            ({"micro_batch_num": 0}, "micro_batch_num=0 should fail"),
            (
                {"session_num": 1 << 31, "micro_batch_num": 1 << 31},
                "micro_batch_num mul overflow",
            ),
            ({"micro_batch_size": 0}, "micro_batch_size=0 should fail"),
            (
                {"micro_batch_num": 1 << 31, "micro_batch_size": 1 << 31},
                "micro_batch_size mul overflow",
            ),
            (
                {
                    "schedule_mode": 1,
                    "micro_batch_num": 1 << 31,
                    "micro_batch_size": 1 << 31,
                },
                "attention micro_batch_size mul overflow",
            ),
            ({"selected_expert_num": 0}, "selected_expert_num=0 should fail"),
            (
                {"micro_batch_size": 1 << 31, "selected_expert_num": 1 << 31},
                "selected_expert_num mul overflow",
            ),
            (
                {
                    "schedule_mode": 1,
                    "micro_batch_size": 1 << 31,
                    "selected_expert_num": 1 << 31,
                },
                "attention selected_expert_num mul overflow",
            ),
            ({"ffn_window": 0}, "ffn_window=0 should fail"),
            ({"ffn_window_size": 0}, "ffn_window_size can not be 0"),
            ({"ffn_window_size": 511}, "ffn_window_size is not enough should fail"),
            ({"schedule_mode": 1, "attention_window": 0}, "ffn_window is null"),
            (
                {"schedule_mode": 1, "attention_window_size": 0},
                "attention_window_size can not be 0",
            ),
            (
                {"schedule_mode": 1, "attention_window_size": 511},
                "attention_window_size is not enough should fail",
            ),
            ({"schedule_mode": 2}, "schedule_mode 2 is not supportted"),
            (
                {"attn_to_ffn_token_size": 1023},
                "attn_to_ffn_token_size must be aligned by 512",
            ),
            (
                {"ffn_to_attn_token_size": 400},
                "ffn_to_attn_token_size must be aligned by 512",
            ),
        ]
        for params, msg in invalid_params:
            with self.subTest(msg=msg):
                test_params = self.default_params.copy()
                test_params.update(params)
                with self.assertRaises(RuntimeError):
                    torch_npu._afd.create_schedule_context_holder(**test_params)

    def test_schedule_ffn(self):
        """测试用有效参数初始化"""
        holder = torch_npu._afd.create_schedule_context_holder(
            **self.default_params
        )
        self.assertIsInstance(holder, torch_npu._afd.ScheduleContextHolder)

        # 获取tensor
        tensor = holder.get_schedule_context_tensor()
        self.assertIsInstance(tensor, torch.Tensor)

        context_info = holder.get_schedule_context_info()
        self.assertIn("ffn info:", context_info)

        holder.stop_schedule()

    def test_schedule_attn(self):
        """测试用有效参数初始化"""
        test_params = self.default_params.copy()
        test_params["schedule_mode"] = 1
        holder = torch_npu._afd.create_schedule_context_holder(**test_params)
        self.assertIsInstance(holder, torch_npu._afd.ScheduleContextHolder)

        # 获取tensor
        tensor = holder.get_schedule_context_tensor()
        self.assertIsInstance(tensor, torch.Tensor)
        holder.stop_schedule()


if __name__ == "__main__":
    run_tests()
