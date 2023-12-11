# Copyright (c) 2021, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

env_device_cnt = torch_npu.npu.device_count()

error_index_list_str = map(lambda x: "npu:" + str(x), [i for i in range(env_device_cnt, env_device_cnt + 3)])
error_index_legacy = [i for i in range(env_device_cnt, env_device_cnt + 3)]
error_device_input = map(lambda x: torch.device("npu:{}".format(x)),
                         [i for i in range(env_device_cnt, env_device_cnt + 3)])

error_input_list = ["npu1", "1npu", "npu::1"]
error_index_list = list(error_index_list_str) + error_index_legacy + list(error_device_input)
device_ofr_info = "Invalid NPU device ordinal. Valid device ordinal ranges from 0 - {}.".format(env_device_cnt - 1)


class TestPtaErrorInterception(TestCase):

    def test_tensor(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.tensor([1], device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.tensor([1], device=err_input)

    def test_full(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.full([2, 3], 2, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.full([2, 3], 2, device=err_input)

    def test_randint(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.randint([2, 3], 2, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.randint([2, 3], 2, device=err_input)

    def test_range(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.range(1, 3, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.range(1, 3, device=err_input)

    def test_arange(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.arange(1, 3, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.arange(1, 3, device=err_input)

    def test_npu_dropout_gen_mask(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch_npu.npu_dropout_gen_mask([1, 1], p=1.0, dtype=torch.float, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch_npu.npu_dropout_gen_mask([1, 1], p=1.0, dtype=torch.float, device=err_input)

    def test_bartlett_window(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.bartlett_window(window_length=1, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.bartlett_window(window_length=1, device=err_input)

    def test_blackman_window(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.blackman_window(window_length=1, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.blackman_window(window_length=1, device=err_input)

    def test_empty(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.empty(10, dtype=torch.int8, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.empty(10, dtype=torch.int8, device=err_input)

    def test_empty_like(self):
        in_tensor = torch.tensor(1, device="npu:0")
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.empty_like(in_tensor, dtype=torch.int8, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.empty_like(in_tensor, dtype=torch.int8, device=err_input)

    def test_empty_strided(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.empty_strided((2, 1, 5, 4), (2, 4, 3, 7), torch.float32, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.empty_strided((2, 1, 5, 4), (2, 4, 3, 7), torch.float32, device=err_input)

    def test_eye(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.eye(1, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.eye(1, device=err_input)

    def test_full_like(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.full_like(torch.tensor([1, 1], device="npu"), 5, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.full_like(torch.tensor([1, 1], device="npu"), 5, device=err_input)

    def test_hamming_window(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.hamming_window(5, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.hamming_window(5, device=err_input)

    def test_hann_window(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.hann_window(5, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.hann_window(5, device=err_input)

    def test_kaiser_window(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.kaiser_window(5, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.kaiser_window(5, device=err_input)

    def test_linspace(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.linspace(3, 10, steps=5, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.linspace(3, 10, steps=5, device=err_input)

    def test_logspace(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.logspace(0.0, 1.0, 10, 0.2, torch.float32, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.logspace(0.0, 1.0, 10, 0.2, torch.float32, device=err_input)

    def test_normal(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.normal(mean=1.0, std=1.0, size=[5, 5], device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.normal(mean=1.0, std=1.0, size=[5, 5], device=err_input)

    def test_ones(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.ones(2, 3, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.ones(2, 3, device=err_input)

    def test_ones_like(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.ones_like(torch.empty(2, 3), device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.ones_like(torch.empty(2, 3), device=err_input)

    def test_rand(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.rand(2, 3, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.rand(2, 3, device=err_input)

    def test_rand_like(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.rand_like(torch.rand(2, 3), device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.rand_like(torch.rand(2, 3), device=err_input)

    def randint_like(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.randint_like(torch.rand(2, 3), device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.randint_like(torch.rand(2, 3), device=err_input)

    def test_randn(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.randn(3, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.randn(3, device=err_input)

    def test_randn_like(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.randn_like(torch.randn(3, 3), device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.randn_like(torch.randn(3, 3), device=err_input)

    def test_randperm(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.randperm(10, dtype=torch.float, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.randperm(10, dtype=torch.float, device=err_input)

    def test_tril_indices(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.tril_indices(3, 3, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.tril_indices(3, 3, device=err_input)

    def test_triu_indices(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.triu_indices(3, 3, device=err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.triu_indices(3, 3, device=err_input)

    def test_zeros(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.zeros((3, 3), device=err_idx, dtype=torch.float)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.zeros((3, 3), device=err_input, dtype=torch.float)

    def test_zeros_like(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.zeros_like(torch.zeros(3), device=err_idx, dtype=torch.float)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.zeros_like(torch.zeros(3), device=err_input, dtype=torch.float)

    def test_set_device(self):
        for err_idx in error_index_list:
            with self.assertRaisesRegex(RuntimeError, device_ofr_info):
                torch.npu.set_device(err_idx)
        for err_input in error_input_list:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string: {}".format(err_input)):
                torch.npu.set_device(err_input)


if __name__ == "__main__":
    run_tests()
