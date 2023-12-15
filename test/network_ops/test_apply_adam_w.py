# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestApplyAdam(TestCase):

    def cpu_op_exec(self, var, m, v, beta1_power, beta2_power, lr, weight_decay,
                    beta1, beta2, eps, grad, max_grad_norm, amsgrad, maximize):
        if amsgrad:
            max_grad_norm = np.random.uniform(-5.0, 5.0, var_shape).astype(dtype)
        gt = -grad if maximize else grad
        m_out = m * beta1 - (beta1 + (-1)) * gt
        v_out = v * beta2 - (beta2 + (-1)) * gt * gt
        var_t = var * (1 + (-lr * weight_decay))
        beta1_power_out = beta1_power * beta1
        beta2_power_out = beta2_power * beta2
        if amsgrad:
            max_grad_norm_out = np.maximum(max_grad_norm, v_out)
            denom = np.sqrt(max_grad_norm_out / (1 - beta2_power_out)) + eps
        else:
            max_grad_norm_out = None
            denom = np.sqrt(v_out / (1 - beta2_power_out)) + eps
        var_out = var_t + (-lr * m_out / (1 - beta1_power_out)) / denom
        return var_out, m_out, v_out

    def npu_op_exec(self, var_tensor, m_tensor, v_tensor, beta1_power, beta2_power, lr, weight_decay,
                    beta1, beta2, eps, grad, max_grad_norm, amsgrad, maximize):
        var_out_npu, m_out_npu, v_out_npu = torch_npu.npu_apply_adam_w(beta1_power[0],
                                                                       beta2_power[0], lr[0], weight_decay[0], beta1[0],
                                                                       beta2[0], eps[0], grad, max_grad_norm,
                                                                       amsgrad, maximize,
                                                                       out=(var_tensor, m_tensor, v_tensor))
        return var_out_npu, m_out_npu, v_out_npu

    def test_apply_adam_w_maximize_true(self):
        amsgrad = False  # at present, the operator supports only false.
        maximize = True
        scalar_shape = [1]
        dtype = np.float32
        shape_format = [
            [np.float32, 2, (21130, 512)],
        ]
        var_cpu, var_npu = create_common_tensor(shape_format[0], 10.0, 20.0)
        m_cpu, m_npu = create_common_tensor(shape_format[0], 5.0, 10.0)
        v_cpu, v_npu = create_common_tensor(shape_format[0], 0.1, 5.0)
        grad_cpu, grad_npu = create_common_tensor(shape_format[0], -5.0, 5.0)

        var_cpu = var_cpu.numpy()
        m_cpu = m_cpu.numpy()
        v_cpu = v_cpu.numpy()
        grad_cpu = grad_cpu.numpy()

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape).astype(dtype)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape).astype(dtype)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape).astype(dtype)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape).astype(dtype)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape).astype(dtype)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape).astype(dtype)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape).astype(dtype)
        max_grad_norm = None

        var_ret_cpu, m_ret_cpu, v_ret_cpu = self.cpu_op_exec(var_cpu, m_cpu, v_cpu, beta1_power, beta2_power, lr,
                                                             weight_decay, beta1, beta2, eps, grad_cpu, max_grad_norm,
                                                             amsgrad, maximize)

        var_ret_npu, m_ret_npu, v_ret_npu = self.npu_op_exec(var_npu, m_npu, v_npu, beta1_power,
                                                             beta2_power, lr, weight_decay, beta1, beta2, eps, grad_npu,
                                                             max_grad_norm, amsgrad, maximize)

        self.assertRtolEqual(var_ret_cpu, var_ret_npu.cpu().numpy())
        self.assertRtolEqual(m_ret_cpu, m_ret_npu.cpu().numpy())
        self.assertRtolEqual(v_ret_cpu, v_ret_npu.cpu().numpy())

    def test_apply_adam_w_maximize_false(self):
        amsgrad = False  # at present, the operator supports only false.
        maximize = False
        scalar_shape = [1]
        dtype = np.float32
        shape_format = [
            [np.float32, 2, (21130, 512)],
        ]
        var_cpu, var_npu = create_common_tensor(shape_format[0], 10.0, 20.0)
        m_cpu, m_npu = create_common_tensor(shape_format[0], 5.0, 10.0)
        v_cpu, v_npu = create_common_tensor(shape_format[0], 0.1, 5.0)
        grad_cpu, grad_npu = create_common_tensor(shape_format[0], -5.0, 5.0)

        var_cpu = var_cpu.numpy()
        m_cpu = m_cpu.numpy()
        v_cpu = v_cpu.numpy()
        grad_cpu = grad_cpu.numpy()

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape).astype(dtype)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape).astype(dtype)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape).astype(dtype)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape).astype(dtype)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape).astype(dtype)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape).astype(dtype)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape).astype(dtype)
        max_grad_norm = None

        var_ret_cpu, m_ret_cpu, v_ret_cpu = self.cpu_op_exec(var_cpu, m_cpu, v_cpu, beta1_power, beta2_power, lr,
                                                             weight_decay, beta1, beta2, eps, grad_cpu, max_grad_norm,
                                                             amsgrad, maximize)

        var_ret_npu, m_ret_npu, v_ret_npu = self.npu_op_exec(var_npu, m_npu, v_npu, beta1_power,
                                                             beta2_power, lr, weight_decay, beta1, beta2, eps, grad_npu,
                                                             max_grad_norm, amsgrad, maximize)

        self.assertRtolEqual(var_ret_cpu, var_ret_npu.cpu().numpy())
        self.assertRtolEqual(m_ret_cpu, m_ret_npu.cpu().numpy())
        self.assertRtolEqual(v_ret_cpu, v_ret_npu.cpu().numpy())


if __name__ == "__main__":
    run_tests()
