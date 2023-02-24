# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
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

import os
import shutil
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torch_npu.onnx


class TestOnnxOps(TestCase):
    test_onnx_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "test_onnx_combined")

    @classmethod
    def setUpClass(cls):
        os.makedirs(TestOnnxOps.test_onnx_path, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        assert os.path.exists(TestOnnxOps.test_onnx_path)
        shutil.rmtree(TestOnnxOps.test_onnx_path, ignore_errors=True)

    def onnx_export(self, model, inputs, onnx_model_name,
                    input_names=None, output_names=None):
        if input_names is None:
            input_names = ["input_names"]
        if output_names is None:
            output_names = ["output_names"]
        model.eval()
        with torch.no_grad():
            torch.onnx.export(model, inputs,
                              os.path.join(
                                  TestOnnxOps.test_onnx_path, onnx_model_name),
                              opset_version=11, input_names=input_names,
                              output_names=output_names)

    def test_wrapper_npu_linear(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.weight = Parameter(torch.empty(3, 16))
                bias = True
                if bias:
                    self.bias = Parameter(torch.empty(3))
                else:
                    self.register_parameter("bias", None)

                self.reset_parameters()

            def reset_parameters(self):
                init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(self.bias, -bound, bound)

            def forward(self, input_):
                return torch_npu.npu_linear(input_, self.weight, self.bias)

        def export_onnx(onnx_model_name):
            input_ = torch.randn(4, 16).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_linear.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_transpose(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_):
                perm = (2, 0, 1)
                require_contiguous = False
                return torch_npu.npu_transpose(input_, perm, require_contiguous)

        def export_onnx(onnx_model_name):
            input_ = torch.randn(2, 3, 5).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_transpose.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_broadcast(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_):
                size = (3, 4)
                return torch_npu.npu_broadcast(input_, size)

        def export_onnx(onnx_model_name):
            input_ = torch.tensor([[1], [2], [3]]).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_broadcast.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_one_(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_):
                return torch_npu.one_(input_)

        def export_onnx(onnx_model_name):
            input_ = torch.rand(3, 4).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_one_.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_conv_transpose2d(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.weight = Parameter(torch.empty([3, 2, 3, 3]))
                bias = True
                if bias:
                    self.bias = Parameter(torch.empty(2))
                else:
                    self.register_parameter("bias", None)

                self.reset_parameters()

            def reset_parameters(self):
                init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        init.uniform_(self.bias, -bound, bound)

            def forward(self, input_):
                stride, paddings, output_padding, dilation, groups = \
                    [1, 1], [0, 0], [0, 0], [1, 1], 1
                return torch_npu.npu_conv_transpose2d(input_, self.weight, self.bias,
                                                      paddings, output_padding, stride, dilation, groups)

        def export_onnx(onnx_model_name):
            input_ = torch.rand([1, 3, 3, 3]).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_conv_transpose2d.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_conv2d(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.weight = Parameter(torch.empty([256, 128, 3, 3]))
                bias = True
                if bias:
                    self.bias = Parameter(torch.empty(256))
                else:
                    self.register_parameter("bias", None)

                self.reset_parameters()

            def reset_parameters(self):
                init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        init.uniform_(self.bias, -bound, bound)

            def forward(self, input_):
                stride, paddings, dilation, groups = \
                    [1, 1], [1, 1], [1, 1], 1
                return torch_npu.npu_conv2d(input_, self.weight, self.bias,
                                            stride, paddings, dilation, groups)

        def export_onnx(onnx_model_name):
            input_ = torch.rand([16, 128, 112, 112]).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_conv2d.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_conv3d(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.weight = Parameter(torch.empty([1, 128, 3, 3, 3]))
                bias = False
                if bias:
                    self.bias = Parameter(torch.empty(1))
                else:
                    self.register_parameter("bias", None)

                self.reset_parameters()

            def reset_parameters(self):
                init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        init.uniform_(self.bias, -bound, bound)

            def forward(self, input_):
                stride, paddings, dilation, groups = \
                    [1, 1, 1], [1, 1, 1], [1, 1, 1], 1
                return torch_npu.npu_conv3d(input_, self.weight, self.bias,
                                            stride, paddings, dilation, groups)

        def export_onnx(onnx_model_name):
            input_ = torch.rand([1, 128, 4, 14, 14]).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_conv3d.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))


if __name__ == "__main__":
    run_tests()
