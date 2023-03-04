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

    def test_wrapper_npu_stride_copy(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_):
                shape = (2, 2)
                stride = (1, 2)
                storage_offset = 0
                return torch_npu.npu_stride_copy(input_, shape, stride, storage_offset)

        def export_onnx(onnx_model_name):
            input_ = torch.rand(3, 3).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_stride_copy.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, 
                                            onnx_model_name)))

    def test_wrapper_npu_sort_v2(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_):
                dim = -1
                descending = False
                return torch_npu.npu_sort_v2(input_, dim, descending)

        def export_onnx(onnx_model_name):
            input_ = torch.randn(3, 4).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_sort_v2.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, 
                                            onnx_model_name)))

    def test_wrapper_npu_reshape(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_):
                shape = (4, 4)
                can_refresh = False
                return torch_npu.npu_reshape(input_, shape, can_refresh)

        def export_onnx(onnx_model_name):
            input_ = torch.rand(2, 8).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_reshape.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, 
                                            onnx_model_name)))

    def test_wrapper_npu_pad(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_):
                paddings = [1, 1, 1, 1]
                return torch_npu.npu_pad(input_, paddings)

        def export_onnx(onnx_model_name):
            input_ = torch.tensor([[20, 20, 10, 10]],
                                  dtype=torch.float16).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"], ["output_name"])

        onnx_model_name = "model_npu_pad.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_convolution(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(torch.empty([64, 32, 3, 3]))
                bias = True
                if bias:
                    self.bias = Parameter(torch.empty([64]))
                else:
                    self.register_parameter('bias', None)
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
                return torch_npu.npu_convolution(input_, self.weight, self.bias,
                                                 stride, paddings, dilation, groups)

        def export_onnx(onnx_model_name):
            input_ = torch.rand([16, 32, 24, 24]).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_convolution.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_convolution_transpose(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(torch.empty([3, 2, 3, 3]))
                bias = True
                if bias:
                    self.bias = Parameter(torch.empty([2]))
                else:
                    self.register_parameter('bias', None)
                self.reset_parameters()

            def reset_parameters(self):
                init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        init.uniform_(self.bias, -bound, bound)

            def forward(self, input_):
                padding, output_padding, stride, dilation, groups = \
                    [1, 1], [0, 0], [1, 1], [1, 1], 1
                return torch_npu.npu_convolution_transpose(input_, self.weight, self.bias, padding,
                                                           output_padding, stride, dilation, groups)

        def export_onnx(onnx_model_name):
            input_ = torch.rand([1, 3, 3, 3]).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_convolution_transpose.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, 
                                            onnx_model_name)))

    def test_wrapper_npu_confusion_transpose(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_):
                perm = (0, 2, 1, 3)
                shape = [1, 576, 32, 80]
                transpose_first = False
                return torch_npu.npu_confusion_transpose(input_, perm, shape, transpose_first)

        def export_onnx(onnx_model_name):
            input_ = torch.rand([1, 576, 2560]).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_confusion_transpose.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_max(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_):
                dim = 2
                return torch_npu.npu_max(input_, dim)

        def export_onnx(onnx_model_name):
            input_ = torch.rand((2, 2, 2, 2)).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, 
                             ["input_"], ["values", "indices"])

        onnx_model_name = "model_npu_max.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_bmmV2(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_, mat2):
                output_sizes = []
                return torch_npu.npu_bmmV2(input_, mat2, output_sizes)

        def export_onnx(onnx_model_name):
            input_ = torch.rand((10, 3, 4)).npu()
            mat2 = torch.rand((10, 4, 5)).npu()
            model = Model().to("npu")
            self.onnx_export(model, (input_, mat2),
                             onnx_model_name, ["input_", "mat2"])

        onnx_model_name = "model_npu_bmmV2.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_dtype_cast(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_):
                return torch_npu.npu_dtype_cast(input_, torch.int32)

        def export_onnx(onnx_model_name):
            input_ = torch.rand((8, 10)).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_dtype_cast.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_silu(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_):
                return torch_npu.npu_silu(input_)

        def export_onnx(onnx_model_name):
            input_ = torch.randn(5, 5).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_silu.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_mish(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_):
                return torch_npu.npu_mish(input_)

        def export_onnx(onnx_model_name):
            input_ = torch.randn(5, 5).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_mish.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_min(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_):
                dim = 2
                return torch_npu.npu_min(input_, dim)

        def export_onnx(onnx_model_name):
            input_ = torch.rand((2, 2, 2, 2)).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, 
                             ["input_"], ["values", "indices"])

        onnx_model_name = "model_npu_min.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))


    def test_wrapper_npu_scaled_masked_softmax(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_, mask):
                scale = 0.56
                fixed_triu_mask = False
                return torch_npu.npu_scaled_masked_softmax(input_, mask,
                                                           scale, fixed_triu_mask)

        def export_onnx(onnx_model_name):
            input_ = torch.rand((4, 3, 64, 64)).npu()
            mask = torch.rand((4, 3, 64, 64)).npu() > 0
            model = Model().to("npu")
            self.onnx_export(model, (input_, mask), onnx_model_name,
                             ["input_", "mask"])

        onnx_model_name = "model_npu_scaled_masked_softmax.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))


if __name__ == "__main__":
    run_tests()
