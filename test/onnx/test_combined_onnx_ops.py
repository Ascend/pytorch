import os
import shutil
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torch_npu.onnx
from torch_npu.utils._path_manager import PathManager


class TestOnnxOps(TestCase):

    test_onnx_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "test_onnx_combined")

    @classmethod
    def setUpClass(cls):
        PathManager.make_dir_safety(TestOnnxOps.test_onnx_path)

    @classmethod
    def tearDownClass(cls):
        assert os.path.exists(TestOnnxOps.test_onnx_path)
        PathManager.remove_path_safety(TestOnnxOps.test_onnx_path)

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

        torch.npu.config.allow_internal_format = True
        torch.npu.set_compile_mode(jit_compile=True)

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

    def test_wrapper_npu_layer_norm_eval(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                normalized_shape = 4
                self.elementwise_affine = True
                if self.elementwise_affine:
                    self.weight = Parameter(torch.empty(normalized_shape))
                    self.bias = Parameter(torch.empty(normalized_shape))
                else:
                    self.register_parameter('weight', None)
                    self.register_parameter('bias', None)

                self.reset_parameters()

            def reset_parameters(self) -> None:
                if self.elementwise_affine:
                    init.ones_(self.weight)
                    init.zeros_(self.bias)

            def forward(self, input_):
                normalized_shape = input_.size()[1:]
                eps = 1e-5
                return torch_npu.npu_layer_norm_eval(input_, normalized_shape,
                                                     self.weight, self.bias, eps)

        def export_onnx(onnx_model_name):
            input_ = torch.rand((6, 4), dtype=torch.float32).npu()
            model = Model().to("npu")
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_layer_norm_eval.onnx"
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

    def test_wrapper_npu_fused_attention_layernorm_qkv_fwd(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).to(dtype=to_dtype).npu()
                self.k_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).to(dtype=to_dtype).npu()
                self.v_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).to(dtype=to_dtype).npu()
                self.q_bias = torch.rand(1024).to(dtype=to_dtype).npu()
                self.k_bias = torch.rand(1024).to(dtype=to_dtype).npu()
                self.v_bias = torch.rand(1024).to(dtype=to_dtype).npu()

            def forward(self, input_, gamma, beta):
                return torch_npu.npu_fused_attention_layernorm_qkv_fwd(input_,
                                                                       self.q_weight, self.k_weight, self.v_weight,
                                                                       gamma, beta,
                                                                       self.q_bias, self.k_bias, self.v_bias,
                                                                       seq_len=512, num_heads=16, eps=1e-05)

        def export_onnx(onnx_model_name):
            input_ = torch.rand(12288, 1024).uniform_(-6, 6).to(dtype=to_dtype).npu()
            gamma = torch.rand(1024).to(dtype=to_dtype).npu()
            beta = torch.rand(1024).to(dtype=to_dtype).npu()
            model = Model().to("npu")
            self.onnx_export(model, (input_, gamma, beta), onnx_model_name,
                             ["input_", "gamma", "beta"], ["o_1", "o_2", "o_3", "o_4", "o_5", "o_6"])

        to_dtype = torch.float16
        onnx_model_name = "model_npu_fused_attention_layernorm_qkv_fwd.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))


if __name__ == "__main__":
    run_tests()
