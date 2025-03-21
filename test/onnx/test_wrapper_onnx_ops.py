import os
import shutil
import unittest
import numpy as np
import torch

import torch_npu
import torch_npu.onnx
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.utils._path_manager import PathManager

# acl format
FORMAT_ND = 2
FORMAT_NZ = 29


class TestOnnxOps(TestCase):

    test_onnx_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "test_onnx_wrapper_ops")

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
        OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
        with torch.no_grad():
            torch.onnx.export(model, inputs,
                              os.path.join(TestOnnxOps.test_onnx_path, onnx_model_name),
                              opset_version=11,
                              operator_export_type=OPERATOR_EXPORT_TYPE,
                              input_names=input_names,
                              output_names=output_names)

    def test_wrapper_npu_one_hot(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                x = torch_npu.npu_one_hot(x, depth=5)
                return x

        def export_onnx(onnx_model_name):
            x = torch.IntTensor([5, 3, 2, 1]).npu()
            model = Model().to("npu")
            model(x)
            self.onnx_export(model, x, onnx_model_name)

        onnx_model_name = "model_npu_one_hot.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_slice(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                x = torch_npu.npu_slice(x, [0, 0], [2, 2])
                return x

        def export_onnx(onnx_model_name):
            x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                             dtype=torch.float16).npu()
            model = Model().to("npu")
            model(x)
            self.onnx_export(model, x, onnx_model_name)

        onnx_model_name = "model_npu_slice.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_roi_align(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, rois):
                x = torch_npu.npu_roi_align(x, rois, 0.25, 3, 3, 2, 0)
                return x

        def export_onnx(onnx_model_name):
            x = torch.FloatTensor([[[[1, 2, 3, 4, 5, 6],
                                     [7, 8, 9, 10, 11, 12],
                                     [13, 14, 15, 16, 17, 18],
                                     [19, 20, 21, 22, 23, 24],
                                     [25, 26, 27, 28, 29, 30],
                                     [31, 32, 33, 34, 35, 36]]]]).npu()
            rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
            model = Model().to("npu")
            model(x, rois)
            self.onnx_export(model, (x, rois), onnx_model_name)

        onnx_model_name = "model_npu_roi_align.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_iou(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, bboxes, gtboxes):
                x = torch_npu.npu_iou(bboxes, gtboxes, 0)
                return x

        def export_onnx(onnx_model_name):
            bboxes = torch.tensor([[0, 0, 10, 10],
                                   [10, 10, 20, 20],
                                   [32, 32, 38, 42]], dtype=torch.float16).npu()
            gtboxes = torch.tensor([[0, 0, 10, 20],
                                    [0, 10, 10, 10],
                                    [10, 10, 20, 20]], dtype=torch.float16).npu()
            model = Model().to("npu")
            model(bboxes, gtboxes)
            self.onnx_export(model, (bboxes, gtboxes), onnx_model_name,
                             input_names=["bboxes", "gtboxes"])

        onnx_model_name = "model_npu_iou.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_wrapper_npu_batch_nms(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, boxes, scores):
                x = torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 3, 4)
                return x

        def export_onnx(onnx_model_name):
            boxes = torch.randn(8, 2, 4, 4, dtype=torch.float32).npu()
            scores = torch.randn(3, 2, 4, dtype=torch.float32).npu()
            model = Model().to("npu")
            model(boxes, scores)
            self.onnx_export(model, (boxes, scores),
                             onnx_model_name, input_names=["boxes", "scores"],
                             output_names=["nmsed_boxes", "nmsed_scores",
                             "nmsed_classes", "nmsed_num"])

        onnx_model_name = "model_npu_batch_nms.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_fast_gelu(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                x = torch_npu.fast_gelu(x)
                return x

        def export_onnx(onnx_model_name):
            x = torch.rand(2).npu()
            model = Model().to("npu")
            model(x)
            self.onnx_export(model, x, onnx_model_name)

        onnx_model_name = "model_npu_fast_gelu.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_geglu(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
            
            def forward(self, x):
                return torch_npu.npu_geglu(x)

        def export_onnx(onnx_model_name):
            x = torch.rand(2, 10, 1024).npu().half()
            model = Model().to("npu")
            model(x)
            self.onnx_export(model, x, onnx_model_name, ["input"], ["output1", "output2"])
        
        onnx_model_name = "model_npu_geglu.onnx"
        export_onnx(onnx_model_name)
        assert(os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                           onnx_model_name)))

    def test_wrapper_npu_multi_head_attention(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                batch = 8
                attn_head_num = 16
                attn_dim_per_head = 64
                src_len = 64
                tgt_len = 64
                dropout_prob = 0.5
                softmax_use_float = True
                weight_col = attn_head_num * attn_dim_per_head
                self.attn_head_num = attn_head_num
                self.attn_dim_per_head = attn_dim_per_head
                self.src_len = src_len
                self.tgt_len = tgt_len
                self.dropout_prob = dropout_prob
                self.softmax_use_float = softmax_use_float
                self.query_weight = torch_npu.npu_format_cast(
                    torch.randn((weight_col, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ
                )
                self.key_weight = torch_npu.npu_format_cast(
                    torch.randn((weight_col, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ
                )
                self.value_weight = torch_npu.npu_format_cast(
                    torch.randn((weight_col, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ
                )
                self.out_proj_weight = torch_npu.npu_format_cast(
                    torch.randn((weight_col, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ
                )
                self.attn_mask = torch_npu.npu_format_cast(
                    torch.randn((batch, attn_head_num,
                                 tgt_len, src_len)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_ND
                )
                self.query_bias = torch_npu.npu_format_cast(
                    torch.randn((weight_col, )).uniform_(-1, 1).to(torch.half).npu(), FORMAT_ND
                )
                self.key_bias = torch_npu.npu_format_cast(
                    torch.randn((weight_col, )).uniform_(-1, 1).to(torch.half).npu(), FORMAT_ND
                )
                self.value_bias = torch_npu.npu_format_cast(
                    torch.randn((weight_col, )).uniform_(-1, 1).to(torch.half).npu(), FORMAT_ND
                )
                self.out_proj_bias = torch_npu.npu_format_cast(
                    torch.randn((weight_col, )).uniform_(-1, 1).to(torch.half).npu(), FORMAT_ND
                )
                self.grad = torch_npu.npu_format_cast(
                    torch.randn((batch * tgt_len,
                                 attn_dim_per_head * attn_head_num)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ
                )
                self.mask = (torch.randn((src_len * tgt_len * attn_head_num)).uniform_(-1,
                                                                                       1).npu() > 0).to(torch.uint8)

            def forward(self, query, key, value):
                return torch_npu.npu_multi_head_attention(
                    query, key, value, self.query_weight, self.key_weight, self.value_weight,
                    self.attn_mask, self.out_proj_weight, self.query_bias, self.key_bias,
                    self.value_bias, self.out_proj_bias, self.mask, self.attn_head_num,
                    self.attn_dim_per_head, self.src_len, self.tgt_len, self.dropout_prob,
                    self.softmax_use_float)

        def export_onnx(onnx_model_name):
            batch = 8
            attn_head_num = 16
            attn_dim_per_head = 64
            src_len = 64
            tgt_len = 64
            weight_col = attn_head_num * attn_dim_per_head

            query = torch_npu.npu_format_cast(
                torch.randn((batch * tgt_len, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ
            )
            key = torch_npu.npu_format_cast(
                torch.randn((batch * src_len, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ
            )
            value = torch_npu.npu_format_cast(
                torch.randn((batch * src_len, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ
            )

            model = Model().to("npu")
            model(query, key, value)
            self.onnx_export(model, (query, key, value), onnx_model_name,
                             ["query", "key", "value"])

        onnx_model_name = "model_npu_multi_head_attention.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_diou(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_diou(box1, box2)

        def export_onnx(onnx_model_name):
            box1 = torch.tensor([[0, 0, 10, 10],
                                [10, 10, 20, 20],
                                [32, 32, 38, 42],
                                [32, 32, 38, 42]], dtype=torch.float32).to("npu")
            box2 = torch.tensor([[0, 0, 10, 20],
                                [0, 10, 10, 10],
                                [10, 10, 20, 20],
                                [10, 10, 20, 20]], dtype=torch.float32).to("npu")
            model = Model().to("npu")
            model(box1, box2)
            self.onnx_export(model, (box1, box2), onnx_model_name, ["box1", "box2"])

        onnx_model_name = "model_npu_diou.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_ciou(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_ciou(box1, box2, False, False)

        def export_onnx(onnx_model_name):
            box1 = torch.rand(4, 8).npu()
            box2 = torch.rand(4, 8).npu()
            model = Model().to("npu")
            model(box1, box2)
            self.onnx_export(model, (box1, box2), onnx_model_name, ["box1", "box2"])

        onnx_model_name = "model_npu_ciou.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_giou(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_giou(box1, box2, True, False, 0)

        def export_onnx(onnx_model_name):
            box1 = torch.tensor([[0.4375, 0.0041, 0.4893, 0.4176],
                                [0.1618, 0.1920, 0.4528, 0.4363],
                                [0.7243, 0.6361, 0.8139, 0.7649],
                                [0.9430, 0.6788, 0.6872, 0.8605]]).npu()
            box2 = torch.tensor([[0.1625, 0.4915, 0.4430, 0.1314],
                                [0.2110, 0.0042, 0.2204, 0.0087],
                                [0.7917, 0.5444, 0.5964, 0.9363],
                                [0.7733, 0.7770, 0.7666, 0.8029]]).npu()
            model = Model().to("npu")
            model(box1, box2)
            self.onnx_export(model, (box1, box2), onnx_model_name, ["box1", "box2"])

        onnx_model_name = "model_npu_giou.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_deformable_conv2d(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.weight = torch.rand((32, 32, 5, 5)).npu()
                self.offset = torch.rand((16, 75, 32, 32)).npu()

            def forward(self, input_):
                return torch_npu.npu_deformable_conv2d(input_, self.weight, self.offset,
                                                       None, kernel_size=[5, 5], stride=[1, 1, 1, 1], padding=[2, 2, 2, 2])

        def export_onnx(onnx_model_name):
            input_ = torch.rand(16, 32, 32, 32).npu()
            model = Model().to("npu")
            model(input_)
            self.onnx_export(model, input_, onnx_model_name, ["input_"], ["out1", "out2"])

        onnx_model_name = "model_npu_deformable_conv2d.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_format_cast(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_):
                return torch_npu.npu_format_cast(input_, 2)

        def export_onnx(onnx_model_name):
            input_ = torch.rand(3, 3).npu()
            model = Model().to("npu")
            model(input_)
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_format_cast.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_softmax_cross_entropy_with_logits(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_, label):
                return torch_npu.npu_softmax_cross_entropy_with_logits(input_, label)

        def export_onnx(onnx_model_name):
            input_ = torch.tensor([[1., 2., 3., 4.]]).npu()
            label = torch.tensor([[1., 2., 3., 4.]]).npu()
            model = Model().to("npu")
            model(input_, label)
            self.onnx_export(model, (input_, label), onnx_model_name, ["input_", "label"])

        onnx_model_name = "model_npu_softmax_cross_entropy_with_logits.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_ps_roi_pooling(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_, roi):
                return torch_npu.npu_ps_roi_pooling(input_, roi, 0.5, 2, 2)

        def export_onnx(onnx_model_name):
            input_ = torch.tensor([[[[1]], [[2]], [[3]], [[4]],
                                    [[5]], [[6]], [[7]], [[8]]],
                                   [[[9]], [[10]], [[11]], [[12]],
                                    [[13]], [[14]], [[15]], [[16]]]
                                   ], dtype=torch.float16).npu()
            roi = torch.tensor([[[1], [2], [3], [4], [5]],
                                [[6], [7], [8], [9], [10]]
                                ], dtype=torch.float16).npu()

            model = Model().to("npu")
            model(input_, roi)
            self.onnx_export(model, (input_, roi), onnx_model_name, ["input_", "roi"])

        onnx_model_name = "model_npu_ps_roi_pooling.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_grid_assign_positive(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, assigned_gt_inds, overlaps, box_responsible_flags, max_overlap,
                        argmax_overlap, gt_max_overlaps, gt_argmax_overlaps):
                return torch_npu.npu_grid_assign_positive(assigned_gt_inds, overlaps,
                                                          box_responsible_flags, max_overlap, argmax_overlap, gt_max_overlaps,
                                                          gt_argmax_overlaps, 128, 0.5, 0.0, True)

        def export_onnx(onnx_model_name):
            assigned_gt_inds = torch.rand((4,), dtype=torch.float32).to("npu")
            overlaps = torch.rand((2, 4), dtype=torch.float32).to("npu")
            box_responsible_flags = torch.tensor([1, 1, 1, 0], dtype=torch.uint8).to("npu")
            max_overlap = torch.rand((4,), dtype=torch.float32).to("npu")
            argmax_overlap = torch.tensor([1, 0, 1, 0], dtype=torch.int32).to("npu")
            gt_max_overlaps = torch.rand((2,), dtype=torch.float32).to("npu")
            gt_argmax_overlaps = torch.tensor([1, 0], dtype=torch.int32).to("npu")
            model = Model().to("npu")
            model(assigned_gt_inds, overlaps, box_responsible_flags, max_overlap,
                  argmax_overlap, gt_max_overlaps, gt_argmax_overlaps)
            self.onnx_export(model, (assigned_gt_inds, overlaps, box_responsible_flags, max_overlap,
                                     argmax_overlap, gt_max_overlaps, gt_argmax_overlaps), onnx_model_name)

        onnx_model_name = "model_npu_grid_assign_positive.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_wrapper_npu_ifmr(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_, min_value, max_value, cdf):
                return torch_npu.npu_ifmr(input_, min_value, max_value, cdf,
                                          0.999999, 0.999999, 0.7, 1.3, 0.01, True)

        def export_onnx(onnx_model_name):
            input_ = torch.rand(3, 3).npu()
            min_value = torch.reshape(torch.min(input_), (1, ))
            max_value = torch.reshape(torch.max(input_), (1, ))
            hist = torch.histc(input_.to('cpu'),
                               bins=128,
                               min=min_value[0].to('cpu'),
                               max=max_value[0].to('cpu'))
            cdf = torch.cumsum(hist, dim=0).int()
            cdf = cdf.to('npu')
            model = Model().to("npu")
            model(input_, min_value, max_value, cdf)
            self.onnx_export(model, (input_, min_value, max_value, cdf), onnx_model_name,
                             ["input_", "min_value", "max_value", "cdf"], ["out1", "out2"])

        onnx_model_name = "model_npu_ifmr.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_sign_bits_unpack(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_):
                size = 79
                return torch_npu.npu_sign_bits_unpack(input_, size, torch.float32)

        def export_onnx(onnx_model_name):
            input_ = torch.randn(4424).uniform_(0, 255).to(torch.uint8).npu()

            model = Model().to("npu")
            model(input_)
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_sign_bits_unpack.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_ptiou(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, bboxs, gtboxes):
                return torch_npu.npu_ptiou(bboxs, gtboxes)

        def export_onnx(onnx_model_name):
            bboxs = torch.tensor([[1, 2, 3, 4],
                                  [5, 6, 7, 8],
                                  [9, 10, 11, 12],
                                  [13, 14, 15, 16]], dtype=torch.float16).npu()
            gtboxes = torch.tensor([[1, 2, 3, 4],
                                    [5, 6, 7, 8]], dtype=torch.float16).npu()

            model = Model().to("npu")
            model(bboxs, gtboxes)
            self.onnx_export(model, (bboxs, gtboxes), onnx_model_name,
                             ["bboxs", "gtboxes"])

        onnx_model_name = "model_npu_ptiou.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_normalize_batch(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_, seq_len):
                return torch_npu.npu_normalize_batch(input_, seq_len)

        def export_onnx(onnx_model_name):
            input_ = torch.rand([32, 3, 6]).npu()
            seq_len = torch.rand(32).uniform_(3, 32).npu().to(torch.int32)
            model = Model().to("npu")
            model(input_, seq_len)
            self.onnx_export(model, (input_, seq_len), onnx_model_name,
                             ["input_", "seq_len"])

        onnx_model_name = "model_npu_normalize_batch.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_nms_v4(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, boxes, scores, iou_threshold, scores_threshold):
                max_output_size = 20
                return torch_npu.npu_nms_v4(boxes, scores, max_output_size,
                                            iou_threshold, scores_threshold)

        def export_onnx(onnx_model_name):
            boxes = torch.rand((100, 4)).uniform_(0, 100).npu()
            scores = torch.rand(100).uniform_(0, 1).npu()
            iou_threshold = torch.tensor(0.5).npu()
            scores_threshold = torch.tensor(0.3).npu()
            model = Model().to("npu")
            model(boxes, scores, iou_threshold, scores_threshold)
            self.onnx_export(model, (boxes, scores, iou_threshold, scores_threshold),
                             onnx_model_name, ["boxes", "scores", "iou_threshold",
                                               "scores_threshold"], ["out1", "out2"])

        onnx_model_name = "model_npu_nms_v4.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_bounding_box_decode(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input1, input2):
                return torch_npu.npu_bounding_box_decode(input1, input2,
                                                         0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)

        def export_onnx(onnx_model_name):
            input1 = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]]).to("npu").half()
            input2 = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]]).to("npu").half()
            model = Model().to("npu")
            model(input1, input2)
            self.onnx_export(model, (input1, input2),
                             onnx_model_name, ["input1", "input2"])

        onnx_model_name = "model_npu_bounding_box_decode.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_bounding_box_encode(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input1, input2):
                return torch_npu.npu_bounding_box_encode(input1, input2,
                                                         0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)

        def export_onnx(onnx_model_name):
            input1 = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]]).to("npu").half()
            input2 = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]]).to("npu").half()
            model = Model().to("npu")
            model(input1, input2)
            self.onnx_export(model, (input1, input2),
                             onnx_model_name, ["input1", "input2"])

        onnx_model_name = "model_npu_bounding_box_encode.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_nms_with_mask(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input1):
                return torch_npu.npu_nms_with_mask(input1, 0.5)

        def export_onnx(onnx_model_name):
            input1 = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.6],
                                   [6.0, 7.0, 8.0, 9.0, 0.4]]).npu()
            model = Model().to("npu")
            model(input1)
            self.onnx_export(model, input1,
                             onnx_model_name, ["input1"],
                             ["out1", "out2", "out3"])

        onnx_model_name = "model_npu_nms_with_mask.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_rotated_iou(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_rotated_iou(box1, box2, False, 0, True, 0.0, 0.0)

        def export_onnx(onnx_model_name):
            box1 = torch.tensor([[[27.6608, 44.8843, 13.0000, 17.0000, 71.0000],
                                  [37.5091, 51.4143, 6.0000, 17.0000, -104.0000]],
                                 [[51.1990, 30.9037, 10.0000, 16.0000, 113.0000],
                                  [31.0586, 52.0749, 19.0000, 17.0000, 110.0000]]]).npu()
            box2 = torch.tensor([[[31.6291, 29.8558, 12.0000, 15.0000, 148.0000],
                                  [49.5315, 55.5690, 18.0000, 14.0000, 56.0000],
                                  [59.4856, 24.6977, 1.0000, 13.0000, 146.0000]],
                                 [[35.7513, 38.1092, 6.0000, 18.0000, -94.0000],
                                  [41.5259, 51.6249, 6.0000, 14.0000, 123.0000],
                                  [38.6335, 37.4133, 17.0000, 10.0000, -3.0000]]]).npu()

            model = Model().to("npu")
            model(box1, box2)
            self.onnx_export(model, (box1, box2),
                             onnx_model_name, ["box1", "box2"])

        onnx_model_name = "model_npu_rotated_iou.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_rotated_overlaps(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_rotated_overlaps(box1, box2, False)

        def export_onnx(onnx_model_name):
            box1 = torch.tensor([[[35.7500, 48.6562, 12.0000, 13.0000, 66.0000],
                                  [43.1250, 53.5625, 17.0000, 6.0000, -130.0000],
                                  [53.4062, 38.1875, 17.0000, 10.0000, 60.0000]
                                  ]]).npu()

            box2 = torch.tensor([[[43.2812, 30.6719, 13.0000, 2.0000, -73.0000],
                                  [38.7188, 37.4062, 12.0000, 12.0000, -99.0000],
                                  [52.1562, 56.6875, 18.0000, 15.0000, 163.0000],
                                  [59.6250, 33.5312, 8.0000, 11.0000, 89.0000]
                                  ]]).npu()
            model = Model().to("npu")
            model(box1, box2)
            self.onnx_export(model, (box1, box2),
                             onnx_model_name, ["box1", "box2"])

        onnx_model_name = "model_npu_rotated_overlaps.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_rotated_box_decode(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, anchor_boxes, deltas, weight):
                return torch_npu.npu_rotated_box_decode(anchor_boxes, deltas, weight)

        def export_onnx(onnx_model_name):
            anchor_boxes = torch.tensor([[[32.1855], [41.9922], [64.1435],
                                        [62.5325], [34.607]]]).to("npu")
            deltas = torch.tensor([[[1.8725], [-1.8915], [0.2395], [-0.4622],
                                    [-34.6539]]]).to("npu")
            weight = torch.tensor([1., 1., 1., 1., 1.]).npu()
            model = Model().to("npu")
            model(anchor_boxes, deltas, weight)
            self.onnx_export(model, (anchor_boxes, deltas, weight),
                             onnx_model_name, ["anchor_boxes", "deltas", "weight"])

        onnx_model_name = "model_npu_rotated_box_decode.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_rotated_box_encode(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, anchor_boxes, gt_bboxes, weight):
                return torch_npu.npu_rotated_box_encode(anchor_boxes, gt_bboxes, weight)

        def export_onnx(onnx_model_name):
            anchor_boxes = torch.tensor([[[44.2877], [9.1412], [88.7575],
                                        [25.8879], [64.8047]]]).to("npu")
            gt_bboxes = torch.tensor([[[39.1763], [0.9838], [78.1028],
                                       [29.5997], [51.5907]]]).to("npu")
            weight = torch.tensor([1., 1., 1., 1., 1.]).npu()
            model = Model().to("npu")
            model(anchor_boxes, gt_bboxes, weight)
            self.onnx_export(model, (anchor_boxes, gt_bboxes, weight),
                             onnx_model_name, ["anchor_boxes", "gt_bboxes", "weight"])

        onnx_model_name = "model_npu_rotated_box_encode.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_yolo_boxes_encode(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, anchor_boxes, gt_bboxes, stride):
                return torch_npu.npu_yolo_boxes_encode(anchor_boxes, gt_bboxes, stride)

        def export_onnx(onnx_model_name):
            anchor_boxes = torch.rand((2, 4)).npu()
            gt_bboxes = torch.rand((2, 4)).npu()
            stride = torch.rand(2).npu().to(torch.int32)
            model = Model().to("npu")
            model(anchor_boxes, gt_bboxes, stride)
            self.onnx_export(model, (anchor_boxes, gt_bboxes, stride),
                             onnx_model_name, ["anchor_boxes", "gt_bboxes", "stride"])

        onnx_model_name = "model_npu_yolo_boxes_encode.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_masked_fill_range(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input1, start, end, value):
                return torch_npu.npu_masked_fill_range(input1, start, end, value, 2)

        def export_onnx(onnx_model_name):
            input1 = torch.rand((32, 64, 1688)).uniform_(1, 100).npu().to(torch.int8)
            start = torch.tensor([list(range(0, 32))], dtype=torch.int32).npu()
            end = torch.tensor([list(range(6, 38))], dtype=torch.int32).npu()
            value = torch.tensor([1.0]).npu().to(torch.int8)
            model = Model().to("npu")
            model(input1, start, end, value)
            self.onnx_export(model, (input1, start, end, value),
                             onnx_model_name, ["input1", "start", "end", "value"])

        onnx_model_name = "model_npu_masked_fill_range.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_anchor_response_flags(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input1):
                return torch_npu.npu_anchor_response_flags(input1, [60, 60], [2, 2], 9)

        def export_onnx(onnx_model_name):
            input1 = torch.rand([100, 4]).npu()
            model = Model().to("npu")
            model(input1)
            self.onnx_export(model, input1, onnx_model_name, ["input1"])

        onnx_model_name = "model_npu_anchor_response_flags.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_indexing(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input1):
                return torch_npu.npu_indexing(input1, [0, 0], [2, 2], [1, 1])

        def export_onnx(onnx_model_name):
            input1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to("npu").float()
            model = Model().to("npu")
            model(input1)
            self.onnx_export(model, input1,
                             onnx_model_name, ["input1"])

        onnx_model_name = "model_npu_indexing.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_sign_bits_pack(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input1):
                return torch_npu.npu_sign_bits_pack(input1, 2)

        def export_onnx(onnx_model_name):
            input1 = torch.tensor([5, 4, 3, 2, 0, -1, -2, 4, 3, 2, 1, 0, -1, -2],
                                  dtype=torch.float32).npu()
            model = Model().to("npu")
            model(input1)
            self.onnx_export(model, input1,
                             onnx_model_name, ["input1"])

        onnx_model_name = "model_npu_sign_bits_pack.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_stride_add(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input1, input2):
                return torch_npu.npu_stride_add(input1, input2, 0, 0, 1)

        def export_onnx(onnx_model_name):
            input1 = torch.tensor([[[[[1.]]]]]).npu()
            input2 = torch.tensor([[[[[1.]]]]]).npu()
            model = Model().to("npu")
            model(input1, input2)
            self.onnx_export(model, (input1, input2),
                             onnx_model_name, ["input1", "input2"])

        onnx_model_name = "model_npu_stride_add.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_lstm_cell(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                input_size = 8
                hidden_size = 7
                self.weight_ih = torch_npu.npu_format_cast(
                    torch.rand((input_size, 4 * hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 2
                )
                self.weight_hh = torch_npu.npu_format_cast(
                    torch.rand((hidden_size, 4 * hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 2
                )
                self.bias_ih = torch_npu.npu_format_cast(
                    torch.rand((4 * hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 2
                )
                self.bias_hh = torch_npu.npu_format_cast(
                    torch.rand((4 * hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 2
                )

            def forward(self, input_data, h0_data, c0_data):
                return torch_npu.npu_lstm_cell(input_data, self.weight_ih, self.weight_hh,
                                               h0_data, c0_data, self.bias_ih, self.bias_hh)

        def export_onnx(onnx_model_name):
            input_size = 8
            hidden_size = 7
            batch_size = 3

            input_shape = (batch_size, input_size)
            h0_shape = (batch_size, hidden_size)
            c0_shape = (batch_size, hidden_size)

            input_data = torch_npu.npu_format_cast(
                torch.rand(input_shape).uniform_(-1, 1).npu().to(torch.float16), 29
            )
            h0_data = torch_npu.npu_format_cast(
                torch.rand(h0_shape).uniform_(-1, 1).npu().to(torch.float16), 29
            )
            c0_data = torch_npu.npu_format_cast(
                torch.rand(c0_shape).uniform_(-1, 1).npu().to(torch.float16), 29
            )
            model = Model().to("npu")
            model(input_data, h0_data, c0_data)
            self.onnx_export(model, (input_data, h0_data, c0_data), onnx_model_name,
                             ["input_", "h0_data", "c0_data"])

        onnx_model_name = "model_npu_lstm_cell.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_lstm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                input_size = 10
                hidden_size = 5
                seq_length = 5
                self.seq_length_t = torch.Tensor((seq_length)).int().npu()

                self.weight_ih = torch_npu.npu_format_cast(
                    torch.rand((input_size, 4 * hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 2
                )
                self.weight_hh = torch_npu.npu_format_cast(
                    torch.rand((hidden_size, 4 * hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 2
                )
                self.bias = torch_npu.npu_format_cast(
                    torch.rand((4 * hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 2
                )
                self.weight = torch.cat((self.weight_ih, self.weight_hh), dim=0)

            def forward(self, input_data, h0_data, c0_data):
                return torch_npu.npu_lstm(input_data, self.weight, self.bias, self.seq_length_t,
                                          h0_data, c0_data, True, 1, 0.0, False, False, False, False, False)

        def export_onnx(onnx_model_name):
            input_size = 10
            hidden_size = 5
            num_layers = 1
            batch_szie = 3
            seq_length = 5
            bidirectional = False
            d = 2 if bidirectional else 1
            input_shape = (seq_length, batch_szie, input_size)
            h0_shape = (d * num_layers, batch_szie, hidden_size)
            c0_shape = (d * num_layers, batch_szie, hidden_size)

            input_data = torch_npu.npu_format_cast(
                torch.rand(input_shape).uniform_(-1, 1).npu().to(torch.float16), 29
            )
            h0_data = torch_npu.npu_format_cast(
                torch.rand(h0_shape).uniform_(-1, 1).npu().to(torch.float16), 29
            )
            c0_data = torch_npu.npu_format_cast(
                torch.rand(c0_shape).uniform_(-1, 1).npu().to(torch.float16), 29
            )

            model = Model().to("npu")
            model(input_data, h0_data, c0_data)
            self.onnx_export(model, (input_data, h0_data, c0_data), onnx_model_name,
                             ["input_", "h0_data", "c0_data"])

        onnx_model_name = "model_npu_lstm.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    def test_wrapper_npu_gru(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_size = 10
                self.hidden_size = 6
                self.batch_size = 3
                self.num_layers = 1
                self.seq_length = 6
                self.has_biases = True
                self.seq_length_t = torch.Tensor([self.seq_length]).int().npu()

                self.weight_ih = torch_npu.npu_format_cast(
                    torch.rand((self.input_size, 3 * self.hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 4
                )
                self.weight_hh = torch_npu.npu_format_cast(
                    torch.rand((self.hidden_size, 3 * self.hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 4
                )
                self.bias_ih = torch_npu.npu_format_cast(
                    torch.rand((3 * self.hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 2
                )
                self.bias_hh = torch_npu.npu_format_cast(
                    torch.rand((3 * self.hidden_size)).uniform_(-1, 1).npu().to(torch.float16), 2
                )

            def forward(self, input, hx):
                return torch_npu.npu_gru(input, hx, self.weight_ih, self.weight_hh,
                                         self.bias_ih, self.bias_hh, self.seq_length_t, self.has_biases,
                                         self.num_layers, 0.0, False, False, False)

        def export_onnx(onnx_model_name):
            input_size = 10
            hidden_size = 6
            batch_size = 3
            num_layers = 1
            seq_length = 6
            input_shape = [seq_length, batch_size, input_size]
            h_0_shape = [num_layers, batch_size, hidden_size]
            input_ = torch_npu.npu_format_cast(torch.rand(input_shape).uniform_(-1, 1).npu().to(torch.float16), 29)
            hx = torch_npu.npu_format_cast(torch.rand(h_0_shape).uniform_(-1, 1).npu().to(torch.float16), 29)
            model = Model().to("npu")
            model(input_, hx)
            self.onnx_export(model, (input_, hx), onnx_model_name,
                             ["input_", "hx"])

        onnx_model_name = "model_npu_gru.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))
    
    def test_wrapper_npu_dropout_with_add_softmax(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_1, input_2):
                alpha = 0.1
                prob = 0
                dim = -1
                return torch_npu.npu_dropout_with_add_softmax(input_1, input_2,
                                                              alpha, prob, dim)

        def export_onnx(onnx_model_name):
            input_1 = torch.rand((4, 3, 64, 64)).npu()
            input_2 = torch.rand((4, 3, 64, 64)).npu()
            model = Model().to("npu")
            self.onnx_export(model, (input_1, input_2), onnx_model_name,
                             ["input_1", "input_2"])

        onnx_model_name = "model_npu_dropout_with_add_softmax.onnx"
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
    
    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_moe_compute_expert_tokens(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, sorted_experts):
                return torch_npu.npu_moe_compute_expert_tokens(sorted_experts=5)
            
        def export_onnx(onnx_model_name):
            data = list(range(20))
            experts = torch.tensor(data, dtype=torch.int32).npu()
            sorted_experts = torch.sort(experts)[0]
            model = Model().to("npu")
            model(sorted_experts)
            self.onnx_export(model, (sorted_experts), onnx_model_name)
        onnx_model_name = "model_moe_compute_expert_tokens.onnx"
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
            model(input_)
            self.onnx_export(model, input_, onnx_model_name, ["input_"])

        onnx_model_name = "model_npu_mish.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_deep_norm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, gx, beta, gamma):
                alpha = 0.3
                epsilon = 1e-6
                mean, rstd, y = torch_npu.npu_deep_norm(x, gx, beta, gamma, alpha, epsilon)
                return mean, rstd, y

        def export_onnx(onnx_model_name):
            x = torch.rand([10, 1024], dtype=torch.float32).npu()
            gx = torch.rand([10, 1024], dtype=torch.float32).npu()
            beta = torch.rand([1024], dtype=torch.float32).npu()
            gamma = torch.rand([1024], dtype=torch.float32).npu()
            model = Model().to("npu")
            model(x, gx, beta, gamma)
            self.onnx_export(model, (x, gx, beta, gamma), onnx_model_name)
        onnx_model_name = "model_npu_deep_norm.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_rms_norm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, gamma):
                epsilon = 1e-6
                x = torch_npu.npu_rms_norm(x, gamma, epsilon)
                return x
            
        def export_onnx(onnx_model_name):
            x = torch.rand(10, 1024).uniform_(-3, 3).npu().half()
            gamma = torch.rand(1024).uniform_(-3, 3).npu().half()
            model = Model().to("npu")
            model(x, gamma)
            self.onnx_export(model, (x, gamma), onnx_model_name)
        onnx_model_name = "model_npu_rms_norm.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_add_rms_norm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x1, x2, gamma):
                epsilon = 1e-6
                x = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon)
                return x
            
        def export_onnx(onnx_model_name):
            x1 = torch.rand(10, 1024).uniform_(-3, 3).npu().half()
            x2 = torch.rand(10, 1024).uniform_(-3, 3).npu().half()
            gamma = torch.rand(1024).uniform_(-3, 3).npu().half()
            model = Model().to("npu")
            model(x1, x2, gamma)
            self.onnx_export(model, (x1, x2, gamma), onnx_model_name)
        onnx_model_name = "model_npu_add_rms_norm.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_rotary_mul(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, r1, r2):
                return torch_npu.npu_rotary_mul(x, r1, r2)

        def export_onnx(onnx_model_name):
            x = torch.rand([8192, 2, 5, 128], dtype=torch.float32).npu()
            r1 = torch.rand([8192, 1, 1, 128], dtype=torch.float32).npu()
            r2 = torch.rand([8192, 1, 1, 128], dtype=torch.float32).npu()
            model = Model().to("npu")
            model(x, r1, r2)
            self.onnx_export(model, (x, r1, r2), onnx_model_name, ["x", "r1", "r2"])

        onnx_model_name = "model_npu_rotary_mul.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_masked_softmax_with_rel_pos_bias(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, atten_mask, relative_pos_bias):
                scale_value = 1.0
                inner_precision_mode = 0
                return torch_npu.npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode)

        def export_onnx(onnx_model_name):
            x = torch.rand([16, 2, 3, 4, 5], dtype=torch.float32).npu()
            atten_mask = torch.rand([2, 4, 5], dtype=torch.float32).npu()
            relative_pos_bias = torch.rand([3, 4, 5], dtype=torch.float32).npu()
            model = Model().to("npu")
            model(x, atten_mask, relative_pos_bias)
            self.onnx_export(model, (x, atten_mask, relative_pos_bias), onnx_model_name, ["x", "atten_mask", "relative_pos_bias"])

        onnx_model_name = "model_npu_masked_softmax_with_rel_pos_bias.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_dynamic_quant(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_dummy, smooth_scales_dummy):
                output, scale = torch_npu.npu_dynamic_quant(input_dummy, smooth_scales=smooth_scales_dummy)
                return output, scale
            
        def export_onnx(onnx_model_name):
            input_dummy = torch.rand(4, 1024, 512).uniform_(-3, 3).npu().to(torch.float16)
            smooth_scales_dummy = torch.rand(512).uniform_(-3, 3).npu().to(torch.float16)
            model = Model().to("npu")
            model(input_dummy, smooth_scales_dummy)
            self.onnx_export(model, (input_dummy, smooth_scales_dummy), onnx_model_name,
                             ["input", "smooth_scale_dummy"], ["output", "scale"])
        onnx_model_name = "model_npu_dynamic_quant.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_dynamic_quant_with_group_index(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_dummy, smooth_scales_dummy, group_index_dummy):
                output, scale = torch_npu.npu_dynamic_quant(input_dummy, smooth_scales=smooth_scales_dummy, group_index=group_index_dummy)
                return output, scale
            
        def export_onnx(onnx_model_name):
            input_dummy = torch.rand(4, 1024, 512).uniform_(-3, 3).npu().to(torch.float16)
            group_num = 10
            smooth_scales_dummy = torch.randn((group_num, input_dummy.shape[-1])).uniform_(-3, 3).npu().to(torch.float16)
            row_num = input_dummy.numel() // input_dummy.shape[-1]
            group_index_list = []
            for _ in range(group_num):
                group_index_list.append(np.random.randint(0, row_num))
            group_index_list = sorted(group_index_list)
            group_index_list[-1] = row_num
            group_index_dummy = torch.tensor(group_index_list).npu().to(torch.int32)
            model = Model().to("npu")
            model(input_dummy, smooth_scales_dummy, group_index_dummy)
            self.onnx_export(model, (input_dummy, smooth_scales_dummy, group_index_dummy), onnx_model_name,
                             ["input", "smooth_scale_dummy", "group_index_dummy"], ["output", "scale"])
        onnx_model_name = "model_npu_dynamic_quant.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_dynamic_quant_asymmetric(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, input_dummy, smooth_scales_dummy, group_index_dummy):
                output, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(input_dummy, smooth_scales=smooth_scales_dummy, group_index=group_index_dummy)
                return output, scale, offset
            
        def export_onnx(onnx_model_name):
            input_dummy = torch.rand(4, 1024, 512).uniform_(-3, 3).npu().to(torch.float16)
            group_num = 10
            smooth_scales_dummy = torch.randn((group_num, input_dummy.shape[-1])).uniform_(-3, 3).npu().to(torch.float16)
            row_num = input_dummy.numel() // input_dummy.shape[-1]
            group_index_list = []
            for _ in range(group_num):
                group_index_list.append(np.random.randint(0, row_num))
            group_index_list = sorted(group_index_list)
            group_index_list[-1] = row_num
            group_index_dummy = torch.tensor(group_index_list).npu().to(torch.int32)
            model = Model().to("npu")
            model(input_dummy, smooth_scales_dummy, group_index_dummy)
            self.onnx_export(model, (input_dummy, smooth_scales_dummy, group_index_dummy), onnx_model_name,
                             ["input", "smooth_scale_dummy", "group_index_dummy"], ["output", "scale", "offset"])
        onnx_model_name = "model_npu_dynamic_quant_asymmetric.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_weight_quant_batchmatmul(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size):
                return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, 0)

        def export_onnx(onnx_model_name):
            x = torch.randn((8192, 320), dtype=torch.bfloat16).npu()
            weight = torch.randn((320, 256), dtype=torch.int8, device="npu")
            antiquantscale = torch.randn((1, 256), dtype=torch.bfloat16).npu()
            antiquantoffset = torch.randn((1, 256), dtype=torch.bfloat16).npu()
            model = Model().to("npu")
            model(x, weight, antiquantscale, antiquantoffset, None, None, None, 0)
            self.onnx_export(model, (x, weight, antiquantscale, antiquantoffset, None, None, None, 0), onnx_model_name)

        onnx_model_name = "model_npu_weight_quant_batchmatmul.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_anti_quant(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, scale, offset=None, dst_dtype=torch.float16, src_dtype=torch.int8):
                return torch_npu.npu_anti_quant(x, scale, offset=offset, dst_dtype=dst_dtype, src_dtype=src_dtype)

        def export_onnx(onnx_model_name):
            x = torch.randint(low=-128, high=127, size=(10, 1), dtype=torch.int8).npu()
            scale = torch.randn((1,), dtype=torch.float).npu()
            offset = torch.randn((1,), dtype=torch.float).npu()
            model = Model().to("npu")
            model(x, scale, offset, None, None)
            self.onnx_export(model, (x, scale, offset, None, None), onnx_model_name)

        onnx_model_name = "mode_npu_anti_quant.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path, onnx_model_name)))

    def test_wrapper_npu_quantize(self):            
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, inputs, scales, zero_points):
                dtype = torch.quint8
                axis = 2
                return torch_npu.npu_quantize(inputs, scales, zero_points, dtype=dtype, axis=axis)

        def export_onnx(onnx_model_name):
            inputs = torch.randn(5, 16, 8).npu()
            scales = torch.tensor([0.1] * 8).npu()
            zero_points = torch.tensor([0] * 8, dtype=torch.int32).npu()
            model = Model().to("npu")
            model(inputs, scales, zero_points)
            self.onnx_export(model, (inputs, scales, zero_points), onnx_model_name)

        onnx_model_name = "model_npu_quantize.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_group_quant(self):            
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, scale, group_index, offset):
                dtype = torch.qint8
                return torch_npu.npu_group_quant(x, scale, group_index, offset=offset, dst_dtype=dtype)

        def export_onnx(onnx_model_name):
            S, H, E = 6, 4, 4
            x_shape, scale_shape, offset_shape = (S, H), (E, H), (1,)
            x = torch.randn(*x_shape).to(torch.float).npu()
            scale = torch.randn(*scale_shape).to(torch.float).npu()
            group_index = torch.tensor([1, 2, 4, 6]).to(torch.int32).npu()
            offset = torch.tensor(*offset_shape).to(torch.float).npu()
            model = Model().to("npu")
            model(x, scale, group_index, offset)
            self.onnx_export(model, (x, scale, group_index, offset), onnx_model_name)

        onnx_model_name = "model_npu_group_quant.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_moe_finalize_routing(self):            
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row,
                        expert_for_source_row):
                return torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales,
                                                          expanded_src_to_dst_row, expert_for_source_row)

        def export_onnx(onnx_model_name):
            expert_num = 16
            token_len = 10
            top_k = 4
            num_rows = 50
            expanded_permuted_rows = torch.randn(num_rows * top_k, token_len).to(torch.float32).npu()
            skip1 = torch.randn(num_rows, token_len).to(torch.float32).npu()
            skip2_optional = torch.randn(num_rows, token_len).to(torch.float32).npu()
            bias = torch.randn(expert_num, token_len).to(torch.float32).npu()
            scales = torch.randn(num_rows, top_k).to(torch.float32).npu()
            expanded_src_to_dst_row = torch.arange(num_rows * top_k).to(torch.int32).npu()
            expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k)).to(torch.int32).npu()
            model = Model().to("npu")
            model(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row)
            self.onnx_export(model, (expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row), onnx_model_name)

        onnx_model_name = "model_npu_moe_finalize_routing.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_moe_finalize_routing_v2(self):            
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, expanded_permuted_rows, skip1, skip2_optional, bias, scales,
                        expanded_src_to_dst_row, expert_for_source_row):
                return torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional,
                                                          bias, scales, expanded_src_to_dst_row, 
                                                          expert_for_source_row, drop_pad_mode=1)

        def export_onnx(onnx_model_name):
            expert_num = 16
            token_len = 10
            top_k = 4
            num_rows = 50
            c = 20
            expanded_permuted_rows = torch.randn(expert_num, c, token_len).to(torch.float32).npu()
            skip1 = torch.randn(num_rows, token_len).to(torch.float32).npu()
            skip2_optional = torch.randn(num_rows, token_len).to(torch.float32).npu()
            bias = torch.randn(expert_num, token_len).to(torch.float32).npu()
            scales = torch.randn(num_rows, top_k).to(torch.float32).npu()
            expanded_src_to_dst_row = torch.arange(num_rows * top_k).to(torch.int32).npu()
            expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k)).to(torch.int32).npu()
            model = Model().to("npu")
            model(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row)
            self.onnx_export(model, (expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row), onnx_model_name)

        onnx_model_name = "model_npu_moe_finalize_routing_v2.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

    @SupportedDevices(['Ascend910B'])
    def test_wrapper_npu_gelu(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                y = torch_npu.npu_gelu(x, approximate="none")
                return y

        def export_onnx(onnx_model_name):
            x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], 
                             dtype=torch.float16).npu()
            model = Model().to("npu")
            model(x)
            self.onnx_export(model, (x), onnx_model_name,
                             input_names=["x"],
                             output_names=["y"])

        onnx_model_name = "model_npu_gelu_none.onnx"
        export_onnx(onnx_model_name)
        assert (os.path.isfile(os.path.join(TestOnnxOps.test_onnx_path,
                                            onnx_model_name)))

if __name__ == '__main__':
    run_tests()
