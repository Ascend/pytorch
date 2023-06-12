import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

# acl format
FORMAT_ND = 2
FORMAT_Z = 4
FORMAT_NZ = 29


class TestJitTrace(TestCase):

    def test_trace_npu_max(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                x = torch_npu.npu_max(x, dim=1)
                return x

        example_input = torch.rand(2, 8).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input)
        output2 = trace_model(example_input)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_min(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                x = torch_npu.npu_min(x, dim=1)
                return x

        example_input = torch.rand(2, 8).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input)
        output2 = trace_model(example_input)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_one_hot(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                x = torch_npu.npu_one_hot(x, depth=5)
                return x

        example_input = torch.IntTensor([5, 3, 2, 1]).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input)
        output2 = trace_model(example_input)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_slice(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                x = torch_npu.npu_slice(x, [0, 0], [2, 2])
                return x

        example_input = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float16).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input)
        output2 = trace_model(example_input)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_roi_align(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x, rois):
                x = torch_npu.npu_roi_align(x, rois, 0.25, 3, 3, 2, 0)
                return x

        example_input = torch.FloatTensor([[[[1, 2, 3, 4, 5, 6],
                                             [7, 8, 9, 10, 11, 12],
                                             [13, 14, 15, 16, 17, 18],
                                             [19, 20, 21, 22, 23, 24],
                                             [25, 26, 27, 28, 29, 30],
                                             [31, 32, 33, 34, 35, 36]]]]).npu()
        rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input, rois)

        trace_model = torch.jit.trace(model, (example_input, rois))
        output2 = trace_model(example_input, rois)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_iou(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, bboxes, gtboxes):
                x = torch_npu.npu_iou(bboxes, gtboxes, 0)
                return x

        bboxes = torch.tensor([[0, 0, 10, 10],
                               [10, 10, 20, 20],
                               [32, 32, 38, 42]], dtype=torch.float16).npu()
        gtboxes = torch.tensor([[0, 0, 10, 20],
                                [0, 10, 10, 10],
                                [10, 10, 20, 20]], dtype=torch.float16).npu()
        model = NpuModel().to("npu")
        output1 = model(bboxes, gtboxes)

        trace_model = torch.jit.trace(model, (bboxes, gtboxes))
        output2 = trace_model(bboxes, gtboxes)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_batch_nms(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, boxes, scores):
                x = torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 4, 4)
                return x

        boxes = torch.randn(8, 4, 1, 4, dtype=torch.float32).npu()
        scores = torch.randn(8, 4, 1, dtype=torch.float32).npu()
        model = NpuModel().to("npu")
        output1 = model(boxes, scores)

        trace_model = torch.jit.trace(model, (boxes, scores))
        output2 = trace_model(boxes, scores)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_fused_attention_score(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, query_layer, key_layer, value_layer, attention_mask):
                scale = 0.125
                keep_prob = 1
                return torch_npu.npu_fused_attention_score(query_layer, key_layer,
                                                           value_layer, attention_mask, scale, keep_prob)

        q = torch.rand(24, 16, 512, 64).uniform_(-3, 3).npu().half()
        k = torch.rand(24, 16, 512, 64).uniform_(-3, 3).npu().half()
        v = torch.rand(24, 16, 512, 64).uniform_(-3, 3).npu().half()
        mask = torch.ones(512) * -10000.
        mask[:6] = -0.
        mask = mask.expand(24, 1, 512, 512).npu().half()
        model = NpuModel().to("npu")
        output1 = model(q, k, v, mask)

        trace_model = torch.jit.trace(model, (q, k, v, mask))
        output2 = trace_model(q, k, v, mask)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_multi_head_attention(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()
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
                    torch.randn((weight_col, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ)
                self.key_weight = torch_npu.npu_format_cast(
                    torch.randn((weight_col, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ)
                self.value_weight = torch_npu.npu_format_cast(
                    torch.randn((weight_col, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ)
                self.out_proj_weight = torch_npu.npu_format_cast(
                    torch.randn((weight_col, weight_col)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_NZ)
                self.attn_mask = torch_npu.npu_format_cast(
                    torch.randn((batch, attn_head_num, tgt_len, src_len)).uniform_(-1, 1).to(torch.half).npu(),
                    FORMAT_ND)
                self.query_bias = torch_npu.npu_format_cast(
                    torch.randn((weight_col,)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_ND)
                self.key_bias = torch_npu.npu_format_cast(
                    torch.randn((weight_col,)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_ND)
                self.value_bias = torch_npu.npu_format_cast(
                    torch.randn((weight_col,)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_ND)
                self.out_proj_bias = torch_npu.npu_format_cast(
                    torch.randn((weight_col,)).uniform_(-1, 1).to(torch.half).npu(), FORMAT_ND)
                self.grad = torch_npu.npu_format_cast(
                    torch.randn((batch * tgt_len, attn_dim_per_head * attn_head_num)).uniform_(-1, 1).to(
                        torch.half).npu(), FORMAT_NZ)
                self.mask = (torch.randn((src_len * tgt_len * attn_head_num)).uniform_(-1, 1).npu() > 0).to(torch.uint8)

            def forward(self, query, key, value):
                return torch_npu.npu_multi_head_attention(
                    query, key, value, self.query_weight, self.key_weight, self.value_weight,
                    self.attn_mask, self.out_proj_weight, self.query_bias, self.key_bias,
                    self.value_bias, self.out_proj_bias, self.mask, self.attn_head_num,
                    self.attn_dim_per_head, self.src_len, self.tgt_len, self.dropout_prob,
                    self.softmax_use_float)

        batch = 8
        attn_head_num = 16
        attn_dim_per_head = 64
        src_len = 64
        tgt_len = 64
        weight_col = attn_head_num * attn_dim_per_head

        query = torch_npu.npu_format_cast(torch.randn((batch * tgt_len, weight_col)).uniform_(-1, 1).to(
            torch.half).npu(), FORMAT_NZ)
        key = torch_npu.npu_format_cast(torch.randn((batch * src_len, weight_col)).uniform_(-1, 1).to(
            torch.half).npu(), FORMAT_NZ)
        value = torch_npu.npu_format_cast(torch.randn((batch * src_len, weight_col)).uniform_(-1, 1).to(
            torch.half).npu(), FORMAT_NZ)

        model = NpuModel().to("npu")
        output1 = model(query, key, value)

        trace_model = torch.jit.trace(model, (query, key, value))
        output2 = trace_model(query, key, value)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_diou(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_diou(box1, box2)

        box1 = torch.tensor([[0, 0, 10, 10],
                             [10, 10, 20, 20],
                             [32, 32, 38, 42],
                             [32, 32, 38, 42]], dtype=torch.float32).to("npu")
        box2 = torch.tensor([[0, 0, 10, 20],
                             [0, 10, 10, 10],
                             [10, 10, 20, 20],
                             [10, 10, 20, 20]], dtype=torch.float32).to("npu")
        model = NpuModel().to("npu")
        output1 = model(box1, box2)

        trace_model = torch.jit.trace(model, (box1, box2))
        output2 = trace_model(box1, box2)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_ciou(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_ciou(box1, box2, False, False)

        box1 = torch.rand(4, 8).npu()
        box2 = torch.rand(4, 8).npu()
        model = NpuModel().to("npu")
        output1 = model(box1, box2)

        trace_model = torch.jit.trace(model, (box1, box2))
        output2 = trace_model(box1, box2)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_giou(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_giou(box1, box2, True, False, 0)

        box1 = torch.tensor([[0.4375, 0.0041, 0.4893, 0.4176],
                             [0.1618, 0.1920, 0.4528, 0.4363],
                             [0.7243, 0.6361, 0.8139, 0.7649],
                             [0.9430, 0.6788, 0.6872, 0.8605]]).npu()
        box2 = torch.tensor([[0.1625, 0.4915, 0.4430, 0.1314],
                             [0.2110, 0.0042, 0.2204, 0.0087],
                             [0.7917, 0.5444, 0.5964, 0.9363],
                             [0.7733, 0.7770, 0.7666, 0.8029]]).npu()
        model = NpuModel().to("npu")
        output1 = model(box1, box2)

        trace_model = torch.jit.trace(model, (box1, box2))
        output2 = trace_model(box1, box2)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_deformable_conv2d(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()
                self.weight = torch.rand((32, 32, 5, 5)).npu()
                self.offset = torch.rand((16, 75, 32, 32)).npu()

            def forward(self, input_):
                return torch_npu.npu_deformable_conv2d(input_, self.weight, self.offset,
                                                       None, kernel_size=[5, 5], stride=[1, 1, 1, 1],
                                                       padding=[2, 2, 2, 2])

        example_input = torch.rand(16, 32, 32, 32).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input)
        output2 = trace_model(example_input)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_format_cast(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input_):
                return torch_npu.npu_format_cast(input_, 2)

        example_input = torch.rand(3, 3).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input)
        output2 = trace_model(example_input)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_softmax_cross_entropy_with_logits(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input_, label):
                return torch_npu.npu_softmax_cross_entropy_with_logits(input_, label)

        input_ = torch.tensor([[1., 2., 3., 4.]]).npu()
        label = torch.tensor([[1., 2., 3., 4.]]).npu()
        model = NpuModel().to("npu")
        output1 = model(input_, label)

        trace_model = torch.jit.trace(model, (input_, label))
        output2 = trace_model(input_, label)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_ps_roi_pooling(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input_, roi):
                return torch_npu.npu_ps_roi_pooling(input_, roi, 0.5, 2, 2)

        input_ = torch.tensor([[[[1]], [[2]], [[3]], [[4]],
                                [[5]], [[6]], [[7]], [[8]]],
                               [[[9]], [[10]], [[11]], [[12]],
                                [[13]], [[14]], [[15]], [[16]]]
                               ], dtype=torch.float16).npu()
        roi = torch.tensor([[[1], [2], [3], [4], [5]],
                            [[6], [7], [8], [9], [10]]
                            ], dtype=torch.float16).npu()
        model = NpuModel().to("npu")
        output1 = model(input_, roi)

        trace_model = torch.jit.trace(model, (input_, roi))
        output2 = trace_model(input_, roi)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_grid_assign_positive(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, assigned_gt_inds, overlaps, box_responsible_flags, max_overlap,
                        argmax_overlap, gt_max_overlaps, gt_argmax_overlaps):
                return torch_npu.npu_grid_assign_positive(assigned_gt_inds, overlaps,
                                                          box_responsible_flags, max_overlap, argmax_overlap,
                                                          gt_max_overlaps,
                                                          gt_argmax_overlaps, 128, 0.5, 0.0, True)

        assigned_gt_inds = torch.rand((4,), dtype=torch.float32).to("npu")
        overlaps = torch.rand((2, 4), dtype=torch.float32).to("npu")
        box_responsible_flags = torch.tensor([1, 1, 1, 0], dtype=torch.uint8).to("npu")
        max_overlap = torch.rand((4,), dtype=torch.float32).to("npu")
        argmax_overlap = torch.tensor([1, 0, 1, 0], dtype=torch.int32).to("npu")
        gt_max_overlaps = torch.rand((2,), dtype=torch.float32).to("npu")
        gt_argmax_overlaps = torch.tensor([1, 0], dtype=torch.int32).to("npu")
        model = NpuModel().to("npu")
        output1 = model(assigned_gt_inds, overlaps, box_responsible_flags, max_overlap,
                        argmax_overlap, gt_max_overlaps, gt_argmax_overlaps)

        trace_model = torch.jit.trace(model, (assigned_gt_inds, overlaps, box_responsible_flags, max_overlap,
                                              argmax_overlap, gt_max_overlaps, gt_argmax_overlaps))
        output2 = trace_model(assigned_gt_inds, overlaps, box_responsible_flags, max_overlap,
                              argmax_overlap, gt_max_overlaps, gt_argmax_overlaps)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_ifmr(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input_, min_value, max_value, cdf):
                return torch_npu.npu_ifmr(input_, min_value, max_value, cdf,
                                          0.999999, 0.999999, 0.7, 1.3, 0.01, True)

        input_ = torch.rand(3, 3).npu()
        min_value = torch.reshape(torch.min(input_), (1,))
        max_value = torch.reshape(torch.max(input_), (1,))
        hist = torch.histc(input_.to('cpu'),
                           bins=128,
                           min=min_value[0].to('cpu'),
                           max=max_value[0].to('cpu'))
        cdf = torch.cumsum(hist, dim=0).int()
        cdf = cdf.to('npu')
        model = NpuModel().to("npu")
        output1 = model(input_, min_value, max_value, cdf)

        trace_model = torch.jit.trace(model, (input_, min_value, max_value, cdf))
        output2 = trace_model(input_, min_value, max_value, cdf)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_fused_attention_layernorm_qkv_fwd(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()
                q_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half()
                k_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half()
                v_weight = torch.rand(1024, 1024).uniform_(-0.1, 0.1).half()
                self.fused_q_w = torch_npu.npu_format_cast(q_weight.npu().t().contiguous(), FORMAT_NZ)
                self.fused_k_w = torch_npu.npu_format_cast(k_weight.npu().t().contiguous(), FORMAT_NZ)
                self.fused_v_w = torch_npu.npu_format_cast(v_weight.npu().t().contiguous(), FORMAT_NZ)
                self.q_bias = torch.rand(1024).half().npu()
                self.k_bias = torch.rand(1024).half().npu()
                self.v_bias = torch.rand(1024).half().npu()

            def forward(self, input_, gamma, beta):
                return torch_npu.npu_fused_attention_layernorm_qkv_fwd(input_,
                                                                       self.fused_q_w, self.fused_k_w, self.fused_v_w,
                                                                       gamma, beta,
                                                                       self.q_bias, self.k_bias, self.v_bias, 512, 16)

        ln_input = torch.rand(12288, 1024).uniform_(-6, 6).half()
        input_ = torch_npu.npu_format_cast(ln_input.npu(), FORMAT_NZ)
        gamma = torch.rand(1024).half().npu()
        beta = torch.rand(1024).half().npu()

        model = NpuModel().to("npu")
        output1 = model(input_, gamma, beta)

        trace_model = torch.jit.trace(model, (input_, gamma, beta))
        output2 = trace_model(input_, gamma, beta)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_fused_attention_score_fwd(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, q, k, v, mask):
                return torch_npu.npu_fused_attention_score_fwd(q, k, v, mask, 0.125, 1)

        q = torch.rand(24, 16, 512, 64).uniform_(-3, 3).half().npu()
        k = torch.rand(24, 16, 512, 64).uniform_(-3, 3).half().npu()
        v = torch.rand(24, 16, 512, 64).uniform_(-3, 3).half().npu()
        mask = torch.ones(512) * -10000.
        mask[:6] = -0.
        mask = mask.expand(24, 1, 512, 512).half().npu()

        model = NpuModel().to("npu")
        output1 = model(q, k, v, mask)

        trace_model = torch.jit.trace(model, (q, k, v, mask))
        output2 = trace_model(q, k, v, mask)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_sign_bits_unpack(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input_):
                size = 79
                return torch_npu.npu_sign_bits_unpack(input_, size, torch.float32)

        example_input = torch.randn(4424).uniform_(0, 255).to(torch.uint8).npu()
        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input)
        output2 = trace_model(example_input)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_ptiou(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, bboxs, gtboxes):
                return torch_npu.npu_ptiou(bboxs, gtboxes)

        bboxs = torch.tensor([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]], dtype=torch.float16).npu()
        gtboxes = torch.tensor([[1, 2, 3, 4],
                                [5, 6, 7, 8]], dtype=torch.float16).npu()
        model = NpuModel().to("npu")
        output1 = model(bboxs, gtboxes)

        trace_model = torch.jit.trace(model, (bboxs, gtboxes))
        output2 = trace_model(bboxs, gtboxes)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_normalize_batch(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input_, seq_len):
                return torch_npu.npu_normalize_batch(input_, seq_len)

        input_ = torch.rand([32, 3, 6]).npu()
        seq_len = torch.rand(32).uniform_(3, 6).npu().to(torch.int32)
        model = NpuModel().to("npu")
        output1 = model(input_, seq_len)

        trace_model = torch.jit.trace(model, (input_, seq_len))
        output2 = trace_model(input_, seq_len)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_nms_v4(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, boxes, scores, iou_threshold, scores_threshold):
                max_output_size = 20
                return torch_npu.npu_nms_v4(boxes, scores, max_output_size,
                                            iou_threshold, scores_threshold)

        boxes = torch.rand((100, 4)).uniform_(0, 100).npu()
        scores = torch.rand(100).uniform_(0, 1).npu()
        iou_threshold = torch.tensor(0.5).npu()
        scores_threshold = torch.tensor(0.3).npu()
        model = NpuModel().to("npu")
        output1 = model(boxes, scores, iou_threshold, scores_threshold)

        trace_model = torch.jit.trace(model, (boxes, scores, iou_threshold, scores_threshold))
        output2 = trace_model(boxes, scores, iou_threshold, scores_threshold)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_bounding_box_decode(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input1, input2):
                return torch_npu.npu_bounding_box_decode(input1, input2,
                                                         0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)

        input1 = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]]).to("npu").half()
        input2 = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]]).to("npu").half()
        model = NpuModel().to("npu")
        output1 = model(input1, input2)

        trace_model = torch.jit.trace(model, (input1, input2))
        output2 = trace_model(input1, input2)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_bounding_box_encode(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input1, input2):
                return torch_npu.npu_bounding_box_encode(input1, input2,
                                                         0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)

        input1 = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]]).to("npu").half()
        input2 = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]]).to("npu").half()
        model = NpuModel().to("npu")
        output1 = model(input1, input2)

        trace_model = torch.jit.trace(model, (input1, input2))
        output2 = trace_model(input1, input2)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_nms_with_mask(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input1):
                return torch_npu.npu_nms_with_mask(input1, 0.5)

        input1 = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.6],
                               [6.0, 7.0, 8.0, 9.0, 0.4]]).npu()
        model = NpuModel().to("npu")
        output1 = model(input1)

        trace_model = torch.jit.trace(model, input1)
        output2 = trace_model(input1)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_rotated_iou(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_rotated_iou(box1, box2, False, 0, True, 0.0, 0.0)

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

        model = NpuModel().to("npu")
        output1 = model(box1, box2)

        trace_model = torch.jit.trace(model, (box1, box2))
        output2 = trace_model(box1, box2)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_rotated_overlaps(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, box1, box2):
                return torch_npu.npu_rotated_overlaps(box1, box2, False)

        box1 = torch.tensor([[[35.7500, 48.6562, 12.0000, 13.0000, 66.0000],
                              [43.1250, 53.5625, 17.0000, 6.0000, -130.0000],
                              [53.4062, 38.1875, 17.0000, 10.0000, 60.0000]
                              ]]).npu()

        box2 = torch.tensor([[[43.2812, 30.6719, 13.0000, 2.0000, -73.0000],
                              [38.7188, 37.4062, 12.0000, 12.0000, -99.0000],
                              [52.1562, 56.6875, 18.0000, 15.0000, 163.0000],
                              [59.6250, 33.5312, 8.0000, 11.0000, 89.0000]
                              ]]).npu()
        model = NpuModel().to("npu")
        output1 = model(box1, box2)

        trace_model = torch.jit.trace(model, (box1, box2))
        output2 = trace_model(box1, box2)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_rotated_box_decode(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, anchor_boxes, deltas, weight):
                return torch_npu.npu_rotated_box_decode(anchor_boxes, deltas, weight)

        anchor_boxes = torch.tensor([[[32.1855], [41.9922], [64.1435],
                                      [62.5325], [34.607]]]).to("npu")
        deltas = torch.tensor([[[1.8725], [-1.8915], [0.2395], [-0.4622],
                                [-34.6539]]]).to("npu")
        weight = torch.tensor([1., 1., 1., 1., 1.]).npu()
        model = NpuModel().to("npu")
        output1 = model(anchor_boxes, deltas, weight)

        trace_model = torch.jit.trace(model, (anchor_boxes, deltas, weight))
        output2 = trace_model(anchor_boxes, deltas, weight)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_rotated_box_encode(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, anchor_boxes, gt_bboxes, weight):
                return torch_npu.npu_rotated_box_encode(anchor_boxes, gt_bboxes, weight)

        anchor_boxes = torch.tensor([[[44.2877], [9.1412], [88.7575],
                                      [25.8879], [64.8047]]]).to("npu")
        gt_bboxes = torch.tensor([[[39.1763], [0.9838], [78.1028],
                                   [29.5997], [51.5907]]]).to("npu")
        weight = torch.tensor([1., 1., 1., 1., 1.]).npu()
        model = NpuModel().to("npu")
        output1 = model(anchor_boxes, gt_bboxes, weight)

        trace_model = torch.jit.trace(model, (anchor_boxes, gt_bboxes, weight))
        output2 = trace_model(anchor_boxes, gt_bboxes, weight)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_yolo_boxes_encode(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, anchor_boxes, gt_bboxes, stride):
                return torch_npu.npu_yolo_boxes_encode(anchor_boxes, gt_bboxes, stride)

        anchor_boxes = torch.rand((2, 4)).npu()
        gt_bboxes = torch.rand((2, 4)).npu()
        stride = torch.rand(2).npu().to(torch.int32)
        model = NpuModel().to("npu")
        output1 = model(anchor_boxes, gt_bboxes, stride)

        trace_model = torch.jit.trace(model, (anchor_boxes, gt_bboxes, stride))
        output2 = trace_model(anchor_boxes, gt_bboxes, stride)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_masked_fill_range(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input1, start, end, value):
                return torch_npu.npu_masked_fill_range(input1, start, end, value, 2)

        input1 = torch.rand((32, 64, 1688)).uniform_(1, 100).npu().to(torch.int8)
        start = torch.tensor([list(range(0, 32))], dtype=torch.int32).npu()
        end = torch.tensor([list(range(6, 38))], dtype=torch.int32).npu()
        value = torch.tensor([1.0]).npu().to(torch.int8)
        model = NpuModel().to("npu")
        output1 = model(input1, start, end, value)

        trace_model = torch.jit.trace(model, (input1, start, end, value))
        output2 = trace_model(input1, start, end, value)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_anchor_response_flags(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input1):
                return torch_npu.npu_anchor_response_flags(input1, [60, 60], [2, 2], 9)

        input1 = torch.rand([100, 4]).npu()
        model = NpuModel().to("npu")
        output1 = model(input1)

        trace_model = torch.jit.trace(model, input1)
        output2 = trace_model(input1)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_indexing(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input1):
                return torch_npu.npu_indexing(input1, [0, 0], [2, 2], [1, 1])

        input1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to("npu").float()
        model = NpuModel().to("npu")
        output1 = model(input1)

        trace_model = torch.jit.trace(model, input1)
        output2 = trace_model(input1)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_sign_bits_pack(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input1):
                return torch_npu.npu_sign_bits_pack(input1, 2)

        input1 = torch.tensor([5, 4, 3, 2, 0, -1, -2, 4, 3, 2, 1, 0, -1, -2],
                              dtype=torch.float32).npu()
        model = NpuModel().to("npu")
        output1 = model(input1)

        trace_model = torch.jit.trace(model, input1)
        output2 = trace_model(input1)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_stride_add(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input1, input2):
                return torch_npu.npu_stride_add(input1, input2, 0, 0, 1)

        input1 = torch.tensor([[[[[1.]]]]]).npu()
        input2 = torch.tensor([[[[[1.]]]]]).npu()
        model = NpuModel().to("npu")
        output1 = model(input1, input2)

        trace_model = torch.jit.trace(model, (input1, input2))
        output2 = trace_model(input1, input2)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_scatter(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, input_, indices, updates):
                return torch_npu.npu_scatter(input_, indices, updates, 0)

        input_ = torch.tensor([[1.6279, 0.1226], [0.9041, 1.0980]]).npu()
        indices = torch.tensor([0, 1], dtype=torch.int32).npu()
        updates = torch.tensor([-1.1993, -1.5247]).npu()
        model = NpuModel().to("npu")
        output1 = model(input_, indices, updates)

        trace_model = torch.jit.trace(model, (input_, indices, updates))
        output2 = trace_model(input_, indices, updates)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_lstm_cell(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                input_size = 8
                hidden_size = 7
                self.weight_ih = torch_npu.npu_format_cast(torch.rand((input_size, 4 * hidden_size)).uniform_(
                    -1, 1).npu().to(torch.float16), FORMAT_ND)
                self.weight_hh = torch_npu.npu_format_cast(torch.rand((hidden_size, 4 * hidden_size)).uniform_(
                    -1, 1).npu().to(torch.float16), FORMAT_ND)
                self.bias_ih = torch_npu.npu_format_cast(torch.rand((4 * hidden_size)).uniform_(
                    -1, 1).npu().to(torch.float16), FORMAT_ND)
                self.bias_hh = torch_npu.npu_format_cast(torch.rand((4 * hidden_size)).uniform_(
                    -1, 1).npu().to(torch.float16), FORMAT_ND)

            def forward(self, input_data, h0_data, c0_data):
                return torch_npu.npu_lstm_cell(input_data, self.weight_ih, self.weight_hh,
                                               h0_data, c0_data, self.bias_ih, self.bias_hh)

        input_size = 8
        hidden_size = 7
        batch_size = 3

        input_shape = (batch_size, input_size)
        h0_shape = (batch_size, hidden_size)
        c0_shape = (batch_size, hidden_size)

        input_data = torch_npu.npu_format_cast(torch.rand(input_shape).uniform_(
            -1, 1).npu().to(torch.float16), FORMAT_NZ)
        h0_data = torch_npu.npu_format_cast(torch.rand(h0_shape).uniform_(
            -1, 1).npu().to(torch.float16), FORMAT_NZ)
        c0_data = torch_npu.npu_format_cast(torch.rand(c0_shape).uniform_(
            -1, 1).npu().to(torch.float16), FORMAT_NZ)
        model = NpuModel().to("npu")
        output1 = model(input_data, h0_data, c0_data)

        trace_model = torch.jit.trace(model, (input_data, h0_data, c0_data))
        output2 = trace_model(input_data, h0_data, c0_data)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_lstm(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                input_size = 10
                hidden_size = 5
                seq_length = 5
                self.seq_length_t = torch.Tensor((seq_length)).int().npu()

                self.weight_ih = torch_npu.npu_format_cast(torch.rand((input_size, 4 * hidden_size)).uniform_(
                    -1, 1).npu().to(torch.float16), FORMAT_ND)
                self.weight_hh = torch_npu.npu_format_cast(torch.rand((hidden_size, 4 * hidden_size)).uniform_(
                    -1, 1).npu().to(torch.float16), FORMAT_ND)
                self.bias = torch_npu.npu_format_cast(torch.rand((4 * hidden_size)).uniform_(
                    -1, 1).npu().to(torch.float16), FORMAT_ND)
                self.weight = torch.cat((self.weight_ih, self.weight_hh), dim=0)

            def forward(self, input_data, h0_data, c0_data):
                return torch_npu.npu_lstm(input_data, self.weight, self.bias, self.seq_length_t,
                                          h0_data, c0_data, True, 1, 0.0, False, False, False, False, False)

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

        input_data = torch_npu.npu_format_cast(torch.rand(input_shape).uniform_(
            -1, 1).npu().to(torch.float16), FORMAT_NZ)
        h0_data = torch_npu.npu_format_cast(torch.rand(h0_shape).uniform_(
            -1, 1).npu().to(torch.float16), FORMAT_NZ)
        c0_data = torch_npu.npu_format_cast(torch.rand(c0_shape).uniform_(
            -1, 1).npu().to(torch.float16), FORMAT_NZ)

        model = NpuModel().to("npu")
        output1 = model(input_data, h0_data, c0_data)

        trace_model = torch.jit.trace(model, (input_data, h0_data, c0_data))
        output2 = trace_model(input_data, h0_data, c0_data)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_gru(self):
        class NpuModel(torch.nn.Module):
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
                    torch.rand((self.input_size, 3 * self.hidden_size)).uniform_(-1, 1).npu().to(torch.float16),
                    FORMAT_Z)
                self.weight_hh = torch_npu.npu_format_cast(
                    torch.rand((self.hidden_size, 3 * self.hidden_size)).uniform_(-1, 1).npu().to(torch.float16),
                    FORMAT_Z)
                self.bias_ih = torch_npu.npu_format_cast(torch.rand((3 * self.hidden_size)).uniform_(-1, 1).npu().to(
                    torch.float16), FORMAT_ND)
                self.bias_hh = torch_npu.npu_format_cast(torch.rand((3 * self.hidden_size)).uniform_(-1, 1).npu().to(
                    torch.float16), FORMAT_ND)

            def forward(self, input_, hx):
                return torch_npu.npu_gru(input_, hx, self.weight_ih, self.weight_hh,
                                         self.bias_ih, self.bias_hh, self.seq_length_t, self.has_biases,
                                         self.num_layers, 0.0, False, False, False)

        input_size = 10
        hidden_size = 6
        batch_size = 3
        num_layers = 1
        seq_length = 6
        input_shape = [seq_length, batch_size, input_size]
        h_0_shape = [num_layers, batch_size, hidden_size]
        input_ = torch_npu.npu_format_cast(torch.rand(input_shape).uniform_(-1, 1).npu().to(
            torch.float16), FORMAT_NZ)
        hx = torch_npu.npu_format_cast(torch.rand(h_0_shape).uniform_(-1, 1).npu().to(
            torch.float16), FORMAT_NZ)
        model = NpuModel().to("npu")
        output1 = model(input_, hx)

        trace_model = torch.jit.trace(model, (input_, hx))
        output2 = trace_model(input_, hx)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_dropout_with_add_softmax(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_1, input_2):
                alpha = 0.1
                prob = 0
                dim = -1
                return torch_npu.npu_dropout_with_add_softmax(input_1, input_2,
                                                              alpha, prob, dim)

        input_1 = torch.rand((4, 3, 64, 64)).npu()
        input_2 = torch.rand((4, 3, 64, 64)).npu()
        model = NpuModel().to("npu")
        output1 = model(input_1, input_2)

        trace_model = torch.jit.trace(model, (input_1, input_2))
        output2 = trace_model(input_1, input_2)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_scaled_masked_softmax(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_, mask):
                scale = 0.56
                fixed_triu_mask = False
                return torch_npu.npu_scaled_masked_softmax(input_, mask,
                                                           scale, fixed_triu_mask)

        input_ = torch.rand((4, 3, 64, 64)).npu()
        mask = torch.rand((4, 3, 64, 64)).npu() > 0
        model = NpuModel().to("npu")
        output1 = model(input_, mask)

        trace_model = torch.jit.trace(model, (input_, mask))
        output2 = trace_model(input_, mask)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_mish(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_):
                return torch_npu.npu_mish(input_)

        input_ = torch.randn(5, 5).npu()
        model = NpuModel().to("npu")
        output1 = model(input_)

        trace_model = torch.jit.trace(model, input_)
        output2 = trace_model(input_)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_silu_(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                torch_npu.npu_silu_(x)
                return x

        example_input = torch.randn(5, 5).npu()
        example_input2 = example_input.clone()
        example_input3 = example_input.clone()

        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input2)
        output2 = trace_model(example_input3)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_slice_out(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x, out):
                torch_npu.npu_slice(x, [0, 0], [2, 2], out=out)
                return out

        example_input = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float16).npu()
        out_ = torch_npu.npu_slice(example_input, [0, 0], [2, 2])
        out1 = torch.zeros(out_.size(), dtype=out_.dtype).npu()
        out2 = out1.clone()
        out3 = out1.clone()

        model = NpuModel().to("npu")
        output1 = model(example_input, out1)

        trace_model = torch.jit.trace(model, (example_input, out2))
        output2 = trace_model(example_input, out3)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_bert_apply_adam_out(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, grad, var_in, m_in, v_in):
                max_grad_norm = -1.
                beta1 = 0.9
                beta2 = 0.99
                weight_decay = 0.
                lr = 0.
                epsilon = 1e-06
                global_grad_norm = 0.

                var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(
                    lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay,
                    out=(var_in, m_in, v_in))
                return var_out, m_out, v_out

        seed = 3
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

        var_in = torch.rand(321538).uniform_(-32., 21.).npu()
        m_in = torch.zeros(321538).npu()
        v_in = torch.zeros(321538).npu()
        grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
        model = NpuModel().to("npu")
        output1 = model(grad, var_in, m_in, v_in)

        seed = 3
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

        var_in = torch.rand(321538).uniform_(-32., 21.).npu()
        m_in = torch.zeros(321538).npu()
        v_in = torch.zeros(321538).npu()
        grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
        trace_model = torch.jit.trace(model, (grad, var_in, m_in, v_in))

        seed = 3
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

        var_in = torch.rand(321538).uniform_(-32., 21.).npu()
        m_in = torch.zeros(321538).npu()
        v_in = torch.zeros(321538).npu()
        grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
        output2 = trace_model(grad, var_in, m_in, v_in)
        self.assertRtolEqual(output1, output2)

    def test_trace_npu_silu_and_npu_max(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                torch_npu.npu_silu_(x)
                out = torch_npu.npu_max(x, dim=1)
                return out

        example_input = torch.randn(5, 5).npu()
        example_input2 = example_input.clone()
        example_input3 = example_input.clone()

        model = NpuModel().to("npu")
        output1 = model(example_input)

        trace_model = torch.jit.trace(model, example_input2)
        output2 = trace_model(example_input3)
        self.assertRtolEqual(output1, output2)


if __name__ == "__main__":
    run_tests()
