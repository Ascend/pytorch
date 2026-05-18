import ctypes
import inspect
from pathlib import Path
import os
import stat
import pathlib
import subprocess
import unittest
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices, SkipIfNotGteCANNVersion


try:
    import torch_test_cpp_extension.npu as npu_extension
    import torch_test_cpp_extension.npu_from_blob as from_blob_ext
except ImportError as e:
    raise RuntimeError(
        "test_cpp_extensions_aot.py cannot be invoked directly. Run "
        "`python run_cpp_test.py` instead.") from e


class TestCppExtensionAOT(TestCase):
    """Tests ahead-of-time cpp extensions
    """

    def test_npu_extension(self):
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = npu_extension.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

        z = npu_extension.tanh_add(x.npu(), y.npu())
        expect_out = x.npu().tanh() + y.npu().tanh()
        self.assertEqual(z.cpu(), expect_out.cpu())

        npu_z = npu_extension.npu_add(x.npu(), y.npu())
        self.assertEqual(npu_z.cpu(), (x + y))

    def test_dispatch_allreduce(self):
        flags = os.O_WRONLY | os.O_RDONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR

        code_file = os.path.join(pathlib.Path(__file__).absolute().parent, "dispatch_allreduce.py")
        log_pth = "allreduce.log"
        with os.fdopen(os.open(log_pth, flags, modes), "w") as f:
            cmd = ["torchrun", "--nproc-per-node=1", code_file]
            p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=f)
            p.wait()

        timeout = 0
        with open(log_pth, 'r', encoding='utf-8') as f:
            tmp = f.readlines()
            for t in tmp:
                print(t)
                if "dispatch timeout" in t:
                    timeout += 1

        os.remove(log_pth)
        self.assertEqual(timeout, 1)

    def test_op_hook_with_add(self):
        # init
        input_1 = torch.tensor((4, 4))
        input_2 = torch.tensor((4, 4))
        expected = torch.add(input_1, input_2)
        input_1 = input_1.npu()
        input_2 = input_2.npu()
        npu_extension.reset_op_hook_call_count()
        count = npu_extension.get_op_hook_call_count()
        self.assertEqual(count, 0)

        # register op_hook, but not enable it
        npu_extension.register_op_hook()
        output_1 = torch.add(input_1, input_2)
        count = npu_extension.get_op_hook_call_count()
        self.assertEqual(count, 0)

        # enable op_hook
        torch.npu.set_option({"OP_HOOK_ENABLE": "enable"})
        output_2 = torch.add(input_1, input_2)
        count = npu_extension.get_op_hook_call_count()
        self.assertEqual(count, 5)

        # disable op_hook
        torch.npu.set_option({"OP_HOOK_ENABLE": "disable"})
        output_3 = torch.add(input_1, input_2)
        count = npu_extension.get_op_hook_call_count()
        self.assertEqual(count, 5)

        # final
        self.assertEqual(output_1.cpu(), expected)
        self.assertEqual(output_2.cpu(), expected)
        self.assertEqual(output_3.cpu(), expected)

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_op_hook_with_all_reduce(cls, rank, input1, world_size, init_pg, c2p):
        # init
        dist_group = init_pg(rank, world_size)
        dst = 0
        input_1 = input1.npu()
        input_2 = input1.npu()
        input_3 = input1.npu()
        npu_extension.reset_op_hook_call_count()
        count_1 = npu_extension.get_op_hook_call_count()

        # register op_hook, but not enable it
        npu_extension.register_op_hook()
        dist_group.all_reduce(input_1)
        count_2 = npu_extension.get_op_hook_call_count()

        # enable op_hook
        torch.npu.set_option({"OP_HOOK_ENABLE": "enable"})
        dist_group.all_reduce(input_2)
        count_3 = npu_extension.get_op_hook_call_count()

        # disable op_hook
        torch.npu.set_option({"OP_HOOK_ENABLE": "disable"})
        dist_group.all_reduce(input_3)
        count_4 = npu_extension.get_op_hook_call_count()

        # final
        all_reduce_ouput = (input_1.cpu(), input_2.cpu(), input_3.cpu())
        op_hook_count = (count_1, count_2, count_3, count_4)
        c2p.put((rank, dst, all_reduce_ouput, op_hook_count))

    def _test_multiprocess(self, f, init_pg, expected, input1, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1.cpu(), world_size, init_pg, c2p))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, dst, all_reduce_ouput, op_hook_count = c2p.get()
            output_1, output_2, output_3 = all_reduce_ouput
            count_1, count_2, count_3, count_4 = op_hook_count
            if rank == dst:
                self.assertEqual(count_1, 0)
                self.assertEqual(count_2, 0)
                self.assertEqual(count_3, 3)
                self.assertEqual(count_4, 3)
                self.assertEqual(output_1, expected,
                                 ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output_1))
                self.assertEqual(output_2, expected,
                                 ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output_2))
                self.assertEqual(output_3, expected,
                                 ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output_3))

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_op_hook_with_all_reduce(self):
        # CI currently supports only 2 devices
        ranks = [2]
        shape = [np.float32, 2, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            expected = 0
            for _ in range(world_size):
                expected += exp_input
            self._test_multiprocess(TestCppExtensionAOT._test_op_hook_with_all_reduce,
                                    TestCppExtensionAOT._init_dist_hccl, expected, input1, world_size)

    def test_dump_allreduce(self):
        dump_pth = "./hccl_trace_rank_0"
        code_file = os.path.join(pathlib.Path(__file__).absolute().parent, "dump_allreduce.py")
        cmd = ["torchrun", "--nproc-per-node=1", code_file]
        p = subprocess.Popen(cmd)
        p.wait()

        self.assertTrue(os.path.exists(dump_pth))
        self.assertTrue(os.path.exists(dump_pth + "_py_traceback"))
        os.remove(dump_pth)
        os.remove(dump_pth + "_py_traceback")


class TestFromBlob(TestCase):
    """Tests for at_npu::native::from_blob interface"""

    def test_from_blob_basic(self):
        self.assertTrue(from_blob_ext.test_from_blob_basic())

    def test_from_blob_deleter(self):
        self.assertTrue(from_blob_ext.test_from_blob_deleter())

    def test_from_blob_strides(self):
        self.assertTrue(from_blob_ext.test_from_blob_strides())

    def test_from_blob_storage_offset(self):
        self.assertTrue(from_blob_ext.test_from_blob_storage_offset())

    def test_from_blob_storage_offset_2d(self):
        self.assertTrue(from_blob_ext.test_from_blob_storage_offset_2d())

    def test_from_blob_storage_offset_dtype(self):
        self.assertTrue(from_blob_ext.test_from_blob_storage_offset_dtype())

    def test_from_blob_storage_offset_contiguous(self):
        self.assertTrue(from_blob_ext.test_from_blob_storage_offset_contiguous())

    def test_from_blob_non_owning(self):
        self.assertTrue(from_blob_ext.test_from_blob_non_owning())

    def test_from_blob_clone(self):
        self.assertTrue(from_blob_ext.test_from_blob_clone())


class TestStableLibtorch(TestCase):
    @classmethod
    def setUpClass(cls):
        so_files = list(Path(inspect.getfile(npu_extension)).parent.glob("*libtorch*"))
        with torch._ops.dl_open_guard():
            loaded_lib = ctypes.CDLL(str(so_files[0]))

    def test_my__adaptive_avg_pool2d(self):
        self_ = torch.randn(1, 3, 6, 6).npu()
        output_size = [3, 3]
        res = torch.ops.libtorch_agn_211.my__adaptive_avg_pool2d(self_, output_size)
        expected = torch._adaptive_avg_pool2d(self_, output_size)
        self.assertEqual(res, expected)

    def test_my__adaptive_avg_pool2d_backward(self):
        grad_output = torch.randn(1, 3, 3, 3).npu()
        self_ = torch.randn(1, 3, 6, 6).npu()
        res = torch.ops.libtorch_agn_211.my__adaptive_avg_pool2d_backward(
            grad_output, self_
        )
        expected = torch.ops.aten._adaptive_avg_pool2d_backward(grad_output, self_)
        self.assertEqual(res, expected)

    def test_my__adaptive_avg_pool3d(self):
        self_ = torch.randn(1, 3, 6, 6, 6).npu()
        output_size = [3, 3, 3]
        res = torch.ops.libtorch_agn_211.my__adaptive_avg_pool3d(self_, output_size)
        expected = torch._adaptive_avg_pool3d(self_, output_size)
        self.assertEqual(res, expected)

    def test_my__adaptive_avg_pool3d_backward(self):
        grad_output = torch.randn(1, 3, 3, 3, 3).npu()
        self_ = torch.randn(1, 3, 6, 6, 6).npu()
        res = torch.ops.libtorch_agn_211.my__adaptive_avg_pool3d_backward(
            grad_output, self_
        )
        expected = torch.ops.aten._adaptive_avg_pool3d_backward(grad_output, self_)
        self.assertEqual(res, expected)

    def test_my__cdist_backward(self):
        grad = torch.randn(2, 3, 4).npu()
        x1 = torch.randn(2, 3, 5).npu()
        x2 = torch.randn(2, 4, 5).npu()
        p = 2.0
        cdist = torch.randn(2, 3, 4).npu()
        res = torch.ops.libtorch_agn_211.my__cdist_backward(grad, x1, x2, p, cdist)
        expected = torch.ops.aten._cdist_backward(grad, x1, x2, p, cdist)
        self.assertEqual(res, expected)

    def test_my__cdist_forward(self):
        x1 = torch.randn(2, 3, 5).npu()
        x2 = torch.randn(2, 4, 5).npu()
        p = 2.0
        compute_mode = None
        res = torch.ops.libtorch_agn_211.my__cdist_forward(x1, x2, p, compute_mode)
        expected = torch.ops.aten._cdist_forward(x1, x2, p, compute_mode)
        self.assertEqual(res, expected)

    def test_my__embedding_bag(self):
        weight = torch.randn(10, 3).npu()
        indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long).npu()
        offsets = torch.tensor([0, 4], dtype=torch.long).npu()
        scale_grad_by_freq = False
        mode = 0
        sparse = False
        per_sample_weights = torch.ones(8).npu()
        include_last_offset = False
        padding_idx = -1
        res = torch.ops.libtorch_agn_211.my__embedding_bag(
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        )
        expected = torch._embedding_bag(
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        )
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my__embedding_bag_forward_only(self):
        weight = torch.randn(10, 3).npu()
        indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long).npu()
        offsets = torch.tensor([0, 4], dtype=torch.long).npu()
        scale_grad_by_freq = False
        mode = 0
        sparse = False
        per_sample_weights = torch.ones(8).npu()
        include_last_offset = False
        padding_idx = -1
        res = torch.ops.libtorch_agn_211.my__embedding_bag_forward_only(
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        )
        expected = torch._embedding_bag_forward_only(
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        )
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my__embedding_bag_per_sample_weights_backward(self):
        grad = torch.randn(2, 3).npu()
        weight = torch.randn(10, 3).npu()
        indices = torch.tensor([1, 2, 4, 5], dtype=torch.long).npu()
        offsets = torch.tensor([0, 2], dtype=torch.long).npu()
        offset2bag = torch.tensor([0, 0, 1, 1], dtype=torch.long).npu()
        mode = 0
        padding_idx = -1
        res = torch.ops.libtorch_agn_211.my__embedding_bag_per_sample_weights_backward(
            grad, weight, indices, offsets, offset2bag, mode, padding_idx
        )
        expected = torch.ops.aten._embedding_bag_per_sample_weights_backward(
            grad, weight, indices, offsets, offset2bag, mode, padding_idx
        )
        self.assertEqual(res, expected)

    def test_my__fft_c2c(self):
        self_ = torch.randn(4, 4, dtype=torch.complex64).npu()
        dim = [0]
        normalization = 0
        forward = True
        res = torch.ops.libtorch_agn_211.my__fft_c2c(self_, dim, normalization, forward)
        expected = torch.fft.fftn(self_, dim=dim, norm="backward")
        self.assertTrue(torch.allclose(res.cpu(), expected.cpu()))

    def test_my__fft_r2c(self):
        self_ = torch.randn(4, 4).npu()
        dim = [0]
        normalization = 0
        onesided = True
        res = torch.ops.libtorch_agn_211.my__fft_r2c(
            self_, dim, normalization, onesided
        )
        expected = torch.fft.rfft(self_, dim=dim[0], norm="backward")
        self.assertTrue(torch.allclose(res.cpu(), expected.cpu()))

    def test_my__fused_moving_avg_obs_fq_helper_functional(self):
        self_ = torch.ones(4, 4).npu()
        observer_on = torch.ones(1).npu()
        fake_quant_on = torch.ones(1).npu()
        running_min = torch.ones(1).npu()
        running_max = torch.ones(1).npu()
        scale = torch.ones(1).npu()
        zero_point = torch.ones(1).npu()
        averaging_const = 0.01
        quant_min = 0
        quant_max = 255
        ch_axis = -1
        per_row_fake_quant = False
        symmetric_quant = False
        res = torch.ops.libtorch_agn_211.my__fused_moving_avg_obs_fq_helper_functional(
            self_,
            observer_on,
            fake_quant_on,
            running_min,
            running_max,
            scale,
            zero_point,
            averaging_const,
            quant_min,
            quant_max,
            ch_axis,
            per_row_fake_quant,
            symmetric_quant,
        )
        expected = torch.ops.aten._fused_moving_avg_obs_fq_helper_functional(
            self_,
            observer_on,
            fake_quant_on,
            running_min,
            running_max,
            scale,
            zero_point,
            averaging_const,
            quant_min,
            quant_max,
            ch_axis,
            per_row_fake_quant,
            symmetric_quant,
        )
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my__fused_rms_norm(self):
        inp = torch.randn(3, 4).npu()
        normalized_shape = [3, 4]
        weight = torch.ones(3, 4).npu()
        eps = 1e-5
        res = torch.ops.libtorch_agn_211.my__fused_rms_norm(
            inp, normalized_shape, weight, eps
        )
        expected = torch.ops.aten._fused_rms_norm(inp, normalized_shape, weight, eps)
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my__pdist_forward(self):
        self_ = torch.randn(4, 3).npu()
        p = 2.0
        res = torch.ops.libtorch_agn_211.my__pdist_forward(self_, p)
        expected = torch.ops.aten._pdist_forward(self_, p)
        self.assertEqual(res, expected)

    # not implemented
    def notest_my__scaled_dot_product_fused_attention_overrideable(self):
        query = torch.randn(2, 4, 8, 16).npu()
        key = torch.randn(2, 4, 8, 16).npu()
        value = torch.randn(2, 4, 8, 16).npu()
        attn_bias = torch.zeros(2, 4, 8, 8).npu()
        dropout_p = 0.0
        is_causal = False
        return_debug_mask = False
        scale = 1.0
        res = torch.ops.libtorch_agn_211.my__scaled_dot_product_fused_attention_overrideable(
            query, key, value, attn_bias, dropout_p, is_causal, return_debug_mask, scale
        )

    # not implemented
    def notest_my__scaled_dot_product_fused_attention_overrideable_backward(self):
        grad_out = torch.randn(2, 4, 8, 16).npu()
        query = torch.randn(2, 4, 8, 16).npu()
        key = torch.randn(2, 4, 8, 16).npu()
        value = torch.randn(2, 4, 8, 16).npu()
        attn_bias = torch.zeros(2, 4, 8, 8).npu()
        grad_input_mask = [True, True, True]
        out = torch.randn(2, 4, 8, 16).npu()
        logsumexp = torch.randn(2, 4, 8).npu()
        cum_seq_q = torch.zeros(1, dtype=torch.int32).npu()
        cum_seq_k = torch.zeros(1, dtype=torch.int32).npu()
        max_q = 8
        max_k = 8
        dropout_p = 0.0
        is_causal = False
        philox_seed = torch.tensor(0, dtype=torch.int64).npu()
        philox_offset = torch.tensor(0, dtype=torch.int64).npu()
        scale = None
        res = torch.ops.libtorch_agn_211.my__scaled_dot_product_fused_attention_overrideable_backward(
            grad_out,
            query,
            key,
            value,
            attn_bias,
            grad_input_mask,
            out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale,
        )

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_my__thnn_fused_lstm_cell(self):
        input_gates = torch.randn(4, 12).npu()
        hidden_gates = torch.randn(4, 12).npu()
        cx = torch.randn(4, 3).npu()
        input_bias = torch.randn(12).npu()
        hidden_bias = torch.randn(12).npu()
        res = torch.ops.libtorch_agn_211.my__thnn_fused_lstm_cell(
            input_gates, hidden_gates, cx, input_bias, hidden_bias
        )
        expected = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, cx, input_bias, hidden_bias
        )
        self.assertEqual(res, expected)

    def test_my__trilinear(self):
        i1 = torch.randn(4, 3, 5).npu()
        i2 = torch.randn(4, 3, 5).npu()
        i3 = torch.randn(4, 3, 5).npu()
        expand1 = [4, 3, 5]
        expand2 = [4, 3, 5]
        expand3 = [4, 3, 5]
        sumdim = [1, 2]
        unroll_dim = 1
        res = torch.ops.libtorch_agn_211.my__trilinear(
            i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim
        )
        expected = torch._trilinear(
            i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim
        )
        self.assertEqual(res, expected)

    def test_my_abs(self):
        self_ = torch.randn(2, 3).npu()
        res = torch.ops.libtorch_agn_211.my_abs(self_)

        self.assertEqual(res, torch.abs(self_))

    def test_my_adaptive_max_pool2d(self):
        self_ = torch.randn(1, 3, 6, 6).npu()
        output_size = [3, 3]
        res = torch.ops.libtorch_agn_211.my_adaptive_max_pool2d(self_, output_size)
        expected = torch.nn.functional.adaptive_max_pool2d(
            self_, output_size, return_indices=True
        )
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_adaptive_max_pool2d_backward(self):
        grad_output = torch.randn(1, 3, 3, 3).npu()
        self_ = torch.randn(1, 3, 6, 6).npu()
        indices = torch.zeros(1, 3, 3, 3, dtype=torch.long).npu()
        res = torch.ops.libtorch_agn_211.my_adaptive_max_pool2d_backward(
            grad_output, self_, indices
        )
        expected = torch.ops.aten.adaptive_max_pool2d_backward(
            grad_output, self_, indices
        )
        self.assertEqual(res, expected)

    def test_my_adaptive_max_pool3d(self):
        self_ = torch.randn(1, 3, 6, 6, 6).npu()
        output_size = [3, 3, 3]
        res = torch.ops.libtorch_agn_211.my_adaptive_max_pool3d(self_, output_size)
        expected = torch.nn.functional.adaptive_max_pool3d(
            self_, output_size, return_indices=True
        )
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_adaptive_max_pool3d_backward(self):
        grad_output = torch.randn(1, 3, 3, 3, 3).npu()
        self_ = torch.randn(1, 3, 6, 6, 6).npu()
        indices = torch.zeros(1, 3, 3, 3, 3, dtype=torch.int32).npu()
        res = torch.ops.libtorch_agn_211.my_adaptive_max_pool3d_backward(
            grad_output, self_, indices
        )
        expected = torch.ops.aten.adaptive_max_pool3d_backward(
            grad_output, self_, indices
        )
        self.assertEqual(res, expected)

    def test_my_add_Scalar(self):
        self_ = torch.randn(2, 3).npu()
        other = 1.0
        alpha = 1.0
        res = torch.ops.libtorch_agn_211.my_add_Scalar(self_, other, alpha)
        expected = torch.add(self_, other, alpha=alpha)
        self.assertEqual(res, expected)

    def test_my_add_Tensor(self):
        self_ = torch.randn(2, 3).npu()
        other = torch.randn(2, 3).npu()
        alpha = 1.0
        res = torch.ops.libtorch_agn_211.my_add_Tensor(self_, other, alpha)
        expected = torch.add(self_, other, alpha=alpha)
        self.assertEqual(res, expected)

    def test_my_addbmm(self):
        self_ = torch.randn(2, 3).npu()
        batch1 = torch.randn(4, 2, 5).npu()
        batch2 = torch.randn(4, 5, 3).npu()
        beta = 1.0
        alpha = 1.0
        res = torch.ops.libtorch_agn_211.my_addbmm(self_, batch1, batch2, beta, alpha)
        expected = torch.addbmm(self_, batch1, batch2, beta=beta, alpha=alpha)
        self.assertEqual(res, expected)

    def test_my_addmm_out(self):
        out = torch.randn(2, 3).npu()
        self_ = torch.randn(2, 3).npu()
        mat1 = torch.randn(2, 4).npu()
        mat2 = torch.randn(4, 3).npu()
        beta = 1.0
        alpha = 1.0
        res = torch.ops.libtorch_agn_211.my_addmm_out(
            out, self_, mat1, mat2, beta, alpha
        )
        expected = torch.addmm(self_, mat1, mat2, beta=beta, alpha=alpha)
        self.assertEqual(out, expected)

    def test_my_addmv(self):
        self_ = torch.randn(2).npu()
        mat = torch.randn(2, 3).npu()
        vec = torch.randn(3).npu()
        beta = 1.0
        alpha = 1.0
        res = torch.ops.libtorch_agn_211.my_addmv(self_, mat, vec, beta, alpha)
        expected = torch.addmv(self_, mat, vec, beta=beta, alpha=alpha)
        self.assertEqual(res, expected)

    def test_my_angle(self):
        self_ = torch.randn(2, 3, dtype=torch.complex64).npu()
        res = torch.ops.libtorch_agn_211.my_angle(self_)
        expected = torch.angle(self_)
        self.assertEqual(res, expected)

    def test_my_avg_pool2d(self):
        self_ = torch.randn(1, 3, 6, 6).npu()
        kernel_size = [3, 3]
        stride = [3, 3]
        padding = [0, 0]
        ceil_mode = False
        count_include_pad = True
        divisor_override = None
        res = torch.ops.libtorch_agn_211.my_avg_pool2d(
            self_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
        expected = torch.nn.functional.avg_pool2d(
            self_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override=None,
        )
        self.assertEqual(res, expected)

    def test_my_avg_pool2d_backward(self):
        grad_output = torch.randn(1, 3, 2, 2).npu()
        self_ = torch.randn(1, 3, 6, 6).npu()
        kernel_size = [3, 3]
        stride = [3, 3]
        padding = [0, 0]
        ceil_mode = False
        count_include_pad = True
        divisor_override = None
        res = torch.ops.libtorch_agn_211.my_avg_pool2d_backward(
            grad_output,
            self_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
        expected = torch.ops.aten.avg_pool2d_backward(
            grad_output,
            self_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override=None,
        )
        self.assertEqual(res, expected)

    def test_my_avg_pool3d(self):
        self_ = torch.randn(1, 3, 6, 6, 6).npu()
        kernel_size = [3, 3, 3]
        stride = [3, 3, 3]
        padding = [0, 0, 0]
        ceil_mode = False
        count_include_pad = True
        divisor_override = None
        res = torch.ops.libtorch_agn_211.my_avg_pool3d(
            self_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
        expected = torch.nn.functional.avg_pool3d(
            self_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override=None,
        )
        self.assertEqual(res, expected)

    def test_my_avg_pool3d_backward(self):
        grad_output = torch.randn(1, 3, 2, 2, 2).npu()
        self_ = torch.randn(1, 3, 6, 6, 6).npu()
        kernel_size = [3, 3, 3]
        stride = [3, 3, 3]
        padding = [0, 0, 0]
        ceil_mode = False
        count_include_pad = True
        divisor_override = None
        res = torch.ops.libtorch_agn_211.my_avg_pool3d_backward(
            grad_output,
            self_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
        expected = torch.ops.aten.avg_pool3d_backward(
            grad_output,
            self_,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override=None,
        )
        self.assertEqual(res, expected)

    def test_my_baddbmm_out(self):
        out = torch.randn(4, 2, 3).npu()
        self_ = torch.randn(4, 2, 3).npu()
        batch1 = torch.randn(4, 2, 5).npu()
        batch2 = torch.randn(4, 5, 3).npu()
        beta = 1.0
        alpha = 1.0
        res = torch.ops.libtorch_agn_211.my_baddbmm_out(
            out, self_, batch1, batch2, beta, alpha
        )
        expected = torch.baddbmm(self_, batch1, batch2, beta=beta, alpha=alpha)
        self.assertEqual(out, expected)

    def test_my_bernoulli__Tensor(self):
        self_ = torch.rand(2, 3).npu()
        p = torch.rand(2, 3).npu()
        self_clone = self_.clone()
        seed = 42
        torch.manual_seed(seed)
        torch.ops.libtorch_agn_211.my_bernoulli__Tensor(self_, p, None)
        torch.manual_seed(seed)
        self_clone.bernoulli_(p)
        self.assertEqual(self_, self_clone)

    def test_my_bernoulli__float(self):
        self_ = torch.rand(2, 3).npu()
        p = 0.5
        self_clone = self_.clone()
        seed = 42
        torch.manual_seed(seed)
        torch.ops.libtorch_agn_211.my_bernoulli__float(self_, p, None)
        torch.manual_seed(seed)
        self_clone.bernoulli_(p)
        self.assertEqual(self_, self_clone)

    def test_my_bmm_out(self):
        out = torch.randn(4, 2, 3).npu()
        self_ = torch.randn(4, 2, 5).npu()
        mat2 = torch.randn(4, 5, 3).npu()
        res = torch.ops.libtorch_agn_211.my_bmm_out(out, self_, mat2)
        expected = torch.bmm(self_, mat2)
        self.assertEqual(out, expected)

    def test_my_bucketize_Tensor(self):
        self_ = torch.tensor([1, 3, 5, 7, 9]).npu()
        boundaries = torch.tensor([2, 4, 6, 8]).npu()
        out_int32 = False
        right = False
        res = torch.ops.libtorch_agn_211.my_bucketize_Tensor(
            self_, boundaries, out_int32, right
        )
        expected = torch.bucketize(self_, boundaries, out_int32=out_int32, right=right)
        self.assertEqual(res, expected)

    def test_my_cat(self):
        tensors = [torch.randn(2, 3).npu(), torch.randn(2, 3).npu()]
        dim = 0
        res = torch.ops.libtorch_agn_211.my_cat(tensors, dim)
        expected = torch.cat(tensors, dim)
        self.assertEqual(res, expected)

    def test_my_cholesky_solve(self):
        self_ = torch.randn(3, 3).npu()
        input2 = torch.randn(3, 3).triu().npu()
        upper = True
        res = torch.ops.libtorch_agn_211.my_cholesky_solve(self_, input2, upper)
        expected = torch.cholesky_solve(self_, input2, upper=upper)
        self.assertEqual(res, expected)

    def test_my_convolution(self):
        inp = torch.randn(1, 3, 4).npu()
        weight = torch.randn(3, 3, 3).npu()
        bias = torch.randn(3).npu()
        stride = [1]
        padding = [0]
        dilation = [1]
        transposed = False
        output_padding = [0]
        groups = 1
        res = torch.ops.libtorch_agn_211.my_convolution(
            inp,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        expected = torch.conv1d(inp, weight, bias, stride, padding, dilation, groups)
        self.assertEqual(res, expected)

    def test_my_convolution_backward(self):
        grad_output = torch.randn(1, 3, 2).npu()
        inp = torch.randn(1, 3, 4).npu()
        weight = torch.randn(3, 3, 3).npu()
        bias_sizes = [3]
        stride = [1]
        padding = [0]
        dilation = [1]
        transposed = False
        output_padding = [0]
        groups = 1
        output_mask = [True, True, True]
        res = torch.ops.libtorch_agn_211.my_convolution_backward(
            grad_output,
            inp,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
        expected = torch.ops.aten.convolution_backward(
            grad_output,
            inp,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_cummax(self):
        self_ = torch.randn(2, 3).npu()
        dim = 1
        res = torch.ops.libtorch_agn_211.my_cummax(self_, dim)
        expected = torch.cummax(self_, dim)
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_cummin(self):
        self_ = torch.randn(2, 3).npu()
        dim = 1
        res = torch.ops.libtorch_agn_211.my_cummin(self_, dim)
        expected = torch.cummin(self_, dim)
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_cumsum(self):
        self_ = torch.randn(2, 3).npu()
        dim = 1
        dtype = None
        res = torch.ops.libtorch_agn_211.my_cumsum(self_, dim, dtype)
        expected = torch.cumsum(self_, dim, dtype=dtype)
        self.assertEqual(res, expected)

    def test_my_exponential(self):
        self_ = torch.rand(2, 3).npu()
        lambd = 1.0
        generator = None
        # generator = torch.Generator(device="npu").manual_seed(12345)
        self_clone = self_.clone()
        torch.ops.libtorch_agn_211.my_exponential(self_, lambd, generator)
        self_clone.exponential_(lambd)
        self.assertEqual(self_.shape, self_clone.shape)

    def test_my_fill__Scalar(self):
        self_ = torch.randn(2, 3).npu()
        value = 1.0
        self_clone = self_.clone()
        torch.ops.libtorch_agn_211.my_fill__Scalar(self_, value)
        self_clone.fill_(value)
        self.assertEqual(self_, self_clone)

    def test_my_grid_sampler_2d_backward(self):
        grad_output = torch.randn(1, 1, 2, 2).npu()
        inp = torch.randn(1, 1, 3, 3).npu()
        grid = torch.randn(1, 2, 2, 2).npu()
        interpolation_mode = 0
        padding_mode = 0
        align_corners = True
        output_mask = [True, True]
        res = torch.ops.libtorch_agn_211.my_grid_sampler_2d_backward(
            grad_output,
            inp,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
            output_mask,
        )
        expected = torch.ops.aten.grid_sampler_2d_backward(
            grad_output,
            inp,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
            output_mask,
        )
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_hann_window(self):
        window_length = 5
        dtype = None
        layout = None
        device = None
        pin_memory = None
        res = torch.ops.libtorch_agn_211.my_hann_window(
            window_length, dtype, layout, device, pin_memory
        )
        expected = torch.hann_window(window_length)
        self.assertEqual(res, expected)

    def test_my_histc(self):
        self_ = torch.randn(10).npu()
        bins = 5
        min_ = 0.0
        max_ = 1.0
        res = torch.ops.libtorch_agn_211.my_histc(self_, bins, min_, max_)
        expected = torch.histc(self_, bins, min_, max_)
        self.assertEqual(res, expected)

    def test_my_index_Tensor(self):
        self_ = torch.randn(3, 4, 5).npu()
        indices = [torch.tensor([0, 2]).npu()]
        res = torch.ops.libtorch_agn_211.my_index_Tensor(self_, indices)
        expected = self_[indices]
        self.assertEqual(res, expected)

    def test_my_index_put(self):
        self_ = torch.randn(3, 4).npu()
        indices = [torch.tensor([0, 2]).npu()]
        values = torch.randn(2, 4).npu()
        accumulate = False
        res = torch.ops.libtorch_agn_211.my_index_put(
            self_, indices, values, accumulate
        )
        expected = self_.index_put(indices, values, accumulate)
        self.assertEqual(res, expected)

    def test_my_kthvalue(self):
        self_ = torch.randn(3, 5).npu()
        k = 2
        dim = 1
        keepdim = False
        res = torch.ops.libtorch_agn_211.my_kthvalue(self_, k, dim, keepdim)
        expected = torch.kthvalue(self_, k, dim, keepdim)
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_logcumsumexp(self):
        self_ = torch.randn(2, 3).npu()
        dim = 1
        res = torch.ops.libtorch_agn_211.my_logcumsumexp(self_, dim)
        expected = torch.logcumsumexp(self_, dim)
        self.assertEqual(res, expected)

    def test_my_masked_scatter(self):
        self_ = torch.randn(2, 3).npu()
        mask = torch.tensor([[True, False, True], [False, True, False]]).npu()
        source = torch.randn(3).npu()
        res = torch.ops.libtorch_agn_211.my_masked_scatter(self_, mask, source)
        expected = self_.masked_scatter(mask, source)
        self.assertEqual(res, expected)

    def test_my_masked_scatter_backward(self):
        grad_output = torch.randn(2, 3).npu()
        mask = torch.tensor([[True, False, True], [False, True, False]]).npu()
        sizes = [2, 3]
        res = torch.ops.libtorch_agn_211.my_masked_scatter_backward(
            grad_output, mask, sizes
        )
        expected = torch.ops.aten.masked_scatter_backward(grad_output, mask, sizes)
        self.assertEqual(res, expected)

    def test_my_masked_select(self):
        self_ = torch.randn(2, 3).npu()
        mask = torch.tensor([[True, False, True], [False, True, False]]).npu()
        res = torch.ops.libtorch_agn_211.my_masked_select(self_, mask)
        expected = torch.masked_select(self_, mask)
        self.assertEqual(res, expected)

    def test_my_max_pool2d_with_indices(self):
        self_ = torch.randn(1, 3, 6, 6).npu()
        kernel_size = [3, 3]
        stride = [3, 3]
        padding = [0, 0]
        dilation = [1, 1]
        ceil_mode = False
        res = torch.ops.libtorch_agn_211.my_max_pool2d_with_indices(
            self_, kernel_size, stride, padding, dilation, ceil_mode
        )
        expected = torch.nn.functional.max_pool2d(
            self_,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            return_indices=True,
        )
        self.assertEqual(res[0], expected[0])
        # self.assertEqual(res[1].shape, expected[1].shape)

    def test_my_max_pool2d_with_indices_backward(self):
        grad_output = torch.randn(1, 3, 2, 2).npu()
        self_ = torch.randn(1, 3, 6, 6).npu()
        kernel_size = [3, 3]
        stride = [3, 3]
        padding = [0, 0]
        dilation = [1, 1]
        ceil_mode = False
        indices = torch.zeros(1, 3, 2, 2, dtype=torch.int8).npu()
        res = torch.ops.libtorch_agn_211.my_max_pool2d_with_indices_backward(
            grad_output,
            self_,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices,
        )
        expected = torch.ops.aten.max_pool2d_with_indices_backward(
            grad_output,
            self_,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices,
        )
        self.assertEqual(res, expected)

    def test_my_max_pool3d_with_indices(self):
        self_ = torch.randn(1, 3, 6, 6, 6).npu()
        kernel_size = [3, 3, 3]
        stride = [3, 3, 3]
        padding = [0, 0, 0]
        dilation = [1, 1, 1]
        ceil_mode = False
        res = torch.ops.libtorch_agn_211.my_max_pool3d_with_indices(
            self_, kernel_size, stride, padding, dilation, ceil_mode
        )
        expected = torch.nn.functional.max_pool3d(
            self_,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            return_indices=True,
        )
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_max_pool3d_with_indices_backward(self):
        grad_output = torch.randn(1, 3, 2, 2, 2).npu()
        self_ = torch.randn(1, 3, 6, 6, 6).npu()
        kernel_size = [3, 3, 3]
        stride = [3, 3, 3]
        padding = [0, 0, 0]
        dilation = [1, 1, 1]
        ceil_mode = False
        indices = torch.zeros(1, 3, 2, 2, 2, dtype=torch.int32).npu()
        res = torch.ops.libtorch_agn_211.my_max_pool3d_with_indices_backward(
            grad_output,
            self_,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices,
        )
        expected = torch.ops.aten.max_pool3d_with_indices_backward(
            grad_output,
            self_,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices,
        )
        self.assertEqual(res, expected)

    def test_my_max_unpool2d(self):
        self_ = torch.randn(1, 3, 2, 2).npu()
        indices = torch.zeros(1, 3, 2, 2, dtype=torch.long).npu()
        output_size = [6, 6]
        res = torch.ops.libtorch_agn_211.my_max_unpool2d(self_, indices, output_size)
        expected = torch.ops.aten.max_unpool2d(self_, indices, output_size)
        self.assertEqual(res, expected)

    def test_my_max_unpool3d(self):
        self_ = torch.randn(1, 3, 2, 2, 2).npu()
        indices = torch.zeros(1, 3, 2, 2, 2, dtype=torch.long).npu()
        output_size = [6, 6, 6]
        stride = [3, 3, 3]
        padding = [0, 0, 0]
        res = torch.ops.libtorch_agn_211.my_max_unpool3d(
            self_, indices, output_size, stride, padding
        )
        expected = torch.ops.aten.max_unpool3d(
            self_, indices, output_size, stride, padding
        )
        self.assertEqual(res, expected)

    def test_my_median(self):
        self_ = torch.randn(2, 3).npu()
        res = torch.ops.libtorch_agn_211.my_median(self_)
        expected = torch.median(self_)
        self.assertEqual(res, expected)

    def test_my_mm_out(self):
        out = torch.randn(2, 3).npu()
        self_ = torch.randn(2, 4).npu()
        mat2 = torch.randn(4, 3).npu()
        res = torch.ops.libtorch_agn_211.my_mm_out(out, self_, mat2)
        expected = torch.mm(self_, mat2)
        self.assertEqual(out, expected)

    def test_my_mul_Scalar(self):
        self_ = torch.randn(2, 3).npu()
        other = 2.0
        res = torch.ops.libtorch_agn_211.my_mul_Scalar(self_, other)
        expected = torch.mul(self_, other)
        self.assertEqual(res, expected)

    def test_my_mul_Tensor(self):
        self_ = torch.randn(2, 3).npu()
        other = torch.randn(2, 3).npu()
        res = torch.ops.libtorch_agn_211.my_mul_Tensor(self_, other)
        expected = torch.mul(self_, other)
        self.assertEqual(res, expected)

    def test_my_nanmedian(self):
        self_ = torch.randn(2, 3).npu()
        res = torch.ops.libtorch_agn_211.my_nanmedian(self_)
        expected = torch.nanmedian(self_)
        self.assertEqual(res, expected)

    def test_my_narrow(self):
        self_ = torch.randn(3, 4).npu()
        dim = 0
        start = 1
        length = 2
        res = torch.ops.libtorch_agn_211.my_narrow(self_, dim, start, length)
        expected = torch.narrow(self_, dim, start, length)
        self.assertEqual(res, expected)

    def test_my_native_dropout(self):
        inp = torch.randn(2, 3).npu()
        p = 0.5
        train = True
        seed = 42
        torch.manual_seed(seed)
        res = torch.ops.libtorch_agn_211.my_native_dropout(inp, p, train)
        torch.manual_seed(seed)
        expected = torch.native_dropout(inp, p, train)
        self.assertEqual(res[0], expected[0])
        self.assertEqual(res[1], expected[1])

    def test_my_nonzero(self):
        self_ = torch.tensor([[1, 0], [0, 1]]).to(torch.float32).npu()
        res = torch.ops.libtorch_agn_211.my_nonzero(self_)
        expected = torch.nonzero(self_)
        self.assertEqual(res, expected)

    def test_my_normal_functional(self):
        self_ = torch.randn(2, 3).npu()
        mean = 0.0
        std = 1.0
        generator = None
        # generator = torch.Generator(device="npu").manual_seed(12345)
        res = torch.ops.libtorch_agn_211.my_normal_functional(
            self_, mean, std, generator
        )
        self.assertEqual(res.shape, self_.shape)

    def test_my_pad(self):
        self_ = torch.randn(2, 3, 4).npu()
        pad = [1, 1]
        mode = "constant"
        value = 0.0
        res = torch.ops.libtorch_agn_211.my_pad(self_, pad, mode, value)
        expected = torch.nn.functional.pad(self_, pad, mode, value)
        self.assertEqual(res, expected)

    def test_my_permute(self):
        self_ = torch.randn(2, 3, 4).npu()
        dims = [1, 2, 0]
        res = torch.ops.libtorch_agn_211.my_permute(self_, dims)
        expected = torch.permute(self_, dims)
        self.assertEqual(res, expected)

    def test_my_polar(self):
        self_ = torch.randn(2, 3).npu()
        angle = torch.randn(2, 3).npu()
        res = torch.ops.libtorch_agn_211.my_polar(self_, angle)
        expected = torch.polar(self_, angle)
        self.assertTrue(torch.allclose(res.cpu(), expected.cpu()))

    def test_my_pow_Scalar(self):
        self_ = 2.0
        exponent = torch.randn(2, 3).npu()
        res = torch.ops.libtorch_agn_211.my_pow_Scalar(self_, exponent)
        expected = torch.pow(self_, exponent)
        self.assertEqual(res, expected)

    def test_my_pow_Tensor_Scalar(self):
        self_ = torch.randn(2, 3).npu()
        exponent = 2.0
        res = torch.ops.libtorch_agn_211.my_pow_Tensor_Scalar(self_, exponent)
        expected = torch.pow(self_, exponent)
        self.assertEqual(res, expected)

    def test_my_pow_Tensor_Tensor(self):
        self_ = torch.randn(2, 3).npu()
        exponent = torch.randn(2, 3).npu()
        res = torch.ops.libtorch_agn_211.my_pow_Tensor_Tensor(self_, exponent)
        expected = torch.pow(self_, exponent)
        self.assertEqual(res, expected)

    def test_my_rand(self):
        size = [2, 3]
        seed = 42
        torch.manual_seed(seed)
        res = torch.ops.libtorch_agn_211.my_rand(size, None, None, None, None)
        torch.manual_seed(seed)
        expected = torch.rand(size)
        self.assertEqual(res.cpu(), expected)

    def test_my_rand_generator(self):
        size = [2, 3]
        seed = 42
        torch.manual_seed(seed)
        res = torch.ops.libtorch_agn_211.my_rand_generator(
            size, None, None, None, None, None
        )
        torch.manual_seed(seed)
        expected = torch.rand(size)
        self.assertEqual(res, expected)

    def test_my_randint(self):
        high = 10
        size = [2, 3]
        seed = 42
        torch.manual_seed(seed)
        res = torch.ops.libtorch_agn_211.my_randint(high, size, None, None, None, None)
        torch.manual_seed(seed)
        expected = torch.randint(high, size)
        self.assertEqual(res.cpu(), expected)

    def test_my_randint_generator(self):
        high = 10
        size = [2, 3]
        seed = 42
        torch.manual_seed(seed)
        res = torch.ops.libtorch_agn_211.my_randint_generator(
            high, size, None, None, None, None, None
        )
        torch.manual_seed(seed)
        expected = torch.randint(high, size)
        self.assertEqual(res, expected)

    def test_my_randint_low(self):
        low = 0
        high = 10
        size = [2, 3]
        seed = 42
        torch.manual_seed(seed)
        res = torch.ops.libtorch_agn_211.my_randint_low(
            low, high, size, None, None, None, None
        )
        torch.manual_seed(seed)
        expected = torch.randint(low, high, size)
        self.assertEqual(res.cpu(), expected)

    def test_my_randint_low_out(self):
        out = torch.empty(2, 3, dtype=torch.long).npu()
        low = 0
        high = 10
        size = [2, 3]
        seed = 42
        torch.manual_seed(seed)
        torch.ops.libtorch_agn_211.my_randint_low_out(out, low, high, size)
        torch.manual_seed(seed)
        expected = torch.randint(low, high, size, device="npu")
        self.assertEqual(out.cpu(), expected.cpu())

    def test_my_randn(self):
        size = [2, 3]
        seed = 42
        torch.manual_seed(seed)
        res = torch.ops.libtorch_agn_211.my_randn(size, None, None, None, None)
        torch.manual_seed(seed)
        expected = torch.randn(size)
        self.assertEqual(res.cpu(), expected)

    def test_my_randn_generator(self):
        size = [2, 3]
        seed = 42
        torch.manual_seed(seed)
        res = torch.ops.libtorch_agn_211.my_randn_generator(
            size, None, None, None, None, None
        )
        torch.manual_seed(seed)
        expected = torch.randn(size)
        self.assertEqual(res, expected)

    def test_my_randperm(self):
        n = 5
        seed = 42
        torch.npu.manual_seed(seed)
        res = torch.ops.libtorch_agn_211.my_randperm(n, None, None, None, None)
        torch.npu.manual_seed(seed)
        expected = torch.randperm(n, device="npu")
        self.assertEqual(res.cpu(), expected.cpu())

    def test_my_replication_pad1d_backward(self):
        grad_output = torch.randn(1, 2, 5).npu()
        self_ = torch.randn(1, 2, 3).npu()
        padding = [1, 1]
        res = torch.ops.libtorch_agn_211.my_replication_pad1d_backward(
            grad_output, self_, padding
        )
        expected = torch.ops.aten.replication_pad1d_backward(
            grad_output, self_, padding
        )
        self.assertEqual(res, expected)

    def test_my_replication_pad2d_backward(self):
        grad_output = torch.randn(1, 2, 5, 5).npu()
        self_ = torch.randn(1, 2, 3, 3).npu()
        padding = [1, 1, 1, 1]
        res = torch.ops.libtorch_agn_211.my_replication_pad2d_backward(
            grad_output, self_, padding
        )
        expected = torch.ops.aten.replication_pad2d_backward(
            grad_output, self_, padding
        )
        self.assertEqual(res, expected)

    def test_my_reshape(self):
        self_ = torch.randn(2, 3, 4).npu()
        shape = [6, 4]
        res = torch.ops.libtorch_agn_211.my_reshape(self_, shape)
        expected = torch.reshape(self_, shape)
        self.assertEqual(res, expected)

    def test_my_resize_(self):
        self_ = torch.randn(2, 3).npu()
        size = [3, 4]
        memory_format = None
        torch.ops.libtorch_agn_211.my_resize_(self_, size, memory_format)
        self.assertEqual(list(self_.shape), size)

    def test_my_resize_as_(self):
        self_ = torch.randn(2, 3).npu()
        the_template = torch.randn(3, 4).npu()
        memory_format = None
        torch.ops.libtorch_agn_211.my_resize_as_(self_, the_template, memory_format)
        self.assertEqual(self_.shape, the_template.shape)

    def test_my_scatter_src_out(self):
        out = torch.randn(2, 3).npu()
        self_ = torch.randn(2, 3).npu()
        dim = 1
        index = torch.randint(0, 3, (2, 3), dtype=torch.long).npu()
        src = torch.randn(2, 3).npu()
        torch.ops.libtorch_agn_211.my_scatter_src_out(out, self_, dim, index, src)
        expected = self_.scatter(dim, index, src)
        self.assertEqual(out, expected)

    def test_my_scatter_value_out(self):
        out = torch.randn(2, 3).npu()
        self_ = torch.randn(2, 3).npu()
        dim = 1
        index = torch.randint(0, 3, (2, 3), dtype=torch.long).npu()
        value = 1.0
        torch.ops.libtorch_agn_211.my_scatter_value_out(out, self_, dim, index, value)
        expected = self_.scatter(dim, index, value)
        self.assertEqual(out, expected)

    def test_my_searchsorted_Scalar(self):
        sorted_sequence = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).npu()
        self_ = 3.0
        out_int32 = False
        right = False
        sorter = torch.arange(5, dtype=torch.long).npu()
        res = torch.ops.libtorch_agn_211.my_searchsorted_Scalar(
            sorted_sequence, self_, out_int32, right, sorter
        )
        expected = torch.searchsorted(
            sorted_sequence, self_, out_int32=out_int32, right=right
        )
        self.assertEqual(res, expected)

    def test_my_searchsorted_Tensor(self):
        sorted_sequence = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).npu()
        self_ = torch.tensor([2.5, 4.5]).npu()
        out_int32 = False
        right = False
        sorter = torch.arange(5, dtype=torch.long).npu()
        res = torch.ops.libtorch_agn_211.my_searchsorted_Tensor(
            sorted_sequence, self_, out_int32, right, sorter
        )
        expected = torch.searchsorted(
            sorted_sequence, self_, out_int32=out_int32, right=right
        )
        self.assertEqual(res, expected)

    def test_my_set__source_Tensor(self):
        self_ = torch.randn(2, 3).npu()
        source = torch.randn(2, 3).npu()
        self_clone = self_.clone()
        torch.ops.libtorch_agn_211.my_set__source_Tensor(self_, source)
        self_clone.set_(source)
        self.assertEqual(self_, self_clone)

    def test_my_slice_Tensor(self):
        self_ = torch.randn(4, 5).npu()
        dim = 1
        start = 0
        end = 3
        step = 1
        res = torch.ops.libtorch_agn_211.my_slice_Tensor(self_, dim, start, end, step)
        idx = [slice(None)] * self_.dim()
        idx[dim] = slice(start, end, step)
        expected = self_[tuple(idx)]
        self.assertEqual(res, expected)

    def test_my_soft_margin_loss_backward(self):
        grad_output = torch.randn(2, 3).npu()
        self_ = torch.randn(2, 3).npu()
        target = torch.randn(2, 3).npu()
        reduction = 1
        res = torch.ops.libtorch_agn_211.my_soft_margin_loss_backward(
            grad_output, self_, target, reduction
        )
        expected = torch.ops.aten.soft_margin_loss_backward(
            grad_output, self_, target, reduction
        )
        self.assertEqual(res, expected)

    def test_my_sort(self):
        self_ = torch.randn(2, 3).npu()
        dim = 1
        descending = False
        res = torch.ops.libtorch_agn_211.my_sort(self_, dim, descending)
        expected = torch.sort(self_, dim, descending)
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_sort_stable(self):
        self_ = torch.randn(2, 3).npu()
        stable = None
        dim = 1
        descending = False
        res = torch.ops.libtorch_agn_211.my_sort_stable(self_, stable, dim, descending)
        expected = torch.sort(self_, stable=True, dim=dim, descending=descending)
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_squeeze_dim(self):
        self_ = torch.randn(2, 3).npu()
        dim = 1
        res = torch.ops.libtorch_agn_211.my_squeeze_dim(self_, dim)
        expected = torch.squeeze(self_, dim)
        self.assertEqual(res, expected)

    def test_my_to_sparse(self):
        self_ = torch.randn(2, 3).npu()
        layout = None
        blocksize = None
        dense_dim = None
        res = torch.ops.libtorch_agn_211.my_to_sparse(
            self_, layout, blocksize, dense_dim
        )
        expected = self_.to_sparse()
        self.assertEqual(res, expected)

    def test_my_topk(self):
        self_ = torch.randn(3, 5).npu()
        k = 2
        dim = 1
        largest = True
        sorted = True
        res = torch.ops.libtorch_agn_211.my_topk(self_, k, dim, largest, sorted)
        expected = torch.topk(self_, k, dim, largest, sorted)
        for r, e in zip(res, expected):
            self.assertEqual(r, e)

    def test_my_uniform(self):
        self_ = torch.randn(2, 3).npu()
        from_ = 1.0
        to = 1.0
        generator = None
        # generator = torch.Generator(device="npu").manual_seed(12345)
        self_clone = self_.clone()
        torch.ops.libtorch_agn_211.my_uniform(self_, from_, to, generator)
        self_clone.uniform_(from_, to)
        self.assertEqual(self_.shape, self_clone.shape)

    def test_my_upsample_bicubic2d_backward(self):
        grad_output = torch.randn(2, 3, 64, 64).npu()
        output_size = [64, 64]
        input_size = [2, 3, 32, 32]
        align_corners = False
        scales_h = None
        scales_w = None
        res = torch.ops.libtorch_agn_211.my_upsample_bicubic2d_backward(
            grad_output, output_size, input_size, align_corners, scales_h, scales_w
        )
        expected = torch.ops.aten.upsample_bicubic2d_backward(
            grad_output, output_size, input_size, align_corners, scales_h, scales_w
        )
        self.assertEqual(res, expected)

    def test_my_upsample_linear1d_backward(self):
        grad_output = torch.randn(1, 1, 3).npu()
        output_size = [3]
        input_size = [1, 1, 6]
        align_corners = False
        scales = None
        res = torch.ops.libtorch_agn_211.my_upsample_linear1d_backward(
            grad_output, output_size, input_size, align_corners, scales
        )
        expected = torch.ops.aten.upsample_linear1d_backward(
            grad_output, output_size, input_size, align_corners, scales
        )
        self.assertEqual(res, expected)

    def test_my_upsample_trilinear3d_backward(self):
        grad_output = torch.randn(2, 2, 2, 2, 2).npu()
        output_size = [2, 2, 2]
        input_size = [2, 2, 1, 1, 1]
        align_corners = False
        scales_d = None
        scales_h = None
        scales_w = None
        res = torch.ops.libtorch_agn_211.my_upsample_trilinear3d_backward(
            grad_output,
            output_size,
            input_size,
            align_corners,
            scales_d,
            scales_h,
            scales_w,
        )
        expected = torch.ops.aten.upsample_trilinear3d_backward(
            grad_output,
            output_size,
            input_size,
            align_corners,
            scales_d,
            scales_h,
            scales_w,
        )
        self.assertEqual(res, expected)

    def test_my_view_dtype(self):
        self_ = torch.randn(2, 3).npu()
        dtype = 0
        res = torch.ops.libtorch_agn_211.my_view_dtype(self_, dtype)
        expected = self_.view(torch.uint8)
        self.assertEqual(res, expected)

    def test_my_view_as_complex(self):
        self_ = torch.randn(2, 3, 2).npu()
        res = torch.ops.libtorch_agn_211.my_view_as_complex(self_)
        expected = torch.view_as_complex(self_)
        self.assertTrue(torch.allclose(res.cpu(), expected.cpu()))

    def test_my_view_as_real(self):
        self_ = torch.randn(2, 3, dtype=torch.complex64).npu()
        res = torch.ops.libtorch_agn_211.my_view_as_real(self_)
        expected = torch.view_as_real(self_)
        self.assertEqual(res, expected)


if __name__ == "__main__":
    run_tests()
