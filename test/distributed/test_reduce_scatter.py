import os
from random import randint
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class HcclReduceScatterTestBase(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    # pylint:disable=huawei-too-many-arguments
    def _test_multiprocess(self, fn, init_pg, expected, input1, world_size, reduce_op=dist.ReduceOp.SUM, use_equal=False):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=fn,
                args=(i, input1, world_size, init_pg, c2p, p2c, reduce_op))
            p.start()
            ps.append(p)
        for _ in range(world_size):
            rank, output = c2p.get()
            # torch has no add/mul stub for UInt64, so assertEqual (which
            # computes output - expected internally) raises. Use torch.equal
            # for uint64, which compares without subtraction.
            if use_equal:
                self.assertTrue(torch.equal(output, expected[rank]),
                                ("rank {} Expect receive tensor {} but got {}.").format(rank, expected[rank], output))
            else:
                self.assertEqual(output, expected[rank],
                                 ("rank {} Expect receive tensor {} but got {}.").format(rank, expected[rank], output))

        for _ in range(world_size):
            p2c.put(0)
        for p in ps:
            p.join()

    def _test_multiprocess_with_error(self, fn, init_pg, input1, world_size):
        ctx = mp.get_context('spawn')

        ps = []
        for i in range(world_size):
            p = ctx.Process(target=fn, args=(i, input1, world_size, init_pg))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")


    def _construct_excepted_result(self, inputs, world_size, op=dist.all_gather, reduce_op=dist.ReduceOp.SUM):
        if op not in [dist.reduce_scatter, dist._reduce_scatter_base, dist.reduce_scatter_tensor,
                      torch_npu.distributed.reduce_scatter_tensor_uneven]:
            raise ValueError("Unsupported op `{}`" % (str(op)))
        if reduce_op == dist.ReduceOp.AVG:
            return [input.cpu() for input in inputs]
        return [input.cpu() * world_size for input in inputs]

    def _numel(self, shape):
        n = 1
        for d in shape:
            n *= d
        return n

    # Expected output for _test_reduce_scatter_lifted. Input convention: the k-th
    # flattened element of rank r's i-th tensor is r*10000 + offset_i + k. After SUM,
    # global position p (< have) reduces to ws*p + 5000*ws*(ws-1); p >= have is 0.
    # rank r takes global [r*out_numel, (r+1)*out_numel).
    def _construct_lifted_expected(self, input_shapes, out_shape, world_size):
        offsets = [0]
        for s in input_shapes:
            offsets.append(offsets[-1] + self._numel(s))
        have = offsets[-1]
        out_numel = self._numel(out_shape)
        base = 5000 * world_size * (world_size - 1)
        expected = []
        for r in range(world_size):
            vals = []
            for k in range(out_numel):
                p = r * out_numel + k
                vals.append(world_size * p + base if p < have else 0.0)
            expected.append(torch.tensor(vals, dtype=torch.float32).reshape(out_shape))
        return expected

    def _test_multiprocess_lifted(self, fn, init_pg, input_shapes, out_shape, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        expected = self._construct_lifted_expected(input_shapes, out_shape, world_size)
        ps = []
        for i in range(world_size):
            p = ctx.Process(target=fn, args=(i, input_shapes, out_shape, world_size, init_pg, c2p, p2c))
            p.start()
            ps.append(p)
        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertEqual(output, expected[rank],
                             ("rank {} Expect receive tensor {} but got {}.").format(rank, expected[rank], output))
        for _ in range(world_size):
            p2c.put(0)
        for p in ps:
            p.join()


class HcclReduceScatterTest(HcclReduceScatterTestBase):

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_reduce_scatter(cls, rank, input_list, world_size, init_pg, c2p, p2c, reduce_op=dist.ReduceOp.SUM):
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        output = torch.empty_like(input_list_npu[rank])
        pg.reduce_scatter(output, input_list_npu, reduce_op)
        c2p.put((rank, output.cpu()))
        pg.barrier()
        p2c.get()

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    # input_shapes/out_shape are built per-rank with deterministic values so
    # the expected result can be computed locally.
    def _test_reduce_scatter_lifted(cls, rank, input_shapes, out_shape, world_size, init_pg, c2p, p2c,
                                    reduce_op=dist.ReduceOp.SUM):
        pg = init_pg(rank, world_size)
        input_list_npu = []
        offset = 0
        for s in input_shapes:
            n = 1
            for d in s:
                n *= d
            vals = torch.arange(offset, offset + n, dtype=torch.float32) + rank * 10000.0
            input_list_npu.append(vals.reshape(s).npu())
            offset += n
        output = torch.zeros(out_shape, dtype=torch.float32).npu()
        pg.reduce_scatter(output, input_list_npu, reduce_op)
        c2p.put((rank, output.cpu()))
        pg.barrier()
        p2c.get()

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_reduce_scatter_with_input_internal_format_and_offset(cls, rank, input_list, world_size, init_pg):
        torch_npu.npu.config.allow_internal_format = True
        pg = init_pg(rank, world_size)
        input_list_npu = []
        for inp in input_list:
            first_dim = inp.shape[0]
            other_dims = inp.shape[1:]
            inp = torch_npu.npu_format_cast(inp.repeat(2, *[1 for i in other_dims]).npu(), 29)[first_dim:]
            input_list_npu.append(inp)
        output = torch.empty_like(input_list_npu[rank])
        test_case = TestCase()
        error_expect = "For a tensor of internal format, it's storage_offset must be 0"
        with test_case.assertRaisesRegex(RuntimeError, error_expect):
            pg.reduce_scatter(output, input_list_npu)

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_reduce_scatter_with_output_internal_format_and_offset(cls, rank, input_list, world_size, init_pg):
        torch_npu.npu.config.allow_internal_format = True
        pg = init_pg(rank, world_size)
        input_list_npu = [input.npu() for input in input_list]
        output = torch.empty_like(input_list_npu[rank])
        first_dim = output.shape[0]
        other_dims = output.shape[1:]
        output = torch_npu.npu_format_cast(output.repeat(2, *[1 for i in other_dims]), 29)[first_dim:]
        test_case = TestCase()
        error_expect = "For a tensor of internal format, it's storage_offset must be 0"
        with test_case.assertRaisesRegex(RuntimeError, error_expect):
            pg.reduce_scatter(output, input_list_npu)

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93'])
    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16]
        format_list = [0, 2, 3, 29]
        shape_format = [
            [i, j, [4, 9]] for i in dtype_list for j in format_list] + \
            [[i, j, [8]] for i in dtype_list for j in format_list]

        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter,
                                        HcclReduceScatterTest._init_dist_hccl, expected, input_list, world_size)

    @SupportedDevices(['Ascend910A', 'Ascend910B', 'Ascend910_93'])
    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_with_different_shape(self):
        ranks = [2]
        format_list = [0, 2, 3, 29]
        dtype_list = [np.int32, np.int8]

        def get_random_input(dim=1, max_value=10, dtype=np.float32):
            shape_list = list()
            for _ in range(dim):
                shape_list.append(randint(1, max_value))
            if dtype == dtype_list[-1]:
                return create_common_tensor([dtype, format_list[0], shape_list], -10, 10)
            else:
                return create_common_tensor([dtype, format_list[randint(0, 3)], shape_list], -10, 10)

        for world_size in ranks:
            for input_dtype in dtype_list:
                input_list = list()
                for _ in range(world_size):
                    _, npu_input = get_random_input(randint(1, 5), randint(1, 10), input_dtype)
                    input_list.append(npu_input.cpu())
                cpu_excepted_result = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter,
                                        HcclReduceScatterTest._init_dist_hccl, cpu_excepted_result, input_list, world_size)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_single_tensor(self):
        # Single-tensor input list (length 1 != world_size). A single long tensor
        # of length world_size*out_numel is split evenly across ranks.
        ranks = [2]
        for world_size in ranks:
            out_shape = [4]
            input_shapes = [[world_size * 4]]  # one tensor, len == world_size*out_numel
            self._test_multiprocess_lifted(HcclReduceScatterTest._test_reduce_scatter_lifted,
                                           HcclReduceScatterTest._init_dist_hccl, input_shapes, out_shape, world_size)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_input_list_not_equal_world_size(self):
        # input_list length differs from world_size: N < ws (zero pad) and N > ws (tail ignore).
        ranks = [2]
        for world_size in ranks:
            out_shape = [4]
            # N=1 < ws: have=4 < need=8, trailing rank gets zero pad
            self._test_multiprocess_lifted(HcclReduceScatterTest._test_reduce_scatter_lifted,
                                           HcclReduceScatterTest._init_dist_hccl, [[4]], out_shape, world_size)
            # N=3 > ws: have=12 > need=8, the 3rd tensor is ignored
            self._test_multiprocess_lifted(HcclReduceScatterTest._test_reduce_scatter_lifted,
                                           HcclReduceScatterTest._init_dist_hccl, [[4]] * 3, out_shape, world_size)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_input_numel_not_equal_output(self):
        # N == world_size but per-tensor numel != output numel: have > need (tail ignore).
        ranks = [2]
        for world_size in ranks:
            out_shape = [4]
            # each tensor has 6 elements, output 4: have=12 > need=8, tail 2 elements per tensor ignored
            input_shapes = [[6]] * world_size
            self._test_multiprocess_lifted(HcclReduceScatterTest._test_reduce_scatter_lifted,
                                           HcclReduceScatterTest._init_dist_hccl, input_shapes, out_shape, world_size)

    # Ascend950 (Atlas A5) extends HCCL data type support with uint64/fp64.
    @SupportedDevices(["Ascend950"])
    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_uint64(self):
        ranks = [2]
        shape_format = [[np.uint64, 2, [4, 9]]]
        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    # uint64 is unsigned, use a non-negative range to avoid wrap-around on cast.
                    _, input1 = create_common_tensor(shape, 0, 10)
                    input_list.append(input1.cpu())
                # _construct_excepted_result uses input.cpu()*world_size (mul,
                # which works for uint64), so pass tensors directly. Only the
                # final comparison needs torch.equal (assertEqual does a-b which
                # uint64 lacks), via use_equal=True.
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter,
                                        HcclReduceScatterTest._init_dist_hccl, expected, input_list, world_size,
                                        use_equal=True)

    @SupportedDevices(["Ascend950"])
    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_fp64(self):
        ranks = [2]
        shape_format = [[np.float64, 2, [4, 9]]]
        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter,
                                        HcclReduceScatterTest._init_dist_hccl, expected, input_list, world_size)


    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_avg(self):
        ranks = [2]
        dtype_list = [np.int32, np.int8]
        shape_format = [[i, 2, [4, 9]] for i in dtype_list]

        for world_size in ranks:
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter, dist.ReduceOp.AVG)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter,
                                        HcclReduceScatterTest._init_dist_hccl, expected, input_list, world_size, dist.ReduceOp.AVG)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_pre_mul(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16]
        shape_format = [[i, 2, [4, 9]] for i in dtype_list]

        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                expected = self._construct_excepted_result(input_list, world_size, dist.reduce_scatter, dist.ReduceOp.SUM)
                expected = [i * 2 for i in expected]
                reduce_op = torch_npu.distributed._make_hccl_premul_sum(2.0)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter,
                                        HcclReduceScatterTest._init_dist_hccl, expected, input_list, world_size, reduce_op)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_with_input_internal_format_and_offset(self):
        ranks = [2]
        shape_format = [[np.float32, 2, [31, 31]]]

        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                self._test_multiprocess_with_error(HcclReduceScatterTest._test_reduce_scatter_with_input_internal_format_and_offset,
                                                   HcclReduceScatterTest._init_dist_hccl, input_list, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_with_output_internal_format_and_offset(self):
        ranks = [2]
        shape_format = [[np.float32, 2, [31, 31]]]

        for world_size in ranks:
            for shape in shape_format:
                input_list = []
                for _ in range(world_size):
                    _, input1 = create_common_tensor(shape, -10, 10)
                    input_list.append(input1.cpu())
                self._test_multiprocess_with_error(HcclReduceScatterTest._test_reduce_scatter_with_output_internal_format_and_offset,
                                                   HcclReduceScatterTest._init_dist_hccl, input_list, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_scatter_with_different_shape_avg(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16]

        def get_random_input(dim=1, max_value=10, dtype=np.float32):
            shape_list = list()
            for _ in range(dim):
                shape_list.append(randint(1, max_value))
            if dtype == dtype_list[-1]:
                return create_common_tensor([dtype, 0, shape_list], -10, 10)
            else:
                return create_common_tensor([dtype, 2, shape_list], -10, 10)

        for world_size in ranks:
            for input_dtype in dtype_list:
                input_list = list()
                for _ in range(world_size):
                    _, npu_input = get_random_input(randint(1, 5), randint(1, 10), input_dtype)
                    input_list.append(npu_input.cpu())
                cpu_excepted_result = self._construct_excepted_result(
                    input_list, world_size, dist.reduce_scatter, dist.ReduceOp.AVG)
                self._test_multiprocess(HcclReduceScatterTest._test_reduce_scatter,
                                        HcclReduceScatterTest._init_dist_hccl, cpu_excepted_result,
                                        input_list, world_size, dist.ReduceOp.AVG)


if __name__ == '__main__':
    run_tests()
