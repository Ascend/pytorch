from types import SimpleNamespace
from unittest import mock

from torch_npu.profiler._flops_formulas import (
    _calculate_common_layout_flops,
    matmul_flops,
    npu_all_gather_base_mm_flops,
    npu_alltoallv_gmm_flops,
    npu_block_sparse_attention_flops,
    npu_fusion_attention_flops,
    npu_gmm_alltoallv_flops,
    npu_grouped_matmul_flops,
    npu_grouped_matmul_swiglu_quant_v2_flops,
    npu_quant_matmul_gelu_flops,
    npu_transpose_batchmatmul_flops,
)
from torch_npu.profiler._flops_hook import FlopsHookManager
from torch_npu.testing.testcase import run_tests, TestCase


class _Tensor:
    def __init__(self, shape):
        self.shape = shape


class _TensorWithList:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


class TestFlopsHook(TestCase):
    def tearDown(self):
        FlopsHookManager.uninstall()

    def test_fusion_attention_formula_accepts_real_positional_arguments(self):
        query = _Tensor((2, 4, 8, 16))
        key = _Tensor((2, 4, 8, 16))
        value = _Tensor((2, 4, 8, 16))
        self.assertEqual(
            2 * 2 * 4 * 8 * 8 * (16 + 16),
            npu_fusion_attention_flops(query, key, value, 4, "BNSD"),
        )

    def test_fusion_attention_formula_uses_value_dim(self):
        query = _Tensor((1, 2, 3, 4))
        key = _Tensor((1, 2, 5, 4))
        value = _Tensor((1, 2, 5, 7))
        self.assertEqual(
            2 * 1 * 2 * 3 * 5 * (4 + 7),
            npu_fusion_attention_flops(query, key, value, 2, "BNSD"),
        )

    def test_sparse_formula_uses_sequence_lengths_for_non_square_attention(self):
        self.assertEqual(
            2 * 1 * 2 * (8 * 4 - 4 * 4 / 2) * (16 + 16),
            _calculate_common_layout_flops(
                (1, 2, 8, 16),
                (1, 2, 4, 16),
                (1, 2, 4, 16),
                "BNSD",
                2,
                2,
                2,
            ),
        )

    def test_all_gather_base_mm_formula_counts_gathered_gemm_only(self):
        self.assertEqual(
            2 * (3 * 2) * 4 * 5,
            npu_all_gather_base_mm_flops(
                _Tensor((3, 4)), _Tensor((4, 5)), "hcom", 2
            ),
        )

    def test_transpose_batchmatmul_formula_counts_batch_gemm_only(self):
        self.assertEqual(
            2 * 2 * 3 * 4 * 5,
            npu_transpose_batchmatmul_flops(
                _Tensor((3, 2, 4)),
                _Tensor((2, 4, 5)),
                perm_x1=(1, 0, 2),
            ),
        )

    def test_grouped_matmul_formula_sums_group_gemms(self):
        self.assertEqual(
            2 * 2 * 3 * 7 + 2 * 4 * 5 * 11,
            npu_grouped_matmul_flops(
                [_Tensor((2, 3)), _Tensor((4, 5))],
                [_Tensor((3, 7)), _Tensor((5, 11))],
            ),
        )

    def test_grouped_matmul_formula_uses_group_list_for_split_groups(self):
        self.assertEqual(
            2 * 2 * 3 * 7 + 2 * 4 * 3 * 11,
            npu_grouped_matmul_flops(
                _Tensor((6, 3)),
                [_Tensor((3, 7)), _Tensor((3, 11))],
                group_list=[2, 6],
            ),
        )

    def test_grouped_matmul_formula_rejects_scalar_group_list(self):
        with self.assertRaisesRegex(ValueError, "returning a sequence"):
            npu_grouped_matmul_flops(
                _Tensor((6, 3)),
                [_Tensor((3, 7)), _Tensor((3, 11))],
                group_list=_TensorWithList(6),
            )

    def test_quant_matmul_gelu_formula_counts_matmul_only(self):
        self.assertEqual(
            2 * 6 * 4 * 5,
            npu_quant_matmul_gelu_flops(
                _Tensor((2, 3, 4)), _Tensor((4, 5)), _Tensor((6,)), _Tensor((5,))
            ),
        )

    def test_grouped_matmul_swiglu_quant_formula_counts_grouped_gemm_only(self):
        self.assertEqual(
            2 * 8 * 4 * 16,
            npu_grouped_matmul_swiglu_quant_v2_flops(
                _Tensor((8, 4)),
                _Tensor((2, 4, 16)),
                _Tensor((2, 16)),
                _Tensor((8,)),
                _Tensor((2,)),
            ),
        )

    def test_alltoallv_gmm_formula_adds_optional_shared_mm(self):
        self.assertEqual(
            2 * 7 * 4 * 6 + 2 * 5 * 2 * 9,
            npu_alltoallv_gmm_flops(
                _Tensor((7, 4)),
                _Tensor((3, 4, 6)),
                "hcom",
                2,
                [3, 4],
                [4, 3],
                mm_x=_Tensor((5, 2)),
                mm_weight=_Tensor((2, 9)),
            ),
        )

    def test_gmm_alltoallv_formula_counts_route_gmm_only_when_no_shared_mm(self):
        self.assertEqual(
            2 * 7 * 4 * 6,
            npu_gmm_alltoallv_flops(
                _Tensor((7, 4)),
                _Tensor((3, 4, 6)),
                "hcom",
                2,
                [3, 4],
                [4, 3],
            ),
        )

    def test_block_sparse_attention_formula_counts_valid_blocks(self):
        mask = [
            [
                [[1, 0], [0, 1], [1, 1]],
                [[0, 1], [1, 0], [0, 0]],
            ]
        ]
        self.assertEqual(
            2 * 30 * (4 + 8),
            npu_block_sparse_attention_flops(
                _Tensor((1, 2, 5, 4)),
                _Tensor((1, 2, 6, 4)),
                _Tensor((1, 2, 6, 8)),
                mask,
                (2, 3),
                q_input_layout="BNSD",
                kv_input_layout="BNSD",
            ),
        )

    def test_block_sparse_attention_formula_keeps_bnsd_lengths_per_batch(self):
        mask = [
            [[[1, 0], [0, 1], [1, 1]]],
            [[[1, 1], [0, 0], [1, 0]]],
        ]
        self.assertEqual(
            2 * 21 * (4 + 4),
            npu_block_sparse_attention_flops(
                _Tensor((2, 1, 5, 4)),
                _Tensor((2, 1, 5, 4)),
                _Tensor((2, 1, 5, 4)),
                mask,
                (2, 3),
                q_input_layout="BNSD",
                kv_input_layout="BNSD",
                actual_seq_lengths=[3, 5],
                actual_seq_lengths_kv=[5, 5],
            ),
        )

    def test_matmul_formula_supports_broadcast_batch_dimensions(self):
        self.assertEqual(
            2 * 5 * 2 * 3 * 4 * 6,
            matmul_flops(_Tensor((5, 1, 3, 4)), _Tensor((2, 4, 6))),
        )

    def test_hook_records_flops_and_op_name_in_range_label(self):
        target = SimpleNamespace()

        def original(value):
            return value + 1

        target.op = original
        with (
            mock.patch(
                "torch_npu.profiler._flops_hook.get_flop_func",
                return_value=lambda value: value * 2,
            ),
            mock.patch(
                "torch_npu.profiler._flops_hook.mstx.range_start", return_value=7
            ) as mock_start,
            mock.patch("torch_npu.profiler._flops_hook.mstx.range_end") as mock_end,
        ):
            FlopsHookManager.install({"demo_op": (target, "op")})
            self.assertEqual(4, target.op(3))

        mock_start.assert_called_once_with("6-demo_op", domain="mfu_flops")
        mock_end.assert_called_once_with(7, domain="mfu_flops")


if __name__ == "__main__":
    run_tests()
