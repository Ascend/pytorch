import unittest

from torch_npu._inductor.codegen.scheduling import (
    MAX_TRITON_KERNEL_NAME_LENGTH,
    _make_triton_kernel_name,
)


class TestTritonKernelName(unittest.TestCase):
    def test_two_digit_suffix_preserves_fused_name(self):
        fused_name = "f" * 35

        kernel_name = _make_triton_kernel_name("poi", fused_name, "99")

        self.assertEqual(kernel_name, f"triton_poi_{fused_name}_99")
        self.assertEqual(len(kernel_name), MAX_TRITON_KERNEL_NAME_LENGTH)

    def test_three_digit_suffix_keeps_triton_prefix(self):
        kernel_name = _make_triton_kernel_name(
            "poi", "fused_arange_constant_pad_nd_index_", "235"
        )

        self.assertEqual(
            kernel_name, "triton_poi_fused_arange_constant_pad_nd_index_235"
        )
        self.assertTrue(kernel_name.startswith("triton_"))
        self.assertEqual(len(kernel_name), MAX_TRITON_KERNEL_NAME_LENGTH)

    def test_four_digit_suffix_shortens_fused_name(self):
        kernel_name = _make_triton_kernel_name("tem", "f" * 35, "1000")

        self.assertEqual(kernel_name, f"triton_tem_{'f' * 33}_1000")
        self.assertEqual(len(kernel_name), MAX_TRITON_KERNEL_NAME_LENGTH)

    def test_short_fused_name_is_unchanged(self):
        kernel_name = _make_triton_kernel_name("poi", "fused_add", "123")

        self.assertEqual(kernel_name, "triton_poi_fused_add_123")


if __name__ == "__main__":
    unittest.main()
