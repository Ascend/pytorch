from unittest import TestCase, main, mock

from torch_npu._compat.inductor import get_cpp_torch_device_options
from torch_npu._inductor import cpp_builder as npu_cpp_builder


class TestCppBuilder(TestCase):
    def test_legacy_patch_entry_point(self):
        with mock.patch(
            "torch_npu._compat.inductor.patch_npu_cpp_device_options"
        ) as patch_npu_cpp_device_options:
            npu_cpp_builder.patch_get_cpp_torch_device_options()

        patch_npu_cpp_device_options.assert_called_once_with()

    def test_npu_cpp_options(self):
        with mock.patch.object(
            npu_cpp_builder, "include_paths", return_value=["/mock/include"]
        ), mock.patch.object(
            npu_cpp_builder, "library_paths", return_value=["/mock/lib"]
        ):
            result = get_cpp_torch_device_options("npu", False, False)

        self.assertEqual(result[0], ["USE_NPU"])
        self.assertEqual(result[1], ["/mock/include"])
        self.assertEqual(result[4], ["/mock/lib"])
        self.assertEqual(result[5], ["torch_npu", "runtime", "ascendcl"])
        self.assertEqual(result[6], ["-DBUILD_LIBTORCH=ON -Wno-unused-function"])


if __name__ == "__main__":
    main()
