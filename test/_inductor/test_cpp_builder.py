from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor.cpp_builder import get_cpp_torch_device_options


class TestCppBuilder(TestUtils):
    def test_cpp_builder(self):
        (
            definations,
            include_dirs,
            cflags, 
            ldflags, 
            libraries_dirs, 
            libraries, 
            passthough_args,
        ) = get_cpp_torch_device_options(
            device_type="npu",
            aot_mode=False,
            compile_only=False
        )
        # get accurate results 
        definations_acc = []
        definations_acc.append("USE_NPU")
        libraries_acc = []
        libraries_acc += ["torch_npu", "runtime", "ascendcl"]
        passthough_args_acc = []
        passthough_args_acc += ["-DBUILD_LIBTORCH=ON -Wno-unused-function"]

        # compare results
        self.assertEqual(definations_acc, definations)
        self.assertEqual(libraries_acc, libraries)
        self.assertEqual(passthough_args_acc, passthough_args)

instantiate_parametrized_tests(TestCppBuilder)

if __name__ == "__main__":
    run_tests()