# Copyright (c) 2023, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain data copy of the License at
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
import unittest

import torch

import torch_npu
import torch_npu.utils.cpp_extension as TorchNpuExtension
from torch_npu.utils.cpp_extension import  BISHENG_CPP_HOME
from torch_npu.testing.testcase import TestCase, run_tests


def remove_build_path():
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root)


class TestCppExtensionJIT(TestCase):
    """Tests just-in-time cpp extensions.
    """

    def setUp(self):
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tearDown(self):
        super().tearDown()
        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        remove_build_path()

    @classmethod
    def tearDownClass(cls):
        remove_build_path()

    def _test_jit_compile_extension_with_cpp(self):
        module = TorchNpuExtension.load(
            name="jit_extension",
            sources=[
                "cpp_extensions/jit_extension.cpp",
                "cpp_extensions/jit_exp_add.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True)
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

        # Checking we can call a method defined not in the main C++ file.
        z = module.exp_add(x, y)
        self.assertEqual(z, x.exp() + y.exp())

        npu_z = module.npu_add(x.npu(), y.npu())
        self.assertRtolEqual(npu_z.cpu(), (x + y))

        # Checking we can use this JIT-compiled class.
        doubler = module.Doubler(2, 2)
        self.assertIsNone(doubler.get().grad)
        self.assertEqual(doubler.get().sum(), 4)
        self.assertEqual(doubler.forward().sum(), 8)

    def test_jit_compile_extension_with_cpp(self):
       self._test_jit_compile_extension_with_cpp()

    @unittest.skipIf(BISHENG_CPP_HOME is None, "BISHENG_CPP_HOME nor found")
    def test_jit_bisheng_extension(self):
        module = TorchNpuExtension.load(
            name="torch_test_bisheng_extension",
            sources=[
                "cpp_extensions/bisheng_extension.cpp",
                "cpp_extensions/bisheng_add_tensor.cpp",
            ],
            verbose=True)

        x = torch.randn(4, 4).npu()
        y = torch.randn(4, 4).npu()
        z = module.bscpp_add(x, y)
        self.assertEqual(z.cpu(), (x + y).cpu())

    @unittest.skipIf(BISHENG_CPP_HOME is None, "BISHENG_CPP_HOME nor found")
    def test_jit_bisheng_extension_with_no_python_module(self):
        TorchNpuExtension.load(
            name="cpp_bisheng",
            sources=[
                "cpp_extensions/bisheng_with_torch_library.cpp",
                "cpp_extensions/bisheng_add_tensor.cpp",
            ],
            with_bishengcpp=True,
            is_python_module=False,
            verbose=True)
        x = torch.randn(4, 4).npu()
        y = torch.randn(4, 4).npu()
        z = torch.ops.cpp_bisheng.bscpp_add(x, y)
        self.assertEqual(z.cpu(), (x + y).cpu())


if __name__ == "__main__":
    run_tests()
