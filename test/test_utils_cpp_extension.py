# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.utils.cpp_extension APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.utils.cpp_extension.BuildExtension (extendable).
"""

from setuptools import Distribution
from setuptools.command.build_ext import build_ext

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.cpp_extension import BuildExtension, is_ninja_available


class TestBuildExtension(TestCase):

    def test_default_options(self):
        command = BuildExtension(Distribution())
        self.assertIsInstance(command, build_ext)
        self.assertEqual(command.use_ninja, is_ninja_available())
        self.assertFalse(command.no_python_abi_suffix)

    def test_optional_options(self):
        self.assertFalse(BuildExtension(Distribution(), use_ninja=False).use_ninja)
        self.assertEqual(
            BuildExtension(Distribution(), use_ninja=True).use_ninja, is_ninja_available()
        )
        self.assertTrue(
            BuildExtension(Distribution(), no_python_abi_suffix=True).no_python_abi_suffix
        )

    def test_with_options_presets_options(self):
        subclass = BuildExtension.with_options(use_ninja=False, no_python_abi_suffix=True)
        self.assertTrue(issubclass(subclass, BuildExtension))
        command = subclass(Distribution())
        self.assertFalse(command.use_ninja)
        self.assertTrue(command.no_python_abi_suffix)

    def test_with_options_without_options(self):
        command = BuildExtension.with_options()(Distribution())
        self.assertEqual(command.use_ninja, is_ninja_available())
        self.assertFalse(command.no_python_abi_suffix)

    def test_with_options_overrides_constructor_options(self):
        command = BuildExtension.with_options(use_ninja=False)(Distribution(), use_ninja=True)
        self.assertFalse(command.use_ninja)

    def test_finalize_options_forces_rebuild_with_ninja(self):
        command = BuildExtension(Distribution())
        command.use_ninja = True
        command.finalize_options()
        self.assertTrue(command.force)

        command = BuildExtension(Distribution(), use_ninja=False)
        command.finalize_options()
        self.assertFalse(command.force)

    def test_get_ext_filename_abi_suffix(self):
        command = BuildExtension(Distribution())
        self.assertIn("cpython", command.get_ext_filename("dummy"))

        command = BuildExtension.with_options(no_python_abi_suffix=True)(Distribution())
        self.assertNotIn("cpython", command.get_ext_filename("dummy"))
        self.assertTrue(command.get_ext_filename("dummy").endswith(".so"))


if __name__ == "__main__":
    run_tests()
