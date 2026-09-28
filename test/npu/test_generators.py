# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add NPU coverage for torch.Generator APIs:
1. PyTorch community lacks sufficient and direct NPU validations for
   torch.Generator.initial_seed, so this file is added.
2. This file validates torch.Generator.initial_seed() / seed() /
   manual_seed() on NPU (extendable).
"""

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

device = 'npu:0'
torch.npu.set_device(device)


def get_npu_type(type_name):
    if isinstance(type_name, type):
        type_name = '{}.{}'.format(type_name.__module__, type_name.__name__)
    module, name = type_name.rsplit('.', 1)
    assert module == 'torch'
    return getattr(torch.npu, name)


class TestGenerators(TestCase):
    def test_generator(self):
        g_npu = torch.Generator(device=device)
        print(g_npu.device)
        self.assertExpectedInline(str(g_npu.device), '''npu:0''')

    def test_default_generator(self):
        output = torch.default_generator
        print(output)

    def test_generator_initial_seed_manual_seed_values(self):
        generator = torch.Generator(device=device)
        max_int64 = 0x7FFF_FFFF_FFFF_FFFF
        min_int64 = -max_int64 - 1
        max_uint64 = 0xFFFF_FFFF_FFFF_FFFF
        # manual_seed accepts [min_int64, max_uint64]; negatives wrap into uint64
        test_cases = [
            (0, 0),
            (max_int64, max_int64),
            (max_int64 + 1, max_int64 + 1),
            (max_uint64, max_uint64),
            (-1, max_uint64),
            (min_int64, max_int64 + 1),
        ]

        for seed, expected_seed in test_cases:
            with self.subTest(seed=seed):
                generator.manual_seed(seed)
                self.assertEqual(generator.initial_seed(), expected_seed)

    def test_generator_initial_seed_after_random_generation(self):
        generator = torch.Generator(device=device)
        expected_seed = 12345
        generator.manual_seed(expected_seed)

        torch.rand(8, device=device, generator=generator)

        self.assertEqual(generator.initial_seed(), expected_seed)

    def test_generator_initial_seed_after_seed(self):
        generator = torch.Generator(device=device)

        seed = generator.seed()

        self.assertEqual(generator.initial_seed(), seed)


if __name__ == "__main__":
    run_tests()
