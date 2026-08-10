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
Add validation cases for torch._C._distributed_c10d.Backend.Options on NPU:

1. PyTorch community lacks sufficient and direct API validations for
   this API, so this file is added.
2. This file validates
   torch._C._distributed_c10d.Backend.Options (extendable).
"""

from datetime import timedelta

from torch._C._distributed_c10d import Backend
from torch.testing._internal.common_utils import TestCase, run_tests


class TestBackendOptions(TestCase):
    def test_init_with_default_and_explicit_timeout(self):
        default_options = Backend.Options("hccl")

        self.assertIsInstance(default_options, Backend.Options)
        self.assertEqual(default_options.backend, "hccl")
        self.assertEqual(default_options._timeout, timedelta(minutes=30))

        explicit_timeout = timedelta(seconds=7)
        keyword_options = Backend.Options(
            backend="hccl",
            timeout=explicit_timeout,
        )

        self.assertEqual(keyword_options.backend, "hccl")
        self.assertEqual(keyword_options._timeout, explicit_timeout)

        positional_options = Backend.Options(
            "gloo",
            timedelta(seconds=11),
        )

        self.assertEqual(positional_options.backend, "gloo")
        self.assertEqual(
            positional_options._timeout,
            timedelta(seconds=11),
        )

    def test_init_with_boundary_values(self):
        empty_backend_options = Backend.Options("")
        zero_timeout_options = Backend.Options(
            "hccl",
            timedelta(0),
        )
        negative_timeout_options = Backend.Options(
            "hccl",
            timedelta(seconds=-1),
        )

        self.assertEqual(empty_backend_options.backend, "")
        self.assertEqual(
            empty_backend_options._timeout,
            timedelta(minutes=30),
        )
        self.assertEqual(zero_timeout_options._timeout, timedelta(0))
        self.assertEqual(
            negative_timeout_options._timeout,
            timedelta(seconds=-1),
        )

    def test_group_metadata_properties(self):
        options = Backend.Options("hccl")

        self.assertEqual(options.global_ranks_in_group, [])
        self.assertEqual(options.group_name, "")

        options.global_ranks_in_group = [0, 2, 4]
        options.group_name = "test_group"

        self.assertEqual(
            options.global_ranks_in_group,
            [0, 2, 4],
        )
        self.assertEqual(options.group_name, "test_group")

    def test_property_access_and_instance_independence(self):
        first = Backend.Options(
            "hccl",
            timedelta(seconds=1),
        )
        second = Backend.Options(
            "gloo",
            timedelta(seconds=2),
        )

        with self.assertRaises(AttributeError):
            first.backend = "gloo"

        first._timeout = timedelta(seconds=20)

        self.assertEqual(first.backend, "hccl")
        self.assertEqual(first._timeout, timedelta(seconds=20))
        self.assertEqual(second.backend, "gloo")
        self.assertEqual(second._timeout, timedelta(seconds=2))
        self.assertIsNot(first, second)

    def test_invalid_constructor_arguments(self):
        with self.assertRaises(TypeError):
            Backend.Options()

        invalid_backends = [
            None,
            1,
            [],
            {},
        ]

        for backend in invalid_backends:
            with self.subTest(backend=backend):
                with self.assertRaises(TypeError):
                    Backend.Options(backend)

        invalid_timeouts = [
            None,
            1,
            "1 second",
            [],
            {},
        ]

        for timeout in invalid_timeouts:
            with self.subTest(timeout=timeout):
                with self.assertRaises(TypeError):
                    Backend.Options("hccl", timeout)

        with self.assertRaises(TypeError):
            Backend.Options(
                "hccl",
                timedelta(seconds=1),
                "unexpected",
            )

        with self.assertRaises(TypeError):
            Backend.Options(
                "hccl",
                unexpected=True,
            )

    def test_invalid_timeout_assignment(self):
        options = Backend.Options("hccl")

        invalid_timeouts = [
            None,
            1,
            "1 second",
            [],
            {},
        ]

        for timeout in invalid_timeouts:
            with self.subTest(timeout=timeout):
                with self.assertRaises(TypeError):
                    options._timeout = timeout

        self.assertEqual(options._timeout, timedelta(minutes=30))


if __name__ == "__main__":
    run_tests()
