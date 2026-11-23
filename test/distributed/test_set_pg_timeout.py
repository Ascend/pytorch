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
Validate torch.distributed.distributed_c10d._set_pg_timeout on NPU.

Cover group selection, timeout updates, invalid arguments, and HCCL group
isolation. Extend this file with further process-group timeout contracts.
"""

import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestSetPgTimeout(TestCase):
    def setUp(self):
        super().setUp()
        torch.npu.set_device(0)
        rendezvous = tempfile.TemporaryDirectory()
        self.addCleanup(rendezvous.cleanup)
        self.initial_timeout = timedelta(seconds=60)
        dist.init_process_group(
            "hccl",
            init_method=f"file://{rendezvous.name}/store",
            rank=0,
            world_size=1,
            timeout=self.initial_timeout,
        )
        self.addCleanup(dist.destroy_process_group)
        self.group = c10d._get_default_group()
        self.assertEqual(self._timeout(self.group), self.initial_timeout)

    def _timeout(self, group):
        return group._get_backend(torch.device(device_type)).options._timeout

    def _assert_collective(self, group):
        value = torch.ones(1).to(device_type)
        dist.all_reduce(value, group=group)
        self.assertEqual(value.cpu(), torch.ones(1))

    def test_explicit_group(self):
        group = dist.new_group([0], backend="hccl", timeout=timedelta(seconds=73))
        self.addCleanup(dist.destroy_process_group, group)
        self._assert_collective(group)
        timeout = timedelta(seconds=47)

        result = c10d._set_pg_timeout(timeout, group)

        self.assertIsNone(result)
        self.assertEqual(self._timeout(group), timeout)
        self.assertEqual(self._timeout(self.group), self.initial_timeout)
        self._assert_collective(group)

    def test_default_group_omitted(self):
        timeout = timedelta(seconds=43)
        result = c10d._set_pg_timeout(timeout)
        self.assertIsNone(result)
        self.assertEqual(self._timeout(self.group), timeout)

    def test_default_group_none(self):
        timeout = timedelta(seconds=41)
        result = c10d._set_pg_timeout(timeout, None)
        self.assertIsNone(result)
        self.assertEqual(self._timeout(self.group), timeout)

    def test_repeated_updates(self):
        for timeout in (timedelta(milliseconds=1500), timedelta(seconds=29)):
            with self.subTest(timeout=timeout):
                self.assertIsNone(c10d._set_pg_timeout(timeout, self.group))
                self.assertEqual(self._timeout(self.group), timeout)

    def test_non_member_group(self):
        with self.assertRaisesRegex(ValueError, "Invalid process group"):
            c10d._set_pg_timeout(timedelta(seconds=41), c10d.GroupMember.NON_GROUP_MEMBER)
        self.assertEqual(self._timeout(self.group), self.initial_timeout)

    def test_invalid_group_type(self):
        with self.assertRaises(AssertionError):
            c10d._set_pg_timeout(timedelta(seconds=41), object())
        self.assertEqual(self._timeout(self.group), self.initial_timeout)

    def test_invalid_timeout_type(self):
        for timeout in (None, "47", 47):
            with self.subTest(timeout=timeout):
                with self.assertRaises(TypeError):
                    c10d._set_pg_timeout(timeout, self.group)
                self.assertEqual(self._timeout(self.group), self.initial_timeout)

    def test_deprecated_alias(self):
        timeout = timedelta(seconds=49)
        with self.assertWarnsRegex(FutureWarning, "deprecated"):
            result = c10d._set_pg_timeout(timeout, self.group)
        self.assertIsNone(result)
        self.assertEqual(self._timeout(self.group), timeout)


if __name__ == "__main__":
    run_tests()
