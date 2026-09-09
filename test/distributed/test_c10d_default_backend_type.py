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
Validate the default backend type resolution of
torch.distributed.distributed_c10d._new_process_group_helper on NPU:

1. torch_npu used to replace _new_process_group_helper with a patched copy
   adding an "HCCL -> BackendType.CUSTOM" branch to the default backend
   resolution. Upstream PyTorch (pytorch#179901) now resolves this natively,
   backed by the Backend.register_backend("hccl", ..., devices=["npu"])
   registration.
2. On torch versions without the upstream support (2.13.x, early 2.14
   ), torch_npu/_compat/distributed.py restores the removed patch at
   import time. This file validates the resolution inputs stable across torch
   2.13+, asserts the resolved CUSTOM type where upstream provides the final
   helper, asserts the compat patch is in effect on old torch (mirroring real
   user behaviour instead of skipping), and runs an end-to-end default
   process group created without an explicit backend (barrier / collectives /
   subgroups).
"""

import os
import inspect

import torch
import torch.distributed.run as launch
from torch.distributed import distributed_c10d as c10d
from torch.distributed.distributed_c10d import Backend, BackendConfig, ProcessGroup

from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


def _upstream_supports_npu_default_backend():
    """Whether upstream _new_process_group_helper natively resolves an NPU
    "undefined" backend to BackendType.CUSTOM (pytorch#179901).

    Local copy of the detection used by torch_npu/_compat/distributed.py.
    """
    src = inspect.getsource(c10d._new_process_group_helper)
    return (
        "_get_default_backend_type_for_backend_config" in src
        or "backend_type_map.get(str(backend))" in src
    )


class TestDefaultBackendType(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    @staticmethod
    def _registrable_backend_types(backend_config):
        """The BackendTypes upstream _new_process_group_helper would register."""
        return {
            Backend.backend_type_map.get(
                str(backend), ProcessGroup.BackendType.CUSTOM
            )
            for backend in backend_config.device_backend_map.values()
        }

    def test_hccl_backend_registration_on_npu(self):
        """The hccl backend registration must drive the default resolution.

        These facts are stable across torch 2.13+ and are what the upstream
        ``_new_process_group_helper`` consumes to resolve the default backend
        type for ``BackendConfig("undefined")``.
        """
        self.assertEqual(Backend.backend_type_map["hccl"],
                         ProcessGroup.BackendType.CUSTOM)
        self.assertEqual(Backend.default_device_backend_map["npu"], "hccl")
        # init_process_group() with no backend -> BackendConfig("undefined")
        backend_config = BackendConfig("undefined")
        self.assertEqual(backend_config.device_backend_map, {"npu": "hccl"})
        for backend_str in ["hccl", "npu:hccl"]:
            self.assertEqual(BackendConfig(backend_str).device_backend_map,
                             {"npu": "hccl"})

    def test_default_backend_type_resolves_to_custom(self):
        """"undefined" must resolve to the hccl/CUSTOM default backend type.

        On torch versions where upstream natively resolves it (pytorch#179901,
        torch >= 2.14.0.dev mid-July 2026) this asserts the resolved
        type directly. On older torch (2.13.x, early 2.14 nightlies) the
        torch_npu compat patch (torch_npu/_compat/distributed.py) restores the
        removed HCCL branch, and this asserts the patch is in effect -- mirror
        the real user behaviour instead of skipping.
        """
        backend_config = BackendConfig("undefined")
        self.assertEqual(backend_config.device_backend_map, {"npu": "hccl"})
        if _upstream_supports_npu_default_backend():
            # Upstream resolves it natively: assert the resolved type where
            # the final helper exists, else the resolution inputs above.
            if hasattr(c10d, "_get_default_backend_type_for_backend_config"):
                default_type = c10d._get_default_backend_type_for_backend_config(
                    backend_config)
                self.assertEqual(default_type, ProcessGroup.BackendType.CUSTOM)
                self.assertIn(default_type,
                              self._registrable_backend_types(backend_config))
                for backend_str in ["hccl", "npu:hccl"]:
                    explicit_config = BackendConfig(backend_str)
                    self.assertEqual(
                        c10d._get_default_backend_type_for_backend_config(
                            explicit_config),
                        ProcessGroup.BackendType.CUSTOM)
                    self.assertIn(
                        c10d._get_default_backend_type_for_backend_config(
                            explicit_config),
                        self._registrable_backend_types(explicit_config))
                mixed_config = BackendConfig(f"cpu:gloo,{self.device_name}:hccl")
                self.assertEqual(
                    c10d._get_default_backend_type_for_backend_config(mixed_config),
                    ProcessGroup.BackendType.CUSTOM)
                self.assertEqual(
                    c10d._get_default_backend_type_for_backend_config(
                        mixed_config, torch.device(f"{self.device_name}:0")),
                    ProcessGroup.BackendType.CUSTOM)
                self.assertEqual(
                    c10d._get_default_backend_type_for_backend_config(
                        mixed_config, torch.device("cpu")),
                    ProcessGroup.BackendType.GLOO)
        else:
            # Old torch: the compat patch must have replaced the upstream
            # helper with the implementation carrying the HCCL branch.
            src = inspect.getsource(c10d._new_process_group_helper)
            self.assertIn("Backend.HCCL", src)

    @skipIfUnsupportMultiNPU(2)
    def test_init_process_group_without_backend_multinpu(self):
        """End-to-end: no backend specified must create a working hccl group."""
        launch.main(
            [
                "--nproc-per-node=2",
                path("default_backend_type/default_backend_type_base.py"),
            ]
        )


if __name__ == "__main__":
    run_tests()
