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
Add validation cases for torch._C._distributed_rpc APIs on NPU:
1. PyTorch community lacks direct and sufficient validation for some APIs,
   so this file is added.
2. This file validates torch._C._distributed_rpc._is_current_rpc_agent_set (extendable).
"""

import torch  # noqa: F401
from torch_npu.testing.testcase import run_tests, TestCase


class TestIsCurrentRpcAgentSet(TestCase):
    """Test cases for torch._C._distributed_rpc._is_current_rpc_agent_set."""

    def test_is_current_rpc_agent_set_import(self):
        """Verify that _is_current_rpc_agent_set is importable and callable."""
        from torch._C._distributed_rpc import _is_current_rpc_agent_set
        self.assertTrue(callable(_is_current_rpc_agent_set))

    def test_is_current_rpc_agent_set_default(self):
        """Verify that _is_current_rpc_agent_set returns False when RPC is not initialized."""
        from torch._C._distributed_rpc import _is_current_rpc_agent_set
        self.assertFalse(_is_current_rpc_agent_set())

    def test_is_current_rpc_agent_set_after_init(self):
        """Verify that _is_current_rpc_agent_set returns True after init_rpc, and False after shutdown."""
        import os
        import torch.distributed.rpc as rpc
        from torch._C._distributed_rpc import _is_current_rpc_agent_set

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

        self.assertFalse(_is_current_rpc_agent_set())
        rpc.init_rpc("worker0", rank=0, world_size=1)
        try:
            self.assertTrue(_is_current_rpc_agent_set())
        finally:
            rpc.shutdown()
        self.assertFalse(_is_current_rpc_agent_set())


if __name__ == "__main__":
    run_tests()
