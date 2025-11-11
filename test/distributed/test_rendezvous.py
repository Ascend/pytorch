import os
import logging
from datetime import timedelta
from unittest.mock import patch

from torch.distributed.rendezvous import register_rendezvous_handler as register_rendezvous_handler
from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT
from torch.distributed import Store, PrefixStore
from torch.distributed.elastic.rendezvous.api import RendezvousParameters, RendezvousHandler, RendezvousInfo, RendezvousStoreInfo
from torch.distributed.elastic.rendezvous.api import rendezvous_handler_registry as handler_registry
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint
from torch_npu.distributed.run import parse_args as torch_parse_cmd_args
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.distributed import ParallelStore
from torch_npu.distributed.rendezvous import _create_c10d_store
from torch_npu.distributed.rendezvous import _rendezvous_error
from torch_npu.distributed.rendezvous import _torchelastic_use_agent_store
from torch_npu.distributed.rendezvous import _parallel_rendezvous_handler
from torch_npu.distributed.rendezvous import _create_parallel_handler
from torch_npu.distributed.rendezvous import _rendezvous_init


class TestRendezvous(TestCase):
    def test_create_c10d_store_without_agent_store(self):
        with patch.dict(os.environ, {
            "TORCH_NPU_ELASTIC_USE_AGENT_STORE": "False",
            "PROXY_AGENT_PID_USE_LOCAL_SOCKET_PATH": "12345"}):
            store = _create_c10d_store("localhost", 12345, 0, 2, timedelta(seconds=600))
            self.assertIsInstance(store, ParallelStore)

    def test_create_c10d_store_invalid_port(self):
        with self.assertRaises(ValueError) as context:
            _create_c10d_store("localhost", 70000, 0, 2, timedelta(seconds=600))
        self.assertIn("port must have value from 0 to 65535", str(context.exception))

    def test_parallel_rendezvous_handler_missing_rank(self):
        with self.assertRaises(ValueError) as context:
            list(_parallel_rendezvous_handler("parallel://localhost:12345?world_size=2"))
        self.assertIn("rank parameter missing", str(context.exception))

    def test_torchelastic_use_agent_store_true(self):
        with patch.dict(os.environ, {"TORCH_NPU_ELASTIC_USE_AGENT_STORE": "True"}):
            self.assertTrue(_torchelastic_use_agent_store())


if __name__ == '__main__':
    run_tests()