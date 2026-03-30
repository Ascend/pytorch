"""
Add validation cases for torch.distributed APIs on NPU:
1. test/distributed/test_store.py from PyTorch community lacks sufficient API validations, so this file is added.
2. This file validates 
torch.distributed.FileStore.path
torch.distributed.Store.__init__
torch.distributed.Store.add
torch.distributed.Store.timeout
torch.distributed.TCPStore.host
torch.distributed.TCPStore.port
(extendable).
"""

import os
import time
import socket
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class TestStoreAPIs(TestCase):
    """Test cases for specific behavioral validations of Store APIs."""

    def test_filestore_path_behavior(self):
        """Test if FileStore actually uses the specified path for data exchange."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "npu_filestore.txt")
            
            store_master = dist.FileStore(filename, 2)
            self.assertEqual(store_master.path, filename)
            store_master.set("shared_key", "npu_data")

            store_worker = dist.FileStore(filename, 2)
            val = store_worker.get("shared_key")
            
            self.assertEqual(val, b"npu_data")

    def test_store_init(self):
        """Test the initialization (super().__init__) of the base Store class."""
        class MyCustomStore(dist.Store):
            def __init__(self):
                super().__init__()
                self.is_initialized = True

        custom_store = MyCustomStore()
        self.assertTrue(custom_store.is_initialized)
        self.assertIsInstance(custom_store, dist.Store)

    def test_store_add_behavior(self):
        """Test the add operation mathematically on a HashStore."""
        store = dist.HashStore()
        key = "test_add_key"
        
        res1 = store.add(key, 5)
        self.assertEqual(res1, 5)
        
        res2 = store.add(key, 10)
        self.assertEqual(res2, 15)
        
        self.assertEqual(store.get(key), b"15")
        
        res3 = store.add(key, -3)
        self.assertEqual(res3, 12)

    def test_store_timeout_behavior(self):
        """Test if the timeout property actively interrupts blocking operations."""
        store = dist.HashStore()
        
        timeout_seconds = 1
        test_timeout = timedelta(seconds=timeout_seconds)
        store.set_timeout(test_timeout)
        self.assertEqual(store.timeout, test_timeout)

        start_time = time.time()
        with self.assertRaises(RuntimeError) as context:
            store.wait(["non_existent_key"], test_timeout)
        elapsed_time = time.time() - start_time

        self.assertTrue(
            "Timeout" in str(context.exception) or "Wait timeout" in str(context.exception),
            f"Exception message does not indicate timeout: {context.exception}"
        )
        
        self.assertTrue(
            0.8 <= elapsed_time <= 2.5,
            f"Actual wait time {elapsed_time:.2f}s did not respect the {timeout_seconds}s timeout."
        )

    def test_tcpstore_host_and_port_behavior(self):
        """Test TCPStore host and port by establishing an actual Client-Server connection."""
        host = "127.0.0.1"
        port = find_free_port()
        
        server_store = dist.TCPStore(
            host_name=host,
            port=port,
            world_size=2,
            is_master=True,
            timeout=timedelta(seconds=5),
            wait_for_workers=False  
        )
        
        self.assertEqual(server_store.host, host)
        self.assertEqual(server_store.port, port)
        
        server_store.set("tcp_key", "tcp_value")

        client_store = dist.TCPStore(
            host_name=host,
            port=port,
            world_size=2,
            is_master=False,
            timeout=timedelta(seconds=5)
        )
        
        client_store.wait(["tcp_key"], timedelta(seconds=5))
        val = client_store.get("tcp_key")
        self.assertEqual(val, b"tcp_value")

        del client_store
        del server_store


if __name__ == "__main__":
    run_tests()
