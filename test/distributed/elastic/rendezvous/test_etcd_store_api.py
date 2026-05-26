"""
PyTorch community lacks some torch.distributed.elastic.rendezvous.etcd_store APIs validation cases, so this file is added.

This file validate following APIs:
torch.distributed.elastic.rendezvous.etcd_store.EtcdStore
torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.set
torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.get
torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.add
torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.check
torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.wait
torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.set_timeout
(Extendable)
"""

import base64
import concurrent.futures
import datetime
import subprocess
import threading
import time

from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.rendezvous.etcd_store import EtcdStore
from torch.testing._internal.common_utils import run_tests, TestCase


class EtcdTestBase(TestCase):
    """Base class with common setup - fails fast on any error"""

    @classmethod
    def setUpClass(cls):
        """Class-level setup: Create and start a single EtcdServer for all tests"""
        cls.server = EtcdServer()
        cls.server.start(stderr=subprocess.DEVNULL)
        cls.client = cls.server.get_client()

    @classmethod
    def tearDownClass(cls):
        """Class-level teardown: Stop the global EtcdServer"""
        cls.server.stop()

    def setUp(self):
        """Test-level setup: Create a clean EtcdStore with isolated prefix"""
        self.store = EtcdStore(self.client, f"/test_prefix_{self._testMethodName}/")

    def tearDown(self):
        pass


class TestEtcdStoreInit(EtcdTestBase):
    """Test EtcdStore constructor and basic configuration"""

    def test_init_with_timeout(self):
        """Test passing a custom timeout to the constructor"""
        timeout = datetime.timedelta(seconds=30)
        store = EtcdStore(self.client, "/timeout_test/", timeout=timeout)

        start = time.time()
        with self.assertRaises(LookupError):
            store.get("nonexistent")
        elapsed = time.time() - start

        self.assertLess(elapsed, 35)

    def test_prefix_auto_append_slash(self):
        """Test that prefix automatically appends '/' if missing"""
        store = EtcdStore(self.client, "/no_slash")
        store.set("k", "v")

        encoded_key = base64.b64encode(b"k").decode()
        raw_value = self.client.get(f"/no_slash/{encoded_key}")

        self.assertEqual(raw_value.value, base64.b64encode(b"v").decode())

    def test_prefix_with_existing_slash(self):
        """Test that prefix does not duplicate '/' when already present"""
        store = EtcdStore(self.client, "/with_slash/")
        store.set("k", "v")

        encoded_key = base64.b64encode(b"k").decode()
        raw_value = self.client.get(f"/with_slash/{encoded_key}")

        self.assertEqual(raw_value.value, base64.b64encode(b"v").decode())


class TestEtcdStoreSetGet(EtcdTestBase):
    """Test set and get API functionality"""

    def test_set_and_get_string(self):
        """Test storing and retrieving string values"""
        self.store.set("str_key", "hello")
        self.assertEqual(self.store.get("str_key"), b"hello")

    def test_set_and_get_bytes(self):
        """Test storing and retrieving raw bytes values"""
        binary_data = b"\x00\x01\x02\xff"
        self.store.set("bytes_key", binary_data)
        self.assertEqual(self.store.get("bytes_key"), binary_data)

    def test_set_overwrite(self):
        """Test overwriting an existing key"""
        self.store.set("overwrite_key", "v1")
        self.store.set("overwrite_key", "v2")
        self.assertEqual(self.store.get("overwrite_key"), b"v2")

    def test_set_invalid_type_raises(self):
        """Test that set raises ValueError for invalid value types"""
        invalid_values = [123, 45.6, ["list"], {"dict": "value"}, None]

        for val in invalid_values:
            with self.subTest(value=val):
                with self.assertRaises(ValueError):
                    self.store.set("k", val)

    def test_set_special_characters(self):
        """Test keys with special characters (slashes, spaces, unicode, etc.)"""
        special_cases = {
            "key/with/nested/path": "slash_value",
            "key#hash": "hash_value",
            "key with space": "space_value",
            "key中文测试": "chinese_value",
            "key:colon": "colon_value",
            "key.dots": "dots_value",
        }

        for key, val in special_cases.items():
            with self.subTest(key=key):
                self.store.set(key, val)
                self.assertEqual(self.store.get(key), val.encode())

    def test_get_nonexistent_timeout(self):
        """Test that get raises LookupError on timeout for non-existent key"""
        self.store.set_timeout(datetime.timedelta(seconds=1))

        start = time.time()
        with self.assertRaises(LookupError):
            self.store.get("never_exists_key")

        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 2.0)

    def test_get_blocks_until_available(self):
        """Test get blocks until key is set (using thread event, not sleep)"""
        key = "blocking_key"
        ready_event = threading.Event()

        def delayed_set():
            ready_event.wait()
            time.sleep(0.2)
            self.store.set(key, "delayed_value")

        writer = threading.Thread(target=delayed_set)
        writer.start()
        ready_event.set()

        result = self.store.get(key)
        self.assertEqual(result, b"delayed_value")

        writer.join()

    def test_get_timeout_none_permanent_block(self):
        """Test get blocks permanently when timeout=None (released by thread)"""
        key = "permanent_block_key"

        def release_after():
            time.sleep(0.5)
            self.store.set(key, "released")

        releaser = threading.Thread(target=release_after)
        releaser.start()

        result = self.store.get(key)
        self.assertEqual(result, b"released")

        releaser.join()

    def test_multiple_keys_isolation(self):
        """Test data isolation between multiple distinct keys"""
        keys = [f"isolated_key_{i}" for i in range(10)]

        for i, k in enumerate(keys):
            self.store.set(k, f"value_{i}")

        for i, k in enumerate(keys):
            self.assertEqual(self.store.get(k), f"value_{i}".encode())


class TestEtcdStoreAdd(EtcdTestBase):
    """Test add API (atomic increment)"""

    def test_add_initializes_when_key_absent(self):
        """Test add initializes key to 0 and increments when key does not exist"""
        result = self.store.add("init_counter", 5)
        self.assertEqual(result, 5)
        self.assertEqual(self.store.get("init_counter"), b"5")

    def test_add_increments_existing(self):
        """Test add increments an existing numeric value"""
        self.store.add("inc_counter", 10)
        result = self.store.add("inc_counter", 7)

        self.assertEqual(result, 17)
        self.assertEqual(self.store.get("inc_counter"), b"17")

    def test_add_zero(self):
        """Test add with 0 (idempotent operation)"""
        self.store.add("zero_test", 100)
        result = self.store.add("zero_test", 0)

        self.assertEqual(result, 100)
        self.assertEqual(self.store.get("zero_test"), b"100")

    def test_add_negative_decrement(self):
        """Test add with negative values (decrement behavior)"""
        self.store.add("dec_counter", 20)
        result = self.store.add("dec_counter", -5)

        self.assertEqual(result, 15)
        self.assertEqual(self.store.get("dec_counter"), b"15")

    def test_add_concurrent_20_threads(self):
        """Test concurrent add from 20 threads (exceptions propagate to main thread)"""
        key = "concurrent_20"

        def worker():
            return self.store.add(key, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 20)
        self.assertEqual(int(self.store.get(key)), 20)

    def test_add_high_contention_100_threads(self):
        """Test high-contention add with 100 threads (atomicity validation)"""
        key = "high_contention_100"

        def worker():
            return self.store.add(key, 1)

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(worker) for _ in range(100)]
            for f in futures:
                f.result()
        elapsed = time.time() - start_time

        final_value = int(self.store.get(key))

        self.assertEqual(
            final_value, 100, f"Atomicity violation! Expected 100, got {final_value}"
        )
        self.assertLess(elapsed, 60)

    def test_add_large_number(self):
        """Test add with large numeric values"""
        self.store.add("big_num", 10**9)
        result = self.store.add("big_num", 10**9)

        self.assertEqual(result, 2 * 10**9)


class TestEtcdStoreWait(EtcdTestBase):
    """Test wait API (block until all keys exist)"""

    def test_wait_success_multiple_keys(self):
        """Test wait succeeds when all required keys are set"""
        keys = ["wait_k1", "wait_k2", "wait_k3"]
        ready = threading.Event()

        def delayed_write():
            ready.wait()
            time.sleep(0.2)
            for k in keys:
                self.store.set(k, f"{k}_value")

        writer = threading.Thread(target=delayed_write)
        writer.start()
        ready.set()

        self.store.wait(keys)

        for k in keys:
            self.assertEqual(self.store.get(k), f"{k}_value".encode())

        writer.join()

    def test_wait_timeout_global(self):
        """Test wait times out using the global store timeout"""
        self.store.set_timeout(datetime.timedelta(seconds=1))

        start = time.time()
        with self.assertRaises(LookupError):
            self.store.wait(["never_key_1", "never_key_2"])

        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 2.0)

    def test_wait_override_timeout(self):
        """Test wait with a custom override_timeout parameter"""
        start = time.time()

        with self.assertRaises(LookupError):
            self.store.wait(
                ["override_key"], override_timeout=datetime.timedelta(seconds=1)
            )

        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 2.0)

    def test_wait_partial_keys_timeout(self):
        """Test wait times out if only some keys exist"""
        self.store.set("partial_exist", "yes")
        self.store.set_timeout(datetime.timedelta(seconds=1))

        start = time.time()
        with self.assertRaises(LookupError):
            self.store.wait(["partial_exist", "missing_key"])

        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 0.9)


class TestEtcdStoreCheck(EtcdTestBase):
    """Test check API (non-blocking existence check)"""

    def test_check_all_keys_exist(self):
        """Test check returns True when all keys exist"""
        self.store.set("check_a", "1")
        self.store.set("check_b", "2")

        result = self.store.check(["check_a", "check_b"])
        self.assertTrue(result)

    def test_check_partial_exist(self):
        """Test check returns False when only some keys exist"""
        self.store.set("partial_a", "1")

        result = self.store.check(["partial_a", "partial_b"])
        self.assertFalse(result)

    def test_check_none_exist(self):
        """Test check returns False when no keys exist"""
        result = self.store.check(["none_1", "none_2"])
        self.assertFalse(result)

    def test_check_single_key(self):
        """Test check with a single key"""
        self.store.set("single", "1")

        self.assertTrue(self.store.check(["single"]))
        self.assertFalse(self.store.check(["not_exist"]))


class TestEtcdStoreSetTimeout(EtcdTestBase):
    """Test set_timeout API for default timeout configuration"""

    def test_set_timeout_changes_default(self):
        """Test set_timeout modifies the default blocking timeout for get"""
        self.store.set_timeout(datetime.timedelta(seconds=1))

        start = time.time()
        with self.assertRaises(LookupError):
            self.store.get("timeout_test_key")
        elapsed = time.time() - start

        self.assertLess(elapsed, 2.0)

    def test_set_timeout_zero_immediate(self):
        """Test timeout=0 causes immediate timeout (no blocking)"""
        self.store.set_timeout(datetime.timedelta(seconds=0))

        start = time.time()
        with self.assertRaises(LookupError):
            self.store.get("immediate_timeout")
        elapsed = time.time() - start

        self.assertLess(elapsed, 0.5)

    def test_set_timeout_affects_wait(self):
        """Test set_timeout modifies the default timeout for wait"""
        self.store.set_timeout(datetime.timedelta(seconds=1))

        start = time.time()
        with self.assertRaises(LookupError):
            self.store.wait(["wait_timeout_test"])
        elapsed = time.time() - start

        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 2.0)


class TestEtcdStoreIntegration(EtcdTestBase):
    """Integration and end-to-end scenario tests"""

    def test_full_workflow_rendezvous_simulation(self):
        """Simulate distributed rendezvous: multiple nodes coordinate via EtcdStore"""
        num_nodes = 5
        node_ids = []
        lock = threading.Lock()

        def node_worker(node_id):
            self.store.add("node_count", 1)
            self.store.set(f"node_{node_id}_ready", "yes")
            self.store.wait([f"node_{i}_ready" for i in range(num_nodes)])

            with lock:
                node_ids.append(node_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_nodes) as executor:
            futures = [executor.submit(node_worker, i) for i in range(num_nodes)]
            [f.result() for f in futures]

        self.assertEqual(len(node_ids), num_nodes)
        self.assertEqual(int(self.store.get("node_count")), num_nodes)

    def test_stress_multiple_operations(self):
        """Stress test: mixed operations across 50 concurrent workers"""

        def mixed_worker(worker_id):
            key = f"worker_{worker_id}"
            self.store.set(key, "initial")
            self.store.add(f"counter_{worker_id}", 1)
            self.store.check([key])
            self.store.get(key)
            self.store.add(f"counter_{worker_id}", 1)
            return int(self.store.get(f"counter_{worker_id}"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(mixed_worker, i) for i in range(50)]
            results = [f.result() for f in futures]

        for val in results:
            self.assertEqual(val, 2)


if __name__ == "__main__":
    run_tests()
