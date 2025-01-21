import random
import unittest
from torch_npu.distributed import ParallelStore


class ParallelStoreTest(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super(ParallelStoreTest, self).__init__(method_name)
        self._begin_port = random.randint(10000, 15000)
        self._port_offset = 0
        self._client = None
        self._server = None

    def setUp(self):
        self._port_offset += 1
        tcp_port = self._begin_port + self._port_offset
        self._server = ParallelStore(host="127.0.0.1", port=tcp_port, agent_run=True, agent_pid=100,
        is_server=True, enable_tiered=True, wait_workers=False, multi_tenant=True)
        self._client = ParallelStore(host="127.0.0.1", port=tcp_port, agent_run=False, agent_pid=100,
        is_server=False, enable_tiered=True)

    def tearDown(self):
        self._client = None
        self._server = None

    def test_client_set_and_server_get(self):
        key = 'key/ParallelStoreTest/client_set_and_server_get'
        value = b'value/ParallelStoreTest/client_set_and_server_get'
        self._client.set(key, value)

        result = self._server.get(key)
        self.assertEqual(value, result)

    def test_client_server_add(self):
        key = 'key/ParallelStoreTest/client_server_add'
        expected = 1
        value = self._client.add(key, 1)
        self.assertEqual(expected, value)

        for i in range(0, 100):
            expected += 1
            value = self._server.add(key, 1)
            self.assertEqual(expected, value)

            expected += 1
            value = self._client.add(key, 1)
            self.assertEqual(expected, value)

    def test_set_delete_key_and_key_count(self):
        key_base = 'key/ParallelStoreTest/set_delete_key_and_key_count'
        value_base = 'value/ParallelStoreTest/set_delete_key_and_key_count'
        keys = list()
        for i in range(0, 100):
            key = f'{key_base}/{i}'
            value = f'{value_base}/{i}'
            keys.append(key)

            old_key_count = self._server.num_keys()
            self._client.set(key, value)
            new_key_count = self._server.num_keys()
            self.assertEqual(old_key_count + 1, new_key_count)

        for key in keys:
            old_key_count = self._server.num_keys()
            self._client.delete_key(key)
            new_key_count = self._server.num_keys()
            self.assertEqual(old_key_count - 1, new_key_count)

    def test_multi_server_set_get(self):
        key = 'key/ParallelStoreTest/test_multi_server_set_get'
        tcp_port = self._begin_port + self._port_offset
        server2 = ParallelStore(host="127.0.0.1", port=tcp_port, agent_run=True, agent_pid=200,
        is_server=True, enable_tiered=True, wait_workers=False, multi_tenant=True)
        value1 = server2.add(key, 100)
        value2 = self._server.add(key, 0)
        self.assertEqual(value1, value2)


if __name__ == '__main__':
    unittest.main()
