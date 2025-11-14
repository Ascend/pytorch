from torch_npu.profiler.analysis.prof_common_func._constant import DbConstant
import torch_npu.profiler.analysis.prof_common_func._id_manager as id_manager
from torch_npu.testing.testcase import TestCase, run_tests


class TestIdManager(TestCase):
    def test_callchain_id_manager_callstack(self):
        manager = id_manager.CallChainIdManager()
        callstack = "stack1;\r\nstack2;\r\nstack3"
        result_id = manager.get_callchain_id_from_callstack(callstack)
        self.assertEqual(result_id, 0)
        callchain_map = manager.get_all_callchain_id()
        self.assertIn(0, callchain_map)
        self.assertEqual(len(callchain_map[0]), 3)

    def test_connection_id_manager_multiple_connections(self):
        manager = id_manager.ConnectionIdManager()
        conn_ids = [1, 2, 3]
        result_id = manager.get_id_from_connection_ids(conn_ids)
        self.assertEqual(result_id, 0)
        self.assertEqual(manager.get_connection_ids_from_id(0), conn_ids)

    def test_str2id_manager_repeated_string(self):
        manager = id_manager.Str2IdManager()
        id1 = manager.get_id_from_str("test")
        id2 = manager.get_id_from_str("test")
        self.assertEqual(id1, id2)

    def test_str2id_manager_empty_string(self):
        manager = id_manager.Str2IdManager()
        result = manager.get_id_from_str("")
        self.assertEqual(result, DbConstant.DB_INVALID_VALUE)


if __name__ == "__main__":
    run_tests()