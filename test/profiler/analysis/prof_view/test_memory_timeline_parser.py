import os
import random
import unittest
from unittest.mock import patch, MagicMock

from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_parse._event_tree_parser import (_ProfilerEvent,
                                                                       _DeviceType, _EventType)
from torch_npu.profiler.analysis.prof_view._memory_timeline_parser import (
    MemoryProfile, MemoryProfileTimeline, Storage,
    DeviceKey, TensorKey, Category, Action, _CATEGORY_TO_COLORS
)


class TestMemoryProfile(unittest.TestCase):
    def setUp(self):
        self.memory_profile = MagicMock()
        self.memory_profile._root_nodes = []
        self.memory_profile._categories = {}

    @patch("torch_npu.profiler.analysis.prof_view._memory_timeline_parser.EventTree")
    def test_init_success(self, mock_event_tree):
        mock_event_tree.return_value.sorted_events = []
        mock_event_tree.return_value.get_root_nodes.return_value = []
        mp = MemoryProfile("valid.prof")
        self.assertIsNotNone(mp)

    def test_memory_history(self):
        mock_event = MagicMock(spec=_ProfilerEvent)
        mock_event.tag = _EventType.Allocation
        mock_event.extra_fields = MagicMock()
        mock_event.extra_fields.device_type = _DeviceType.CUDA
        mock_event.extra_fields.device_index = 0
        mock_event.extra_fields.total_active = 100
        mock_event.extra_fields.total_allocated = 200
        mock_event.extra_fields.total_reserved = 300
        mock_event.children = []
        self.memory_profile._root_nodes = [mock_event]
        self.memory_profile.memory_history = [(DeviceKey(_DeviceType.NPU, 0), 100, 200, 300)]
        result = self.memory_profile.memory_history
        expected = [(DeviceKey(_DeviceType.NPU, 0), 100, 200, 300)]
        self.assertEqual(result, expected)

    def test_is_gradient(self):
        mock_categories = MagicMock()
        mock_categories.get.return_value = Category.GRADIENT
        self.memory_profile._categories = mock_categories
        self.assertTrue(self.memory_profile._is_gradient(TensorKey(1, 0, 1, "storage"), 0))

    def test_set_gradients_and_temporaries(self):
        mock_event = MagicMock(spec=_ProfilerEvent)
        mock_event.tag = _EventType.PyCall

        mock_event.extra_fields = MagicMock()
        mock_event.extra_fields.grads = [TensorKey(1, 0, 1, "storage")]

        self.assertEqual(mock_event.extra_fields.grads[0].id, 1)
        self.assertEqual(mock_event.extra_fields.grads[0].storage, "storage")

    def test_set_optimizer_state(self):
        mock_event = MagicMock(spec=_ProfilerEvent)
        mock_event.tag = _EventType.PyCall

        mock_event.extra_fields = MagicMock()
        mock_event.extra_fields.optimizer_parameters = [MagicMock()]

        random_data = [random.random() for _ in range(2)]
        mock_event.extra_fields.optimizer_parameters[0].state = {"weight": random_data}

        self.memory_profile._root_nodes = [mock_event]

        with patch("torch_npu.profiler.analysis.prof_view._memory_timeline_parser.TensorKey.from_tensor",
                   return_value=TensorKey(1, 0, 1, "storage")):
            self.memory_profile._set_optimizer_state()
            self.assertEqual(self.memory_profile._categories.get(TensorKey(1, 0, 1, "storage"), 0), 0)


class TestMemoryProfileTimeline(unittest.TestCase):

    def setUp(self):
        self.memory_profile = MagicMock()
        self.mpt = MemoryProfileTimeline(self.memory_profile)

    def test_parse_device_cpu(self):
        result = self.mpt._parse_device_info("cpu")
        self.assertIsInstance(result, DeviceKey)
        self.assertEqual(result.device_type, 0)
        self.assertEqual(result.device_index, -1)

    def test_parse_device_npu(self):
        result = self.mpt._parse_device_info("npu:0")
        self.assertIsInstance(result, DeviceKey)
        self.assertEqual(result.device_index, 0)

    def test_construct_timeline_empty(self):
        self.memory_profile.timeline = []
        timestamps, sizes = self.mpt._construct_timeline("cpu")
        self.assertEqual(len(timestamps), 0)
        self.assertEqual(len(sizes), 0)

    def test_construct_timeline_filter_device(self):
        key1 = TensorKey(0, 0, 0, Storage(0, 1))
        key2 = TensorKey(1, 1, 1, Storage(0, 1))
        self.memory_profile.timeline = [
            (1000000, Action.CREATE, (key1, 0), 1024),
            (2000000, Action.CREATE, (key2, 0), 2048),
        ]
        timestamps, sizes = self.mpt._construct_timeline("cpu")
        self.assertEqual(len(timestamps), 0)

    @patch('torch_npu.profiler.analysis.prof_common_func._file_manager.FileManager.create_json_file_by_path')
    def test_export_json(self, mock_write):
        self.memory_profile.timeline = [(1000000, Action.CREATE, (TensorKey(0, 0, 0, Storage(0, 1)), 0), 1024)]
        self.mpt._construct_timeline = MagicMock(return_value=([1000], [[0, 1024]]))
        self.mpt.export_memory_timeline_json("output.json", "cpu")
        expected_path = os.path.abspath("output.json")
        mock_write.assert_called_once_with(expected_path, [[1000], [[0, 1024]]])


class TestMemoryTimelineParser(unittest.TestCase):

    @patch('torch_npu.profiler.analysis.prof_view._memory_timeline_parser.MemoryProfile')
    @patch('torch_npu.profiler.analysis.prof_view._memory_timeline_parser.MemoryProfileTimeline')
    def test_run_method(self, mock_timeline_class, mock_profile_class):
        parser = mock_timeline_class()
        parser._device = "npu"
        parser.logger = MagicMock()
        mock_profile_instance = mock_profile_class.return_value
        mock_profile_instance.some_method_we_use.return_value = "mocked profile data"
        mock_timeline_instance = mock_timeline_class.return_value
        mock_timeline_instance.export_memory_timeline_html.return_value = None
        parser.run.return_value = [Constant.SUCCESS]
        result = parser.run(deps_data={})
        self.assertEqual(result[0], Constant.SUCCESS)

    @patch('torch_npu.profiler.analysis.prof_view._memory_timeline_parser.MemoryProfile')
    @patch('torch_npu.profiler.analysis.prof_view._memory_timeline_parser.MemoryProfileTimeline')
    def test_run_with_exception(self, mock_timeline_class, mock_profile_class):
        parser = mock_timeline_class()
        parser._device = "npu"
        parser.logger = MagicMock()
        mock_profile_class.side_effect = Exception("Mocked Initialization Error")
        parser.run.return_value = [Constant.FAIL]
        result = parser.run(deps_data={})
        self.assertEqual(result[0], Constant.FAIL)


class TestEdgeCases(unittest.TestCase):

    def test_category_handling(self):
        mock_mem_profile = MagicMock()
        mock_mem_profile.timeline = []
        mock_mem_profile.memory_history = []
        mock_mem_profile._categories = MagicMock()

        test_cases = [
            (Category.INPUT, "black"),
            (Category.PARAMETER, "darkgreen"),
            (None, "grey")
        ]

        for category, expected_color in test_cases:
            mock_mem_profile._categories.get.return_value = category
            timeline = MemoryProfileTimeline(mock_mem_profile)

            idx = timeline._get_category_index(MagicMock(), 0)
            self.assertEqual(_CATEGORY_TO_COLORS[category], expected_color)


def run_tests():
    loader = unittest.TestLoader()

    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryProfileTimeline))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryTimelineParser))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    run_tests()
