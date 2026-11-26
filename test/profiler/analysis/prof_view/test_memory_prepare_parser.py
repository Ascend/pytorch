import unittest
from collections import defaultdict, namedtuple
from types import SimpleNamespace
from unittest.mock import patch

from torch_npu.profiler.analysis.prof_view._memory_prepare_parser import (
    MemoryPrepareParser,
    _is_alloc_plumbing,
    _is_pypto_op_name,
    _resolve_pypto_owner,
)


def run_test():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPyptoNameRecognition))
    suite.addTests(loader.loadTestsFromTestCase(TestResolvePyptoOwner))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryPrepareParserPypto))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def make_node(name, start_time, end_time, parent_node=None):
    return SimpleNamespace(
        name=name,
        start_time=start_time,
        end_time=end_time,
        parent_node=parent_node,
        event=True,
    )


def make_parser():
    parser = object.__new__(MemoryPrepareParser)
    parser.torch_ops_tid_dict = defaultdict(list)
    parser._torch_ops_index = {}
    parser._record_to_name = {}
    parser._has_pypto_op = True
    parser._dequeue_pids = set()
    parser._dequeue_tids = set()
    parser._dequeue_record_dict = defaultdict(list)
    parser._dequeue_record_index = {}
    parser._enqueue_record_dict = {}
    return parser


FakeRecord = namedtuple("FakeRecord", ("time_ns", "tid", "pid"))


def make_record(time_ns=15, tid=1, pid=1):
    return FakeRecord(time_ns, tid, pid)


class TestPyptoNameRecognition(unittest.TestCase):

    def test_pypto_names(self):
        for name in (
            "pypto::write_read_write",
            "pypto_pro::silu_pypto_pro",
            "pypto.write_read_write",
            "pypto_pro.silu_pypto_pro",
        ):
            with self.subTest(name=name):
                self.assertTrue(_is_pypto_op_name(name))

    def test_non_pypto_names(self):
        for name in ("", "aten::empty", "npu::empty", "atb::op", "pyptox::op", "pypto_prox::op"):
            with self.subTest(name=name):
                self.assertFalse(_is_pypto_op_name(name))

    def test_alloc_plumbing(self):
        for name in (
            "aten::empty",
            "aten::empty_like",
            "aten::empty_strided",
            "aten::new_empty",
            "aten::new_empty_strided",
            "aten.empty_like",
            "aten::empty_like.",
            "aten::empty.memory_format",
            "aten::new_empty_strided.default",
            "npu::empty",
            "npu.empty",
        ):
            with self.subTest(name=name):
                self.assertTrue(_is_alloc_plumbing(name))
        for name in (
            "",
            "aten::zeros",
            "aten::add",
            "aten::empty_permuted",
            "aten::empty_like_quantized",
            "npu::empty_cache",
            "pypto::empty",
        ):
            with self.subTest(name=name):
                self.assertFalse(_is_alloc_plumbing(name))


class TestResolvePyptoOwner(unittest.TestCase):

    def test_resolve_plumbing_to_pypto_parent(self):
        cases = (
            ("aten::empty_like", "pypto_pro::silu_pypto_pro"),
            ("aten.empty_like", "pypto::write_read_write"),
            ("aten::empty.memory_format", "pypto::write_read_write"),
            ("npu::empty", "pypto::write_read_write"),
            ("aten::empty", "pypto_pro.silu_pypto_pro"),
        )
        for plumbing_name, pypto_name in cases:
            with self.subTest(plumbing_name=plumbing_name, pypto_name=pypto_name):
                pypto_node = make_node(pypto_name, 0, 100)
                empty_node = make_node(plumbing_name, 10, 20, pypto_node)
                self.assertEqual(_resolve_pypto_owner(empty_node, plumbing_name), pypto_name)

    def test_resolve_nested_pypto_takes_nearest(self):
        outer = make_node("pypto::outer", 0, 100)
        inner = make_node("pypto::inner", 10, 90, outer)
        empty = make_node("aten::empty", 20, 30, inner)
        self.assertEqual(_resolve_pypto_owner(empty, "aten::empty"), "pypto::inner")

    def test_resolve_keeps_non_plumbing_aten(self):
        pypto = make_node("pypto::op", 0, 100)
        add = make_node("aten::add", 10, 20, pypto)
        self.assertEqual(_resolve_pypto_owner(add, "aten::add"), "aten::add")

    def test_resolve_keeps_already_pypto(self):
        pypto = make_node("pypto_pro::silu_pypto_pro", 0, 100)
        self.assertEqual(
            _resolve_pypto_owner(pypto, "pypto_pro::silu_pypto_pro"),
            "pypto_pro::silu_pypto_pro",
        )

    def test_resolve_plumbing_without_pypto_unchanged(self):
        for name in ("aten::empty", "aten.empty_like", "aten::empty.memory_format", "npu::empty"):
            with self.subTest(name=name):
                node = make_node(name, 10, 20)
                self.assertEqual(_resolve_pypto_owner(node, name), name)

    def test_resolve_none_node(self):
        self.assertEqual(_resolve_pypto_owner(None, "aten::empty"), "aten::empty")


class TestMemoryPrepareParserPypto(unittest.TestCase):

    def test_init_torch_op_detects_pypto(self):
        cases = (
            ([make_node("root", 0, 100), make_node("pypto::write_read_write", 10, 20)], True),
            ([make_node("root", 0, 100), make_node("aten::empty_like", 10, 20)], False),
        )
        for nodes, expected in cases:
            with self.subTest(expected=expected):
                parser = make_parser()
                parser._profiler_path = "/tmp/fake"
                parser._torch_op_node = nodes
                with patch(
                    "torch_npu.profiler.analysis.prof_view._memory_prepare_parser"
                    ".ProfilerPathManager.get_cann_path",
                    return_value="/cann",
                ):
                    parser._init_torch_op()
                self.assertIs(parser._has_pypto_op, expected)
                self.assertEqual(len(parser._torch_op_node), 1)

    def test_direct_path_rollup(self):
        parser = make_parser()
        pypto = make_node("pypto_pro::silu_pypto_pro", 0, 100)
        empty = make_node("aten::empty_like", 10, 20, pypto)
        parser.torch_ops_tid_dict[1] = [empty]
        self.assertEqual(parser._get_op_name_of_record(make_record()), "pypto_pro::silu_pypto_pro")

    def test_direct_path_climbs_to_active_pypto_parent(self):
        parser = make_parser()
        pypto = make_node("pypto::write_read_write", 0, 100)
        empty = make_node("aten::empty", 10, 12, pypto)
        parser.torch_ops_tid_dict[1] = [empty]
        self.assertEqual(parser._get_op_name_of_record(make_record()), "pypto::write_read_write")

    def test_direct_path_without_pypto_unchanged(self):
        for name in ("aten::add", "aten::empty"):
            with self.subTest(name=name):
                parser = make_parser()
                node = make_node(name, 10, 20)
                parser.torch_ops_tid_dict[1] = [node]
                self.assertEqual(parser._get_op_name_of_record(make_record()), name)

    def test_dequeue_path_plumbing_rollup(self):
        parser = make_parser()
        pypto = make_node("pypto::write_read_write", 0, 100)
        empty = make_node("aten::empty", 10, 20, pypto)
        parser.torch_ops_tid_dict[1] = [empty]
        enqueue = SimpleNamespace(time_ns=15, tid=1)
        node = parser._get_aten_op_node_by_enqueue_record(enqueue)
        self.assertIs(node, empty)
        self.assertEqual(_resolve_pypto_owner(node, node.name), "pypto::write_read_write")

    def test_dequeue_path_already_pypto(self):
        parser = make_parser()
        pypto = make_node("pypto::write_read_write", 0, 100)
        parser.torch_ops_tid_dict[1] = [pypto]
        enqueue = SimpleNamespace(time_ns=15, tid=1)
        node = parser._get_aten_op_node_by_enqueue_record(enqueue)
        self.assertIs(node, pypto)
        self.assertEqual(_resolve_pypto_owner(node, node.name), "pypto::write_read_write")

    def test_dequeue_path_non_plumbing_unchanged(self):
        parser = make_parser()
        pypto = make_node("pypto::write_read_write", 0, 100)
        add = make_node("aten::add", 10, 20, pypto)
        parser.torch_ops_tid_dict[1] = [add]
        enqueue = SimpleNamespace(time_ns=15, tid=1)
        node = parser._get_aten_op_node_by_enqueue_record(enqueue)
        self.assertIs(node, add)
        self.assertEqual(_resolve_pypto_owner(node, node.name), "aten::add")

    def test_dequeue_record_rollup_via_get_op_name(self):
        parser = make_parser()
        pypto = make_node("pypto::write_read_write", 0, 100)
        empty = make_node("aten::empty", 10, 20, pypto)
        parser.torch_ops_tid_dict[1] = [empty]
        parser._dequeue_pids = {1}
        parser._dequeue_tids = {1}
        dequeue = SimpleNamespace(corr_id=1, ts=10, dur=10, pid=1, tid=1)
        parser._dequeue_record_dict[(1, 1)] = [dequeue]
        parser._enqueue_record_dict = {
            1: SimpleNamespace(corr_id=1, time_ns=15, tid=1),
        }
        self.assertEqual(parser._get_op_name_of_record(make_record()), "pypto::write_read_write")

    def test_no_torch_ops_returns_empty(self):
        parser = make_parser()
        self.assertEqual(parser._get_op_name_of_record(make_record()), "")

    def test_cache_reuses_resolved_name(self):
        parser = make_parser()
        pypto = make_node("pypto_pro::silu_pypto_pro", 0, 100)
        empty = make_node("aten::empty", 10, 20, pypto)
        parser.torch_ops_tid_dict[1] = [empty]
        record = make_record()
        self.assertEqual(parser._get_op_name_of_record(record), "pypto_pro::silu_pypto_pro")
        parser.torch_ops_tid_dict[1] = []
        self.assertEqual(parser._get_op_name_of_record(record), "pypto_pro::silu_pypto_pro")


if __name__ == '__main__':
    run_test()
