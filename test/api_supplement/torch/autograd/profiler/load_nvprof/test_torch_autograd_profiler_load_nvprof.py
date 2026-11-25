# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler.load_nvprof 接口功能正确性
API 名称：torch.autograd.profiler.load_nvprof
API 签名：torch.autograd.profiler.load_nvprof(path)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空 SQLite 数据库与含事件记录数据库 | 已覆盖（empty_and_valid_nvprof_db） |
| 枚举选项 | 无枚举参数 | 未覆盖（接口仅接收文件路径 path） |
| 参数类型 | 合法路径字符串与缺参调用 | 已覆盖（empty_and_valid_nvprof_db、error_cases_raise） |
| 传参与不传参 | 必选 path 传入与不传参报错 | 已覆盖（empty_and_valid_nvprof_db、error_cases_raise） |
| 等价类/边界值 | 空表、多事件含 kernel、损坏文件、缺失目录 | 已覆盖（empty_and_valid_nvprof_db、error_cases_raise） |
| 正常传参场景 | 解析真实 SQLite 文件与 mock 解析器封装 | 已覆盖（empty_and_valid_nvprof_db、mocked_parser_preserves_references） |
| 异常传参场景 | 路径不存在、损坏文件、缺参 | 已覆盖（error_cases_raise） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（接口仅读取磁盘 SQLite 文件，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口仅接收单个文件路径参数 path，无枚举选项。
- 混合设备类型：接口仅解析 nvprof SQLite 文件并返回 EventList，不接收 Tensor 参数。

注意：本接口解析磁盘 nvprof SQLite 文件并返回 EventList，
     按 API 语义使用返回结构、事件字段、对象引用及异常断言验证。
"""
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.autograd.profiler import load_nvprof
from torch.autograd.profiler_util import EventList, FunctionEvent

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


def _create_nvprof_db(db_path, with_data=False):
    ddl = (
        "CREATE TABLE StringTable (_id_ INTEGER, value TEXT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MARKER (id INTEGER, name INTEGER, timestamp INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
        "(_id_ INTEGER, cbid INTEGER, start INTEGER, end INTEGER, correlationId INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL "
        "(start INTEGER, end INTEGER, name TEXT, correlationId INTEGER);"
    )
    dml = (
        "INSERT INTO StringTable VALUES (1, 'aten::matmul'), (2, 'aten::add');"
        "INSERT INTO CUPTI_ACTIVITY_KIND_MARKER VALUES (20, 2, 500), (20, 0, 700), (10, 1, 100), (10, 0, 400);"
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1, 211, 150, 250, 99);"
        "INSERT INTO CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL VALUES (200, 320, 'gemm_k', 99);"
    )
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(ddl + (dml if with_data else ""))
    finally:
        conn.close()


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLoadNvprof(TestCase):
    """Test cases for torch.autograd.profiler.load_nvprof."""

    def test_empty_and_valid_nvprof_db(self):
        """Empty and populated nvprof SQLite databases parse into expected EventList objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = os.path.join(tmpdir, "empty.nvprof")
            _create_nvprof_db(empty_path, with_data=False)
            empty_events = load_nvprof(empty_path)
            self.assertIsInstance(empty_events, EventList)
            self.assertEqual(len(empty_events), 0)
            db_path = os.path.join(tmpdir, "trace.nvprof")
            _create_nvprof_db(db_path, with_data=True)
            events = load_nvprof(db_path)
            self.assertIsInstance(events, EventList)
            self.assertEqual(len(events), 2)
            ev0 = events[0]
            self.assertEqual((ev0.name, ev0.time_range.start, ev0.kernels[0].name, ev0.kernels[0].duration,
                              events[1].name), ("aten::matmul", 100, "gemm_k", 120, "aten::add"))

    def test_mocked_parser_preserves_references(self):
        """load_nvprof wraps parsed FunctionEvent instances in an EventList preserving identity."""
        e1 = FunctionEvent(id=1, name="op_a", thread=1, start_us=0, end_us=10)
        with patch("torch.autograd.profiler._parse_nvprof_trace", return_value=[e1]) as mock_parse:
            events = load_nvprof("dummy.prof")
            mock_parse.assert_called_once_with("dummy.prof")
        self.assertIsInstance(events, EventList)
        self.assertIs(events[0], e1)

    def test_error_cases_raise(self):
        """Missing directory, corrupt file, and missing path argument raise expected errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises((sqlite3.OperationalError, OSError)):
                load_nvprof(os.path.join(tmpdir, "missing_subdir", "trace.nvprof"))
            bad_file = os.path.join(tmpdir, "corrupt.nvprof")
            with open(bad_file, "wb") as f:
                f.write(b"not a sqlite db")
            with self.assertRaises(sqlite3.DatabaseError):
                load_nvprof(bad_file)
        with self.assertRaises(TypeError):
            load_nvprof()


if __name__ == "__main__":
    run_tests()
