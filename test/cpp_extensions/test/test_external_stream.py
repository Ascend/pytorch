import os
import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
CPP_EXTENSIONS_DIR = REPO_ROOT / "test" / "cpp_extensions"

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
for path in ("", str(REPO_ROOT)):
    while path in sys.path:
        sys.path.remove(path)

sys.path.insert(0, str(CPP_EXTENSIONS_DIR))

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from torch_test_cpp_extension.load_external_stream import load_external_stream_extension

ext_stream_ext = load_external_stream_extension()


class TestExternalStream(TestCase):
    """Tests for external NPU stream functionality

    These tests verify:
    1. External stream creation via getStreamFromExternal
    2. Stream ID encoding (bit 30 marker for external streams)
    3. Proper restrictions on external streams (event, graph, query, sync)
    4. Basic operations work on external streams
    """

    def test_external_stream_creation(self):
        """Test external stream creation and ID encoding"""
        self.assertTrue(ext_stream_ext.test_external_stream_creation())

    def test_external_stream_as_current(self):
        """Test setCurrentNPUStream/getCurrentNPUStream with external stream"""
        self.assertTrue(ext_stream_ext.test_external_stream_as_current())

    def test_operations_on_external_stream(self):
        """Test tensor operations on external stream"""
        self.assertTrue(ext_stream_ext.test_operations_on_external_stream())

    def test_event_record_restriction(self):
        """Test NPUEvent.record() throws for external stream"""
        self.assertTrue(ext_stream_ext.test_event_record_restriction())

    def test_event_block_restriction(self):
        """Test NPUEvent.block() throws for external stream"""
        self.assertTrue(ext_stream_ext.test_event_block_restriction())

    def test_graph_capture_restriction(self):
        """Test NPUGraph.capture_begin() throws when current stream is external"""
        self.assertTrue(ext_stream_ext.test_graph_capture_restriction())

    def test_query_restriction(self):
        """Test NPUStream.query() throws for external stream"""
        self.assertTrue(ext_stream_ext.test_query_restriction())

    def test_synchronize_restriction(self):
        """Test NPUStream.synchronize() throws for external stream"""
        self.assertTrue(ext_stream_ext.test_synchronize_restriction())

    def test_is_sync_launch_stream(self):
        """Test external stream is not a sync launch stream"""
        self.assertTrue(ext_stream_ext.test_is_sync_launch_stream())

    def test_multiple_external_streams(self):
        """Test multiple external streams have unique IDs"""
        self.assertTrue(ext_stream_ext.test_multiple_external_streams())

    def test_same_acl_stream_same_npu_stream(self):
        """Test getStreamFromExternal is idempotent"""
        self.assertTrue(ext_stream_ext.test_same_acl_stream_same_npu_stream())

    def test_pool_vs_external_stream(self):
        """Test pool stream vs external stream ID encoding"""
        self.assertTrue(ext_stream_ext.test_pool_vs_external_stream())


if __name__ == "__main__":
    run_tests()