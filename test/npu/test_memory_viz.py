import sys
import os
import io
import json
from functools import lru_cache
from itertools import groupby
import warnings
import yaml
import torch_npu

from torch_npu.npu._memory_viz import format_flamegraph
from torch_npu.npu._memory_viz import _frame_fmt
from torch_npu.npu._memory_viz import _frame_filter
from torch_npu.npu._memory_viz import _block_extra_legacy
from torch_npu.npu._memory_viz import _block_extra
from torch_npu.npu._memory_viz import _write_blocks

from torch_npu.testing.testcase import TestCase, run_tests


class TestMemoryViz(TestCase):
    def test_block_extra_new_format(self):
        block = {
            'frames': [{'line': 10, 'filename': '/path/file.py', 'name': 'func1'}],
            'requested_size': 1024
        }
        frames, size = _block_extra(block)

        self.assertEqual(frames, block['frames'])
        self.assertEqual(size, block['requested_size'])

    def test_block_extra_legacy_with_history(self):
        block = {
            'history': [
                {
                    'frames': [{'line': 10, 'filename': '/path/file.py', 'name': 'func1'}],
                    'real_size': 1024
                }
            ],
            'size': 2048
        }
        frames, real_size = _block_extra_legacy(block)
        self.assertEqual(frames, [{'line': 10, 'filename': '/path/file.py', 'name': 'func1'}])
        self.assertEqual(real_size, 1024)

    def test_write_blocks_with_history(self):
        blocks = [{
            'state': 'allocated',
            'history': [
                {
                    'real_size': 1024,
                    'frames': [
                        {'line': 10, 'filename': '/path/to/file.py', 'name': 'func1'}
                    ]
                }
            ],
            'size': 2048
        }]
        f = io.StringIO()
        _write_blocks(f, 'prefix', blocks)
        result = f.getvalue()
        self.assertIn('prefix;allocated;file.py:10:func1 1024', result)
        self.assertIn('prefix;allocated;<gaps> 1024', result)

    def test_frame_filter_omitted_functions(self):
        result = _frame_filter("user_function", "/some/path")
        self.assertTrue(result)


if __name__ == "__main__":
    run_tests()