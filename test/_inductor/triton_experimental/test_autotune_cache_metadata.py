# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import unittest
from unittest import mock

from torch._inductor.runtime import autotune_cache as upstream
from torch._inductor.runtime.triton_compat import Config, HAS_WARP_SPEC
from torch_npu._inductor.runtime.autotune_cache import _load_cached_autotuning


class TestAutotuneCacheMetadata(unittest.TestCase):
    def test_save_read_roundtrip(self):
        for coordesc in (False, True):
            for extra in (None, {"auto_blockify_size": 4}):
                with self.subTest(coordesc=coordesc, extra=extra):
                    config = Config({"XBLOCK": 3136}, num_warps=8, num_stages=1)
                    config.extra_options = extra
                    if HAS_WARP_SPEC:
                        config.num_consumer_groups = 2
                        config.num_buffers_warp_spec = 3
                    storage = mock.Mock()
                    cache = upstream.AutotuneCache("hash", remote_cache=(storage, "key"))
                    cache.save(config, 1000000, found_by_coordesc=coordesc, triton_cache_hash="binary")
                    record = storage.put.call_args.args[1]
                    storage.get.return_value = record
                    with mock.patch.object(upstream, "_load_cached_autotuning", _load_cached_autotuning):
                        # A dynamically selected winner need not be in the original candidates.
                        loaded = cache.read_best({"coordinate_descent_tuning": coordesc}, [])
                    self.assertEqual(loaded.kwargs, {"XBLOCK": 3136})
                    self.assertEqual((loaded.num_warps, loaded.num_stages), (8, 1))
                    self.assertEqual(loaded.extra_options, extra)
                    self.assertTrue(loaded.found_by_coordesc)
                    if HAS_WARP_SPEC:
                        self.assertEqual(loaded.num_consumer_groups, 2)
                        self.assertEqual(loaded.num_buffers_warp_spec, 3)
                    self.assertIs(loaded.kwargs, record)
                    self.assertEqual(record, {"XBLOCK": 3136})

    def test_missing_optional_fields(self):
        record = {"configs_hash": "hash", "XBLOCK": 64, "num_warps": 8, "num_stages": 1}
        loaded = _load_cached_autotuning(record, "hash", [], {})
        self.assertEqual(loaded.kwargs, {"XBLOCK": 64})
        self.assertIsNone(loaded.extra_options)
        self.assertTrue(loaded.found_by_coordesc)
        self.assertIs(loaded.kwargs, record)

    def test_cache_miss(self):
        self.assertIsNone(_load_cached_autotuning(None, "hash", [], {}))
        record = {"configs_hash": "other"}
        self.assertIsNone(_load_cached_autotuning(record, "hash", [], {}))
        self.assertEqual(record, {})


if __name__ == "__main__":
    unittest.main()
