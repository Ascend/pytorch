import copy
import os
import pickle
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch_npu  # noqa: F401
from torch._inductor import config
from torch._inductor.codecache import StaticAutotunerFuture
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.triton_compat import Config
from torch_npu._inductor.triton_experimental import npu_triton_heuristics as h


class RestoredAutotuner(h.NPUCachingAutotuner):
    # Exercise serialization and cache I/O without creating device handles.
    def is_statically_launchable(self):
        return bool(self.compile_results)


def candidate(block, extra=None):
    cfg = h.NPUConfig({"XBLOCK": block}, num_warps=8, num_stages=1)
    cfg.extra_options = extra or {}
    return cfg


class TestStaticAutotuneCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.environment = mock.patch.dict(os.environ, {
            "TORCHINDUCTOR_CACHE_DIR": self.tmp.name, "TRITON_INTERPRET": "0",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.settings = config.patch(
            force_disable_caches=False, autotune_local_cache=True,
            autotune_remote_cache=False,
        )
        self.settings.__enter__()
        self.addCleanup(self.settings.__exit__, None, None, None)
        self.meta = {"autotune_local_cache": True, "autotune_remote_cache": False}
        self.configs = [candidate(512), candidate(1024), candidate(2048)]
        self.filename = os.path.join(self.tmp.name, "ab", "kernel.py")

    def cache(self, configs=None, filename=None):
        return h._create_npu_autotune_cache(
            self.configs if configs is None else configs,
            filename or self.filename, self.meta, HeuristicType.POINTWISE,
        )

    def restored(self, configs=None, successful=None):
        configs = self.configs if configs is None else configs
        tuner = RestoredAutotuner.__new__(RestoredAutotuner)
        tuner._npu_autotune_configs = copy.deepcopy(configs)
        tuner._npu_cache_relative_filename = "ab/kernel.py"
        tuner.filename = self.filename
        tuner.heuristic_type = HeuristicType.POINTWISE
        tuner.inductor_meta = dict(self.meta)
        tuner.compile_results = [SimpleNamespace(config=cfg) for cfg in (
            configs if successful is None else successful
        )]
        tuner.launchers = []
        tuner.lock = threading.Lock()
        tuner.save_cache_hook = None
        return pickle.loads(pickle.dumps(tuner))

    def recheck(self, tuner):
        loader = mock.Mock(side_effect=AssertionError("unexpected source reload"))
        tuner.recheck_autotune_cache(loader)
        return tuner.autotune_cache_info["autotune_cache_state"]

    def test_plain_config_options_survive_conversion_and_serialization(self):
        original = Config({"XBLOCK": 512}, num_warps=8, num_stages=1)
        original.extra_options = {"auto_blockify_size": 4}
        converted = h._npu_unique_configs([original])[0]
        self.assertIsInstance(converted, h.NPUConfig)
        for restored in (copy.deepcopy(converted), pickle.loads(pickle.dumps(converted))):
            self.assertEqual(h._npu_config_key(restored), h._npu_config_key(original))
            restored.extra_options["auto_blockify_size"] = 8
            self.assertEqual(original.extra_options, {"auto_blockify_size": 4})
            self.assertEqual(converted.extra_options, {"auto_blockify_size": 4})

    def test_config_restores_without_parent_setstate(self):
        original = candidate(512, {"auto_blockify_size": 4})
        with mock.patch.object(Config, "__setstate__", None, create=True):
            for restored in (copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
                self.assertEqual(vars(restored), vars(original))
                restored.extra_options["auto_blockify_size"] = 8
                self.assertEqual(original.extra_options, {"auto_blockify_size": 4})

    def test_failed_candidates_do_not_change_identity(self):
        self.cache().save(self.configs[1], 1)
        tuner = self.restored(successful=self.configs[:2])
        self.assertEqual(self.recheck(tuner), "hit")
        self.assertEqual(len(tuner.compile_results), 1)
        self.assertEqual(h._npu_config_key(tuner.compile_results[0].config), h._npu_config_key(self.configs[1]))
        self.assertEqual(len(tuner._npu_autotune_configs), 3)

    def test_extra_options_select_the_correct_binary(self):
        configs = [candidate(512, {"auto_blockify_size": v}) for v in (2, 4)]
        self.cache(configs).save(configs[1], 1)
        tuner = self.restored(configs)
        self.assertEqual(self.recheck(tuner), "hit")
        self.assertEqual(tuner.compile_results[0].config.extra_options, {"auto_blockify_size": 4})
        self.assertEqual(tuner._npu_autotune_configs[0].extra_options, {"auto_blockify_size": 2})

    def test_candidate_changes_invalidate_best_config(self):
        self.cache().save(self.configs[0], 1)
        for configs in (self.configs[:2], self.configs + [candidate(4096)],
                        [candidate(512, {"auto_blockify_size": 2}), *self.configs[1:]]):
            with self.subTest(configs=configs):
                tuner = self.restored(configs)
                self.assertEqual(self.recheck(tuner), "miss")
                self.assertEqual(len(tuner.compile_results), len(configs))

    def test_cache_directory_migration_rebinds_reader_and_writer(self):
        cache = self.cache()
        cache.save(self.configs[1], 1)
        old_key = Path(cache.local_cache[1])
        tuner = self.restored(successful=self.configs[:2])
        with tempfile.TemporaryDirectory() as destination:
            new_key = Path(destination) / old_key.relative_to(self.tmp.name)
            new_key.parent.mkdir(parents=True)
            new_key.write_bytes(old_key.read_bytes())
            old_key.unlink()
            with mock.patch.dict(os.environ, TORCHINDUCTOR_CACHE_DIR=destination):
                self.assertEqual(self.recheck(tuner), "hit")
                rebound = tuner.save_cache_hook.__self__
                self.assertEqual(Path(rebound.local_cache[1]), new_key)
                tuner.save_cache_hook(self.configs[0], 1)
                self.assertTrue(new_key.exists())
                self.assertFalse(old_key.exists())

    def test_single_config_without_coordinate_descent_skips_cache(self):
        configs = [candidate(512)]
        self.assertIsNone(self.cache(configs))
        tuner = self.restored(configs)
        tuner._npu_cache_relative_filename = None
        self.assertEqual(self.recheck(tuner), "disabled")
        self.assertIsNone(tuner.save_cache_hook)

    def test_single_config_coordinate_descent_static_cache_reuse(self):
        self.meta["coordinate_descent_tuning"] = True
        configs = [candidate(512)]
        winner = candidate(256)
        winner.found_by_coordesc = True
        cache = self.cache(configs)
        self.assertIsNotNone(cache)
        cache.save(winner, 1, found_by_coordesc=True)
        # The binary existed before its coordinate-descent result was saved.
        tuner = self.restored(configs, successful=[candidate(256)])
        self.assertEqual(self.recheck(tuner), "hit")
        selected = tuner.compile_results[0].config
        self.assertTrue(selected.found_by_coordesc)
        self.assertIsNotNone(tuner.save_cache_hook)
        launcher = mock.Mock(config=selected, store_cubin=False)
        tuner.launchers = [launcher]
        with (
            mock.patch.object(h.TritonBundler, "_entries", None),
            mock.patch.object(tuner, "coordinate_descent_tuning") as tune,
        ):
            tuner.run(stream=0, benchmark_run=True)
        tune.assert_not_called()
        launcher.assert_called_once_with(stream=0)

    def test_missing_winner_keeps_successful_candidates(self):
        self.cache().save(self.configs[2], 1)
        tuner = self.restored(successful=self.configs[:2])
        self.assertEqual(self.recheck(tuner), "winner_missing")
        self.assertEqual(len(tuner.compile_results), 2)

    def test_legacy_snapshot_does_not_invent_candidate_identity(self):
        self.cache().save(self.configs[0], 1)
        tuner = self.restored()
        del tuner._npu_autotune_configs
        self.assertEqual(self.recheck(tuner), "miss")
        self.assertIsNone(tuner.save_cache_hook)
        self.assertEqual(len(tuner.compile_results), 3)

    def test_cache_disabled_in_producer_or_consumer(self):
        self.cache().save(self.configs[0], 1)
        for where in ("producer", "consumer", "local"):
            with self.subTest(where=where):
                tuner = self.restored()
                if where == "producer":
                    tuner.inductor_meta["force_disable_caches"] = True
                with config.patch(force_disable_caches=where == "consumer",
                                  autotune_local_cache=where != "local"):
                    self.assertEqual(self.recheck(tuner), "disabled")
                self.assertIsNone(tuner.save_cache_hook)
                self.assertEqual(len(tuner.compile_results), 3)

    def test_dynamic_fallback_winner_is_preserved(self):
        fallback = candidate(128)
        self.cache().save(fallback, 1)
        tuner = self.restored(successful=[fallback, self.configs[0]])
        self.assertEqual(self.recheck(tuner), "hit")
        self.assertEqual(tuner.compile_results[0].config.kwargs, fallback.kwargs)

    def test_cold_lookup_retains_original_candidates_before_narrowing(self):
        def kernel(x, XBLOCK):
            pass

        for configs in (self.configs, self.configs[:1]):
            coordesc = len(configs) == 1
            with self.subTest(coordinate_descent=coordesc):
                self.meta["coordinate_descent_tuning"] = coordesc
                winner = candidate(256) if coordesc else configs[1]
                self.cache(configs).save(winner, 1, found_by_coordesc=coordesc)
                with mock.patch.object(h, "NPUCachingAutotuner", return_value=SimpleNamespace()) as constructor:
                    tuner = h.cached_autotune(
                        {"x": 4096}, copy.deepcopy(configs), {}, HeuristicType.POINTWISE,
                        filename=self.filename, inductor_meta=dict(self.meta),
                    )(SimpleNamespace(fn=kernel))
                selected = constructor.call_args.kwargs["configs"]
                self.assertEqual([h._npu_config_key(cfg) for cfg in selected], [h._npu_config_key(winner)])
                if coordesc:
                    self.assertTrue(selected[0].found_by_coordesc)
                self.assertIsNotNone(constructor.call_args.kwargs["save_cache_hook"])
                self.assertEqual(h._npu_hash_configs(tuner._npu_autotune_configs), h._npu_hash_configs(configs))
                self.assertEqual(tuner._npu_cache_relative_filename, "ab/kernel.py")

    def test_signature_normalization_precedes_cache_identity(self):
        def kernel(x):
            pass

        configs = [candidate(1, {"auto_blockify_size": v}) for v in (2, 4)]
        with mock.patch.object(h, "NPUCachingAutotuner", return_value=SimpleNamespace()) as constructor:
            tuner = h.cached_autotune(
                {"x": 4096}, configs, {}, HeuristicType.POINTWISE,
                filename=self.filename, inductor_meta=dict(self.meta),
            )(SimpleNamespace(fn=kernel))
        self.assertTrue(all("XBLOCK" not in cfg.kwargs for cfg in tuner._npu_autotune_configs))
        self.assertEqual(constructor.call_args.kwargs["save_cache_hook"].__self__.configs_hash,
                         h._npu_hash_configs(tuner._npu_autotune_configs))

    def test_existing_best_config_remains_readable(self):
        # Identity produced by the pre-existing NPU cold path for self.configs.
        old_hash = "784b61e3587a42051aa1bd2398a40bdbb58f2dd84606f95dba66e65ee03a64bf"
        cache = h.AutotuneCache.create(self.meta, self.filename, old_hash)
        cache.save(self.configs[1], 1)
        tuner = self.restored(successful=self.configs[:2])
        self.assertEqual(self.recheck(tuner), "hit")
        self.assertEqual(tuner.compile_results[0].config.kwargs, self.configs[1].kwargs)

    def test_upstream_future_selects_winner_before_precompile(self):
        self.cache().save(self.configs[1], 1)
        tuner = self.restored(successful=self.configs[:2])
        future = StaticAutotunerFuture(tuner)
        future.reload_kernel_from_src = mock.Mock(side_effect=AssertionError("unexpected source reload"))
        prepared = []

        def precompile(**kwargs):
            prepared.extend(result.config.kwargs for result in tuner.compile_results)

        with mock.patch.object(tuner, "precompile", side_effect=precompile):
            self.assertIs(future.result(), tuner)
        self.assertEqual(prepared, [self.configs[1].kwargs])

    def test_external_source_uses_consumer_loader_filename(self):
        self.cache().save(self.configs[1], 1)
        tuner = self.restored()
        tuner._npu_cache_relative_filename = None
        tuner.filename = "/unavailable/producer/kernel.py"
        loader = mock.Mock(return_value=SimpleNamespace(filename=self.filename))
        tuner.recheck_autotune_cache(loader)
        loader.assert_called_once_with()
        self.assertEqual(tuner.autotune_cache_info["autotune_cache_state"], "hit")
        self.assertEqual(tuner.filename, self.filename)

    def test_disabled_cache_does_not_reload_external_source(self):
        tuner = self.restored()
        tuner._npu_cache_relative_filename = None
        for patch in ({"force_disable_caches": True}, {"autotune_local_cache": False}):
            with self.subTest(patch=patch), config.patch(patch):
                self.assertEqual(self.recheck(tuner), "disabled")


if __name__ == "__main__":
    unittest.main()
