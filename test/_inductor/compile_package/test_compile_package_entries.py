# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Cross-process reuse for the three entry points a DVM compilation can use.

Each case runs a producer and a consumer as separate interpreters against one
cache root, so what is measured is reuse across a process boundary rather than
state a single process already holds. The entries are isolated by the worker:
``fx`` disables the AOT cache (which is consulted first and would otherwise
report the hit) and only ``package`` enables ``caching_precompile``.

Assertions are about the entry under test: the producer misses, the consumer
hits, and both results are correct. Counters of the other layers are asserted to
stay quiet so a passing case cannot come from the wrong cache.
"""
from pathlib import Path
import shutil
import tempfile
import unittest

import torch
import torch_npu  # noqa: F401  - registers torch.npu; autoload is off in the children
from torch.testing._internal.common_utils import TestCase, run_tests

from compile_package_entry_runner import EntryRunner

RUN_NPU = torch.npu.is_available()
# test/_inductor/compile_package -> repository root holding torch_npu/
SOURCE_TREE = Path(__file__).resolve().parents[3]


@unittest.skipIf(not RUN_NPU, "requires npu")
class EntryReuseTests(TestCase):
    """One producer/consumer pair per entry point."""

    @classmethod
    def setUpClass(cls):
        cls._root = Path(tempfile.mkdtemp(prefix="npu-entry-tests-"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._root, ignore_errors=True)

    def runner(self, entry="run"):
        return EntryRunner(self._root / self._testMethodName / entry,
                           source_tree=SOURCE_TREE)

    def pair(self, entry):
        """Run one entry's producer/consumer pair and return their results."""
        runner = self.runner(entry)
        producer, consumer = runner.pair(entry)
        self.assertEqual(
            producer["returncode"], 0,
            f"producer failed: {producer.get('error')}\n"
            f"log: {runner.root / (entry + '_producer') / 'log.txt'}")
        self.assertEqual(
            consumer["returncode"], 0,
            f"consumer failed: {consumer.get('error')}\n"
            f"log: {runner.root / (entry + '_consumer') / 'log.txt'}")
        return producer["result"], consumer["result"]

    # ---------------------------------------------------------------- package
    def test_package_entry_reuses_the_whole_compilation(self):
        # The package carries Dynamo code, guards and backend artifacts; a hit
        # means the Python layer was restored instead of re-traced.
        producer, consumer = self.pair("package")
        self.assertTrue(producer["correct"])
        self.assertTrue(consumer["correct"])
        self.assertEqual(producer["dynamo_cache"].get("dynamo_cache_miss"), 1)
        self.assertEqual(consumer["dynamo_cache"].get("dynamo_cache_hit"), 1)
        self.assertNotIn("dynamo_cache_miss", consumer["dynamo_cache"])

    def test_package_entry_persists_a_reusable_environment(self):
        # An entry written without chip/CANN would be rejected by the very host
        # that produced it, once the NPU environment check is in place. That is
        # the failure this pins: if the producer omits the values, the consumer
        # on the same machine misses instead of hitting.
        #
        # Note what a same-host round trip cannot prove: with "npu" absent from
        # CHECK_GPUS the device branch never runs, yet producer and consumer on
        # one host still accept each other. Cross-chip protection is only
        # observable by comparing two *different* environments, which
        # test_compile_package_system_info covers. Here the assertion is reuse.
        producer, consumer = self.pair("package")
        self.assertTrue(producer.get("saved"))
        self.assertGreaterEqual(producer["cache_dirs"]["dynamo"], 1)
        self.assertEqual(consumer["dynamo_cache"].get("dynamo_cache_hit"), 1)

    def test_changed_codegen_option_runs_correctly_without_crashing(self):
        # A DVM codegen knob feeds the backend pass UUID, so the lower keys move.
        # What must hold either way is that the consumer produces the right
        # answer; whether it hits is deliberately not asserted, because that is
        # a reuse policy upstream is still tightening.
        runner = self.runner()
        cache = runner.root / "cache"
        producer = runner.run("producer", "package", "producer", cache=cache)
        consumer = runner.run("consumer", "package", "consumer", cache=cache,
                              change_config=True)
        self.assertEqual(producer["returncode"], 0, producer.get("error"))
        self.assertEqual(
            consumer["returncode"], 0,
            f"a changed option must not break the consumer: {consumer.get('error')}\n"
            f"log: {runner.root / 'consumer' / 'log.txt'}")
        result = consumer["result"]
        self.assertTrue(result["correct"])
        self.assertIn("disable_addmm_fusion", result["changed_config"])

    # -------------------------------------------------------------------- fx
    def test_fx_entry_reuses_the_graph_without_a_package(self):
        # No package involved, so only the FX graph cache can supply the hit.
        producer, consumer = self.pair("fx")
        self.assertTrue(producer["correct"])
        self.assertTrue(consumer["correct"])
        self.assertEqual(producer["fxgraph"].get("fxgraph_cache_miss"), 1)
        self.assertEqual(consumer["fxgraph"].get("fxgraph_cache_hit"), 1)
        self.assertNotIn("dynamo_cache_miss", consumer["dynamo_cache"])

    def test_fx_entry_leaves_the_aot_layer_out(self):
        # AOTAutogradCache is consulted before FxGraphCache. If it were active it
        # would report the hit and this entry would prove nothing.
        producer, consumer = self.pair("fx")
        self.assertEqual(producer["aot_autograd"].get("autograd_cache_miss", 0), 0)
        self.assertEqual(consumer["aot_autograd"].get("autograd_cache_hit", 0), 0)

    def test_fx_entry_persists_its_own_graph_entry(self):
        producer, _ = self.pair("fx")
        self.assertGreaterEqual(producer["cache_dirs"]["fxgraph"], 1)
        self.assertEqual(producer["cache_dirs"]["dynamo"], 0)

    # ------------------------------------------------------------------- aot
    def test_aot_entry_reuses_the_bundled_graphs(self):
        producer, consumer = self.pair("aot")
        self.assertTrue(producer["correct"])
        self.assertTrue(consumer["correct"])
        self.assertEqual(producer["aot_autograd"].get("autograd_cache_miss"), 1)
        self.assertEqual(consumer["aot_autograd"].get("autograd_cache_hit"), 1)

    def test_aot_entry_is_not_shadowed_by_a_package(self):
        _, consumer = self.pair("aot")
        self.assertNotIn("dynamo_cache_hit", consumer["dynamo_cache"])

    def test_aot_entry_persists_its_own_entry(self):
        producer, _ = self.pair("aot")
        self.assertGreaterEqual(producer["cache_dirs"]["aotautograd"], 1)
        self.assertEqual(producer["cache_dirs"]["dynamo"], 0)

    # ------------------------------------------------------- cross-entry guard
    def test_an_empty_cache_root_never_hits(self):
        # The control for every case above: on a fresh root the consumer behaves
        # like a producer, so a hit observed elsewhere cannot come from ambient
        # state left by another test.
        runner = self.runner()
        cold = runner.run("cold", "package", "consumer", cache=runner.root / "empty")
        self.assertEqual(cold["returncode"], 0, cold.get("error"))
        self.assertEqual(cold["result"]["dynamo_cache"].get("dynamo_cache_miss"), 1)
        self.assertNotIn("dynamo_cache_hit", cold["result"]["dynamo_cache"])

    # ---------------------------------------------------------------- backward
    def test_backward_reuses_the_package(self):
        # The backward graph travels inside the package: a consumer that runs
        # forward *and* backward must hit the package and produce the eager
        # gradients, which is what makes a lazy backward compilation reusable
        # by the process that never compiled it.
        runner = self.runner()
        cache = runner.root / "cache"
        producer = runner.run("producer", "package", "producer", cache=cache,
                              backward=True)
        consumer = runner.run("consumer", "package", "consumer", cache=cache,
                              backward=True)
        self.assertEqual(producer["returncode"], 0, producer.get("error"))
        self.assertEqual(consumer["returncode"], 0, consumer.get("error"))
        self.assertTrue(consumer["result"]["correct"])
        self.assertEqual(consumer["result"]["dynamo_cache"].get("dynamo_cache_hit"), 1)
        self.assertNotIn("dynamo_cache_miss", consumer["result"]["dynamo_cache"])

    def test_a_flipped_initialization_keeps_the_backward_correct(self):
        # An initialization-time knob is flipped between the forward and the
        # backward of the *producer*. The deferred backward must notice the
        # drift (the bypass counter names it) instead of reusing the forward's
        # identity for another cache save, and the consumer still restores a
        # package whose answer is right.
        runner = self.runner()
        cache = runner.root / "cache"
        producer = runner.run("producer", "package", "producer", cache=cache,
                              backward=True, flip_init=True)
        consumer = runner.run("consumer", "package", "consumer", cache=cache,
                              backward=True)
        self.assertEqual(producer["returncode"], 0, producer.get("error"))
        self.assertEqual(consumer["returncode"], 0, consumer.get("error"))
        self.assertTrue(producer["result"]["correct"])
        self.assertTrue(consumer["result"]["correct"])
        self.assertIn("flipped_init", producer["result"])
        self.assertIn(
            "NPU_INITIALIZATION_CONFIG_MISMATCH", producer["result"]["npu_package"])

    # ------------------------------------------------------------ backend scope
    def test_package_restores_under_a_foreign_backend(self):
        # The generated wrapper imports torch_npu._inductor, which loads
        # whichever backend the environment names. Restoring a DVM package
        # under an ambient ascendc must still run the DVM graphs: the restore
        # republishes the producer's backend instead of trusting the consumer's
        # environment.
        runner = self.runner()
        cache = runner.root / "cache"
        producer = runner.run("producer", "package", "producer", cache=cache)
        consumer = runner.run("consumer", "package", "consumer", cache=cache,
                              backend="ascendc")
        self.assertEqual(producer["returncode"], 0, producer.get("error"))
        self.assertEqual(consumer["returncode"], 0, consumer.get("error"))
        self.assertTrue(consumer["result"]["correct"])
        self.assertEqual(consumer["result"]["dynamo_cache"].get("dynamo_cache_hit"), 1)

    # --------------------------------------------------------- disk resilience
    def _asset_files(self, cache):
        return sorted((Path(cache) / "npu_fx_assets").glob("*/asset"))

    def test_package_restore_survives_a_deleted_fx_asset(self):
        # The FX asset bytes travel inside the package entry, so a consumer
        # can rewrite them when the cache root lost the file: deleting every
        # asset on disk must not turn the hit into a crash. The float64
        # workload is what makes DVM lower through an FX fallback at all.
        runner = self.runner()
        cache = runner.root / "cache"
        producer = runner.run("producer", "package", "producer", cache=cache,
                              fx_fallback=True)
        self.assertEqual(producer["returncode"], 0, producer.get("error"))
        assets = self._asset_files(cache)
        self.assertTrue(assets, "the producer should have written FX assets")
        for asset in assets:
            asset.unlink()

        consumer = runner.run("consumer", "package", "consumer", cache=cache,
                              fx_fallback=True)
        self.assertEqual(consumer["returncode"], 0, consumer.get("error"))
        self.assertTrue(consumer["result"]["correct"])
        self.assertEqual(consumer["result"]["dynamo_cache"].get("dynamo_cache_hit"), 1)

    def test_package_restore_survives_a_corrupted_fx_asset(self):
        # Corrupt bytes are replaced atomically from the package's own copy,
        # and the same-size-overwrite trap (a shared filesystem timestamp)
        # is avoided by comparing bytes, not mtimes.
        runner = self.runner()
        cache = runner.root / "cache"
        producer = runner.run("producer", "package", "producer", cache=cache,
                              fx_fallback=True)
        self.assertEqual(producer["returncode"], 0, producer.get("error"))
        assets = self._asset_files(cache)
        self.assertTrue(assets, "the producer should have written FX assets")
        for asset in assets:
            asset.write_bytes(b"\x00corrupted-not-an-asset")

        consumer = runner.run("consumer", "package", "consumer", cache=cache,
                              fx_fallback=True)
        self.assertEqual(consumer["returncode"], 0, consumer.get("error"))
        self.assertTrue(consumer["result"]["correct"])
        self.assertEqual(consumer["result"]["dynamo_cache"].get("dynamo_cache_hit"), 1)


if __name__ == "__main__":
    run_tests()
