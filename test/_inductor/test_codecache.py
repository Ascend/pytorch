# lintrunner: skip PYFMT
# Owner(s): ["module: inductor"]
"""Module for inductor codecache tests."""

import os
import shutil
import unittest

import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._inductor import config
from torch._inductor.codecache import CacheBase, PyCodeCache
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches, fresh_cache
from torch.compiler._cache import CacheArtifactManager
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import requires_triton


DEVICES = ("npu", "cpu")


@instantiate_parametrized_tests
class TestCachingPrecompileCodeCache(TestCase):
    def setUp(self):
        super().setUp()
        counters.clear()
        DynamoCache.clear()
        PrecompileContext.clear()
        AOTAutogradCache.clear()
        CacheArtifactManager.clear()
        torch._dynamo.reset()

    def reset(self):
        AOTAutogradCache.clear()
        DynamoCache.clear()
        PrecompileContext.clear()
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()

    @requires_triton()
    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @torch._dynamo.config.patch({"caching_precompile": True})
    @parametrize("dynamic", (False, True))
    @parametrize("device", DEVICES)
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_cache_hot_load_caching_precompile(self, device, dtype, dynamic):
        if device == "npu" and not torch.npu.is_available():
            raise unittest.SkipTest("Requires NPU")

        def fn(x, y):
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)
        b = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)

        # Record artifacts.
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 1)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 0)

        artifacts = torch.compiler.save_cache_artifacts()
        self.assertEqual(artifacts is not None, True)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 1 if device == "npu" else 0
        self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)
        self.assertEqual(len(cache_info.precompile_artifacts), 1)

        self.reset()
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Without loading artifacts, a new compile should not hit dynamo cache.
        with fresh_cache():
            eager_result = fn(a, b)
            compiled_fn = torch.compile(fn, dynamic=dynamic)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 2)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 0)

        self.reset()
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # After loading artifacts, recompilation is forbidden and dynamo cache hits.
        with fresh_cache(), torch.compiler.set_stance("fail_on_recompile"):
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
            self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)
            self.assertEqual(len(cache_info.precompile_artifacts), 1)

            compiled_fn = torch.compile(fn, dynamic=dynamic)
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 2)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 1)


class TestCodeCache(TestCase):
    def test_codecache(self):
        device_properties = torch_npu.npu.get_device_properties(
            torch_npu.npu.current_device()
        )

        system1 = CacheBase.get_system()
        self.assertEqual(system1["device"]["name"], device_properties.name)
        self.assertEqual(system1["version"]["cann"], torch.version.cann)

        from torch_npu.contrib import transfer_to_npu  # noqa: F401

        system2 = CacheBase.get_system()
        self.assertEqual(system2["device"]["name"], device_properties.name)
        self.assertEqual(system2["version"]["cann"], torch.version.cann)


if __name__ == "__main__":
    run_tests()
