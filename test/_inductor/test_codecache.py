import pytest  # noqa: F401
import torch
from torch.testing._internal.common_utils import run_tests
from torch._inductor.codecache import CacheBase
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
import os
import tempfile


class TestCodeCache(TestUtils):

    def setUp(self):
        super().setUp()
        self._saved_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        self._saved_fx_cache = torch._inductor.config.fx_graph_cache
        self._tmpdir = tempfile.TemporaryDirectory()
        cache_dir = os.path.join(self._tmpdir.name, ".inductor_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        torch._inductor.config.fx_graph_cache = True

    def tearDown(self):
        if self._saved_cache_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = self._saved_cache_dir
        torch._inductor.config.fx_graph_cache = self._saved_fx_cache
        self._tmpdir.cleanup()
        super().tearDown()

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

    def test_fx_graph_cache_constant_device(self):
        """fx_graph_cache: tensor constants should remain on NPU after cache load.

        When map_location=None in torch_npu.utils.load, inductor-compiled
        functions loaded from cache should have their tensor constants on
        the correct device (NPU), not forced to CPU.
        """

        def fn(x):
            c = torch.tensor(list(range(15)), dtype=torch.int64)
            return x.index_select(0, c.to(x.device))

        x = torch.randn(16, device="npu")
        expected = torch.compile(fn, backend="inductor")(x)
        torch._dynamo.reset()
        actual = torch.compile(fn, backend="inductor")(x)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    run_tests()
