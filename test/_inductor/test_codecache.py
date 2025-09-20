import pytest
import torch
from torch.testing._internal.common_utils import run_tests
from torch._inductor.codecache import CacheBase
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class TestCodeCache(TestUtils):
    def test_codecache(self):
        device_properties = torch_npu.npu.get_device_properties(
            torch_npu.npu.current_device()
        )

        system1 = CacheBase.get_system()
        self.assertEqual(system1["device"]["name"], device_properties.name)
        self.assertEqual(system1["version"]["cann"], torch.version.cann)

        from torch_npu.contrib import transfer_to_npu
        system2 = CacheBase.get_system()
        self.assertEqual(system2["device"]["name"], device_properties.name)
        self.assertEqual(system2["version"]["cann"], torch.version.cann)


if __name__ == "__main__":
    run_tests()