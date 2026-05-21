# Owner(s): ["oncall: distributed"]

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestFakeDTensor(TestCase):
    def test_fake_collectives(self):
        # Ensure that we can run (non-functional) collectives under FakeTensorMode.
        # This requires the meta impls for non-functional collectives
        # to be registered at impor
        fake_mode = FakeTensorMode()
        world_size = 4

        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=world_size
        )
        default_pg = torch.distributed.distributed_c10d._get_default_group()
        with fake_mode:
            x = torch.randn(2, 2, device="npu")
            torch.distributed.all_reduce(x, group=default_pg)


if __name__ == "__main__":
    run_tests()