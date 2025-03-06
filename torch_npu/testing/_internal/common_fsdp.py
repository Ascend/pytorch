import torch
import torch.distributed as dist
from torch.testing._internal.common_fsdp import FSDPTest

import torch_npu

torch.testing._internal.common_fsdp.DEVICE_TYPE = "npu"
torch.testing._internal.common_fsdp.DISTRIBUTED_BACKEND = "hccl"
torch.testing._internal.common_fsdp.DEVICE_COUNT = torch.npu.device_count()


class FSDPNPUTest(FSDPTest):
    @classmethod
    def _run(cls, rank, test_name, file_name, pipe, **kwargs):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        fake_pg = kwargs.get("fake_pg", False)

        print(f"dist init r={self.rank}, world={self.world_size}")

        # Specify gloo backend to make 'init_process_group()' succeed,
        # Actual tests will be skipped if there is no enough GPUs.
        try:
            if fake_pg:
                store = torch.testing._internal.distributed.fake_pg.FakeStore()
                dist.init_process_group(
                    backend="fake",
                    world_size=self.world_size,
                    rank=rank,
                    store=store,
                )
            else:
                dist.init_process_group(
                    init_method=self.init_method,
                    backend="hccl",
                    world_size=int(self.world_size),
                    rank=self.rank,
                )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        device_ids = None
        device_id = self.rank % torch.npu.device_count()
        torch.npu.set_device(device_id)
        device_ids = [device_id]

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier(device_ids=device_ids)

        torch._dynamo.reset()
        self.run_test(test_name, pipe)
        torch._dynamo.reset()

        dist.barrier(device_ids=device_ids)

        dist.destroy_process_group()
