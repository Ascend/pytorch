# Owner(s): ["module: dynamo"]
import functools
import unittest
from unittest.mock import patch
import torch
# for some reason importing functional collectives after dynamo breaks collectives handling!
import torch.distributed._functional_collectives as _functional_collectives
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.utils import same
from torch.testing._internal.common_distributed import (
    DynamoDistributedMultiProcTestCase,
)
import torch._dynamo.logging
import torch_npu
from torch_npu.testing.common_distributed import (
    skipIfUnsupportMultiNPU,
    _dynamo_dist_per_rank_init,
)


class TestCollectivesMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Run correctness checks in multi-proc runner, mark with minimum # NPUs to run under
    """
    def get_world_trs(self):
        return {
            "tag": "",
            "ranks": list(range(self.world_size)),
            "group_size": self.world_size,
        }

    @property
    def world_size(self) -> int:
        # hack: no matter whether we have 2 or 3 or 4 npus, just run on 2
        # works around issue with skipif<2 and workers with unpredictable #s npu
        return 2

    @skipIfUnsupportMultiNPU(2)
    def test_permute_tensor(self):
        def func(tensor, src_dst_pairs, *, tag, ranks, group_size):
            return _functional_collectives.permute_tensor(tensor, src_dst_pairs, ranks, tag)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            inputs = (
                # rank0: [0., 1.], rank1: [2., 3.]
                torch.arange(2, dtype=torch.float32, device="npu") + 2 * self.rank,
                [1, 0],
            )
            compiled = torch.compile(func)
            out = compiled(*inputs, **self.get_world_trs())
            correct = func(*inputs, **self.get_world_trs())
            self.assertTrue(same(out, correct))

            # rank0: [2., 3.], rank1: [0., 1.]
            expected = torch.arange(
                2,
                dtype=torch.float32,
                device="npu"
            ) + 2 * ((self.rank - 1 + self.world_size) % self.world_size)
            self.assertEqual(out, expected)
            self.assertEqual(correct, expected)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()
