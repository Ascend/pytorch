"""
Test purpose: Verify get_backend_config behavior.
API name: torch.distributed.get_backend_config
API signature: get_backend_config(group=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Device and backend mapping | Check the explicit device/backend mapping for the default group and a subgroup. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank accelerator process group on one host.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_backend_mapping(case, rank, device):
    import torch.distributed as dist

    group = dist.new_group([0, 1])
    try:
        for target in (None, dist.group.WORLD, group):
            mapping = dict(item.split(":") for item in dist.get_backend_config(target).split(","))
            case.assertEqual(mapping[device.type], dist.get_backend(group))
    finally:
        dist.destroy_process_group(group)


class TestGetBackendConfig(TwoRankTestCase):
    def test_get_backend_config_backend_mapping(self):
        self.run_on_two_ranks(check_backend_mapping)


if __name__ == "__main__":
    run_tests()
