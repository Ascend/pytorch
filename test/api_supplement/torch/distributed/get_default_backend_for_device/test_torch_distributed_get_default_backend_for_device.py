"""
Test purpose: Verify get_default_backend_for_device behavior.
API name: torch.distributed.get_default_backend_for_device
API signature: get_default_backend_for_device(device)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Device argument forms | Resolve CPU, accelerator and indexed device inputs. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank accelerator process group on one host.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_device_resolution(case, rank, device):
    import torch.distributed as dist

    for spec in (device, device.type, str(device)):
        case.assertEqual(dist.get_default_backend_for_device(spec), dist.get_backend())
    case.assertEqual(dist.get_default_backend_for_device("cpu"), "gloo")


class TestGetDefaultBackendForDevice(TwoRankTestCase):
    def test_get_default_backend_for_device_device_resolution(self):
        self.run_on_two_ranks(check_device_resolution)


if __name__ == "__main__":
    run_tests()
