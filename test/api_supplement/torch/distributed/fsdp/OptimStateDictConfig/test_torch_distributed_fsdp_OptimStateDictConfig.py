"""
Test purpose: Verify OptimStateDictConfig behavior.
API name: torch.distributed.fsdp.OptimStateDictConfig
API signature: OptimStateDictConfig(offload_to_cpu=True)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Optimizer offload options | Check defaults, overrides and instance independence. |

Execution: Single-process unittest.
Scope and prerequisites:
- This is a host-side value object. Device, gradient and mixed-device behavior are inapplicable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestOptimStateDictConfig(SupplementTestCase):
    def test_optim_state_dict_config_offload_configuration(self):
        from torch.distributed.fsdp import OptimStateDictConfig

        default = OptimStateDictConfig()
        self.assertIs(default.offload_to_cpu, True)
        for enabled in (False, True):
            explicit = OptimStateDictConfig(offload_to_cpu=enabled)
            self.assertIs(explicit.offload_to_cpu, enabled)
            self.assertIs(default.offload_to_cpu, True)


if __name__ == "__main__":
    run_tests()
