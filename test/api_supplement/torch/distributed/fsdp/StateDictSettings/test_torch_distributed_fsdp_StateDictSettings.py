"""
Test purpose: Verify StateDictSettings behavior.
API name: torch.distributed.fsdp.StateDictSettings
API signature: StateDictSettings(state_dict_type, state_dict_config, optim_state_dict_config)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Paired state configurations | Check the state type and model and optimizer config references. |

Execution: Single-process unittest.
Scope and prerequisites:
- This is a host-side value object. Device, gradient and mixed-device behavior are inapplicable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestStateDictSettings(SupplementTestCase):
    def test_state_dict_settings_paired_configs(self):
        from torch.distributed.fsdp import (
            LocalOptimStateDictConfig,
            LocalStateDictConfig,
            StateDictSettings,
            StateDictType,
        )

        for offload in (False, True):
            model = LocalStateDictConfig(offload_to_cpu=offload)
            optimizer = LocalOptimStateDictConfig(offload_to_cpu=not offload)
            settings = StateDictSettings(StateDictType.LOCAL_STATE_DICT, model, optimizer)
            self.assertIs(settings.state_dict_type, StateDictType.LOCAL_STATE_DICT)
            self.assertIs(settings.state_dict_config, model)
            self.assertIs(settings.optim_state_dict_config, optimizer)


if __name__ == "__main__":
    run_tests()
