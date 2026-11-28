"""
Test purpose: Verify always_wrap_policy behavior.
API name: torch.distributed.fsdp.wrap.always_wrap_policy
API signature: always_wrap_policy(*args, **kwargs)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Recursion and wrapping decisions | Check both policy decisions on modules with and without parameters. |

Execution: Single-process unittest.
Scope and prerequisites:
- This host-side contract has no device or gradient inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestAlwaysWrapPolicy(SupplementTestCase):
    def test_always_wrap_policy_recursive_wrap_policy(self):
        from torch import nn
        from torch.distributed.fsdp.wrap import always_wrap_policy

        for module in (nn.Identity(), nn.Linear(2, 3)):
            for recurse in (False, True):
                self.assertIs(
                    always_wrap_policy(
                        module=module,
                        recurse=recurse,
                        nonwrapped_numel=sum(p.numel() for p in module.parameters()),
                    ),
                    True,
                )


if __name__ == "__main__":
    run_tests()
