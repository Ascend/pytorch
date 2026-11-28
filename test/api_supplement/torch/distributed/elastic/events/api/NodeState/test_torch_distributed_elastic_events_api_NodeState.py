"""
Test purpose: Verify NodeState behavior.
API name: torch.distributed.elastic.events.api.NodeState
API signature: NodeState(value)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Enum values and invalid input | Round-trip each serialized node state and reject unknown event state strings. |

Execution: Single-process unittest.
Scope and prerequisites:
- This is a host-side value object. Device, gradient and mixed-device behavior are inapplicable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestNodeState(SupplementTestCase):
    def test_node_state_event_states(self):
        from torch.distributed.elastic.events.api import NodeState

        for name in ("INIT", "RUNNING", "SUCCEEDED", "FAILED"):
            self.assertIs(NodeState(name), NodeState[name])
            self.assertEqual(NodeState[name].value, name)
        with self.assertRaises(ValueError):
            NodeState("UNKNOWN")


if __name__ == "__main__":
    run_tests()
