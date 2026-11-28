"""
Test purpose: Verify record_rdzv_event behavior.
API name: torch.distributed.elastic.events.record_rdzv_event
API signature: record_rdzv_event(event)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Event routing and serialization | Check the destination and serialized event delivered to the rendezvous logger. |

Execution: Single-process unittest.
Scope and prerequisites:
- The logger is intercepted at its boundary, so no persistent logging configuration is changed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestRecordRdzvEvent(SupplementTestCase):
    def test_record_rdzv_event_serialized_event(self):
        import json
        from unittest.mock import patch

        from torch.distributed.elastic.events import record_rdzv_event
        from torch.distributed.elastic.events.api import NodeState, RdzvEvent

        event = RdzvEvent(
            name="join",
            run_id="job",
            message="ready",
            hostname="host",
            pid=42,
            node_state=NodeState.RUNNING,
        )
        with patch("torch.distributed.elastic.events._get_or_create_logger") as destination:
            record_rdzv_event(event)
            destination.assert_called_once_with("dynamic_rendezvous")
            (payload,) = destination.return_value.info.call_args.args
            self.assertEqual(json.loads(payload), json.loads(event.serialize()))
            self.assertEqual(json.loads(payload)["node_state"], "RUNNING")


if __name__ == "__main__":
    run_tests()
