# Owner(s): ["oncall: distributed"]

import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


DUMP_PREFIX = "comm_lib_trace_rank_"


class FlightRecorderCliTest(unittest.TestCase):
    def _write_dumps(self, trace_dir: Path) -> None:
        entry = {
            "record_id": 0,
            "process_group": ("0", "default_pg"),
            "collective_seq_id": 1,
            "p2p_seq_id": 0,
            "is_p2p": False,
            "profiling_name": "hccl:all_reduce",
            "time_created_ns": 1,
            "input_sizes": [[4]],
            "output_sizes": [[4]],
            "input_dtypes": ["Float"],
            "output_dtypes": ["Float"],
            "state": "completed",
            "frames": [],
        }
        dump = {
            "version": "2.10",
            "entries": [entry],
            "pg_config": {
                "0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}
            },
            "comm_lib_version": "HCCL test",
        }
        for rank in (0, 1):
            with (trace_dir / f"{DUMP_PREFIX}{rank}").open("wb") as output:
                pickle.dump(dump, output)


    def test_output_pickle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "rank_files"
            trace_dir.mkdir()
            self._write_dumps(trace_dir)
            output_path = Path(temp_dir) / "output.pkl"

            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-m",
                    "torch.distributed.flight_recorder.fr_trace",
                    str(trace_dir),
                    "--prefix",
                    DUMP_PREFIX,
                    "-o",
                    str(output_path),
                ],
                cwd=temp_dir,
                env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"),
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with patch.dict(os.environ, {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}):
                with output_path.open("rb") as output:
                    _, database = pickle.load(output)
            self.assertEqual(
                type(database).__module__,
                "torch.distributed.flight_recorder.components.types",
            )
            self.assertEqual(type(database).__name__, "Database")
            self.assertEqual(len(database.collectives), 1)
            collective = database.collectives[0]
            self.assertEqual(collective.collective_name, "hccl:all_reduce")
            self.assertTrue(collective.pass_check)
            self.assertCountEqual(
                [
                    (call.global_rank, call.collective_id, call.collective_type)
                    for call in database.ncclcalls
                ],
                [(0, collective.id, "hccl:all_reduce"), (1, collective.id, "hccl:all_reduce")],
            )


if __name__ == "__main__":
    unittest.main()
