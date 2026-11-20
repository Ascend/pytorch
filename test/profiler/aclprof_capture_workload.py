"""Check a real NPU MatMul with Linux, CANN (including msprof), and matching wheels.

Run with the CANN environment active and a new evidence directory:
python3.11 -I test/profiler/aclprof_capture_workload.py --evidence /tmp/aclprof-acceptance-unique
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="New directory for the offline-analysis log")
    evidence = parser.parse_args().evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=False)

    import torch
    import torch_npu

    x = torch.ones((128, 128), device="npu")
    torch.npu.synchronize()
    handler = torch_npu.profiler.tensorboard_trace_handler(str(evidence), analyse_flag=False)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.PrivateUse1],
        experimental_config=torch_npu.profiler._NPUExperimentalConfig(backend="aclprof"),
        on_trace_ready=handler,
    ):
        torch.mm(x, x)

    output = torch_npu._C._profiler.GetAclprofOutputPath(handler.prof_path)
    if not output:
        raise RuntimeError("ACLprof did not complete the capture")
    root = Path(output).resolve()
    script = "from torch_npu.profiler.profiler import analyse; import sys; analyse(sys.argv[1])"
    with (evidence / "analysis.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-I", "-c", script, str(root)], stdout=log, stderr=subprocess.STDOUT, check=True)

    trace = root / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    events = json.loads(trace.read_text(encoding="utf-8"))
    if not isinstance(events, list) or not any(
        event.get("ph") == "X"
        and event.get("args", {}).get("Task Type")
        and float(event.get("dur", 0)) > 0
        and "matmul" in event.get("name", "").lower()
        for event in events
    ):
        raise RuntimeError(f"No positive-duration NPU MatMul activity in {trace}")
    print(f"ACLprof capture passed; trace: {trace}; evidence: {evidence}")


if __name__ == "__main__":
    main()
