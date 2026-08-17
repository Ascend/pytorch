# Owner(s): ["oncall: profiler"]

import json
import os
import tempfile
import glob
import gzip
from typing import Any
import torch_npu
import numpy as np
from torch_npu.testing.testcase import run_tests, TestCase
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
import torch
import torch.nn as nn
from torch.profiler import ExecutionTraceObserver
from torch.autograd.profiler import record_function
from unittest.mock import patch


worker_id = 1
Json = dict[str, Any]

class TestNpuExecutionTrace(TestCase):
    """
        NestedTensor Layout on Ascend is not supported yet.
    """
    def trace_root(self, out_file_name) -> Json:
        nodes = []
        with (
            gzip.open(out_file_name)
            if out_file_name.endswith(".gz")
            else open(out_file_name)
        ) as f:
            et_graph = json.load(f)
            if "nodes" not in et_graph:
                raise AssertionError(f"Missing 'nodes' in execution trace: {et_graph}")
            nodes = et_graph["nodes"]
        return nodes

    def workload(self):
        device = torch.device("npu:0")
        ut = torch.randn(3, 4, 5, requires_grad=True)
        with record_function("## TEST 1 ##", "1, 2, 3"):
            t1 = torch.randn(10, 10, device=device, requires_grad=True)
            t2 = torch.randn(10, 10, device=device, requires_grad=True)
            t3 = t1 + t2
            t3.backward(t3)
            gelu = nn.GELU()
            tm = torch.randn(2)
            _ = gelu(tm)
            t3 = t3.cpu()
            rec_fun_handler = _record_function_with_args_enter(
                "## TEST 2 ##",
                1,
                False,
                2.5,
                [ut, ut],
                (ut, ut),
                "hi",
                ut,
                float("inf"), float("-inf"), float("nan")
            )
            _record_function_with_args_exit(rec_fun_handler)

    @property
    def worker_name(self):
        global worker_id
        worker_name = f"npu_profiler_test{worker_id}"
        worker_id += 1
        return worker_name

    @patch.dict(
        os.environ,
        {"ENABLE_PYTORCH_EXECUTION_TRACE_SAVE_INTEGRAL_TENSOR_RANGE": "1"},
    )
    def test_npu_execution_trace_record_integral_tensor_range(self):
        device = torch.device("npu:0")
        x = torch.tensor([[1, 2], [3, 4]], device=device)
        y = torch.tensor([[0, 0], [1, 0]], device=device)
        with tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False) as trace_file:
            filename = trace_file.name
        et = ExecutionTraceObserver()
        et.register_callback(filename)
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(
                    skip_first=0, wait=0, warmup=0, active=1, repeat=1
                ),
                record_shapes=True,
                execution_trace_observer=et
        ) as prof:
            torch.gather(x, 1, y)
            prof.step()
        et.unregister_callback()
        nodes = self.trace_root(filename)
        os.remove(filename)
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            target_range = '{"0":[1,4],"1":[0,1]}'
            if "aten::gather" in n["name"]:
                for attr in n["attrs"]:
                    if attr["name"] == "tensor_range" and attr["value"] != target_range:
                        raise AssertionError(f"Expected tensor_range value to match {target_range}")

    def test_npu_execution_trace_record_integral_tensor_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fp_name = os.path.join(temp_dir, "test.et.json")

            os.environ["ENABLE_PYTORCH_EXECUTION_TRACE_SAVE_INTEGRAL_TENSOR_DATA"] = (
                "aten::gather"
            )
            et = ExecutionTraceObserver()
            et.register_callback(fp_name)
            et.set_extra_resource_collection(True)

            device = torch.device("npu:0")
            t1 = torch.tensor([[1, 2], [3, 4]], device=device)
            t2 = torch.tensor([[0, 0], [1, 0]], device=device)
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(
                    skip_first=0, wait=0, warmup=0, active=1, repeat=1
                ),
                record_shapes=True,
                execution_trace_observer=et,
            ) as p:
                torch.gather(t1, 1, t2)
                p.step()
            et.unregister_callback()

            resourceDir = fp_name.replace(".json", "_resources")
            dat_files = sorted(glob.glob(os.path.join(resourceDir, "*.dat")))
            if len(dat_files) < 2:
                raise AssertionError(f"Expected at least 2 .dat files in {resourceDir}, found {len(dat_files)}")

            dumped_t1 = np.fromfile(dat_files[0], dtype=np.int64)
            dumped_t2 = np.fromfile(dat_files[1], dtype=np.int64)

            if not (dumped_t1 == np.array([1, 2, 3, 4])).all():
                raise AssertionError("Expected t1 contents to match [1, 2, 3, 4]")
            if not (dumped_t2 == np.array([0, 0, 1, 0])).all():
                raise AssertionError("Expected t2 contents to match [0, 0, 1, 0]")

if __name__ == "__main__":
    run_tests()
