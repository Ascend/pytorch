import glob
import os
import re
import tempfile

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


PLOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hccl_logs")
# A2/A3 HCCL side: the QoS value the runtime actually loaded for a comm.
HCCL_LOAD_QOS_LOG_RE = re.compile(
    r"\[Load\] comm config of \[([^\]]*)\]:.*?hcclQos\[(\d+)\]")
# A5 HCCL side: the new funnel logs the applied value without a comm name.
HCCL_APPLY_QOS_LOG_RE = re.compile(r"\[ApplyHcclQos\] hcclQos\[(\d+)\]")
# A5 version gate: hcclQos is skipped when the comm config version is too old.
HCCL_SKIP_QOS_LOG_RE = re.compile(r"skip hcclQos by version")
# 0xffffffff means "not configured", kept from HcclCommConfigInit.
HCCL_QOS_NOT_SET = 0xffffffff
# PTA-side migration warning emitted when both keys are present.
DEPRECATION_WARN_RE = re.compile(r"hccl_sdma_qos is no longer recommended")


def _worker_entry(fn, stderr_path, args):
    r"""Redirect worker fd 2 to a file so PTA warnings can be asserted by the parent."""
    import sys
    if stderr_path:
        fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(fd, 2)
        os.close(fd)
        sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
    fn(*args)


class PlogCapture:
    r"""Route worker plog to a per-test dir; env must be set before spawn.
    Level 1 keeps HCCL's "[Load] comm config" line."""

    def __init__(self, label):
        self.label = label
        self.path = None
        self._saved_env = {}

    def __enter__(self):
        os.makedirs(PLOG_ROOT, exist_ok=True)
        self.path = tempfile.mkdtemp(prefix=f"plog_{self.label}_", dir=PLOG_ROOT)
        # Some CANN versions prefer ASCEND_WORK_PATH; set both to the same root.
        for key, value in (("ASCEND_GLOBAL_LOG_LEVEL", "1"),
                           ("ASCEND_PROCESS_LOG_PATH", self.path),
                           ("ASCEND_WORK_PATH", self.path)):
            self._saved_env[key] = os.environ.get(key)
            os.environ[key] = value
        self._saved_env["ASCEND_SLOG_PRINT_TO_STDOUT"] = os.environ.pop(
            "ASCEND_SLOG_PRINT_TO_STDOUT", None)
        return self

    def __exit__(self, *args):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False

    def text_for_pid(self, pid):
        chunks = []
        pattern = os.path.join(self.path, "**", f"plog-{pid}*.log")
        for log_path in sorted(glob.glob(pattern, recursive=True)):
            with open(log_path, "r", errors="replace") as f:
                chunks.append(f.read())
        return "".join(chunks)

    def stderr_path(self, rank):
        return os.path.join(self.path, f"stderr_rank{rank}.log")

    @staticmethod
    def parse(text):
        r"""Return (named, applied).
        named:   {comm_name: value}, only on A2/A3 (has the comm name).
        applied: [value, ...], only on A5 (the new funnel drops the comm name)."""
        named = {}
        for comm_name, value in HCCL_LOAD_QOS_LOG_RE.findall(text):
            named[comm_name] = int(value)
        applied = [int(value) for value in HCCL_APPLY_QOS_LOG_RE.findall(text)]
        return named, applied


class HcclQosTest(TestCase):
    r"""Test per-communication-domain HCCL QoS via hccl_config: hccl_qos wins over
    the legacy hccl_sdma_qos, which is then warned about on stderr."""

    @classmethod
    def _init_dist_hccl(cls, rank, options, world_size, seq):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(seq + 29500)
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', pg_options=options, world_size=world_size, rank=rank)

    @classmethod
    def _test_qos_all_reduce(cls, rank, ranks, world_size, input1, hccl_config):
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = hccl_config
        HcclQosTest._init_dist_hccl(rank, options, world_size, seq=10)

        dist.all_reduce(input1.npu(rank))

    @classmethod
    def _test_qos_multi_group(cls, rank, ranks, world_size, input1, input2):
        # Default group (new key) and a conflict new_group must stay isolated;
        # repeated collectives must not amplify the per-group warning.
        options1 = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options1.hccl_config = {"hccl_qos": 7}
        HcclQosTest._init_dist_hccl(rank, options1, world_size, seq=20)

        dist.all_reduce(input1.npu(rank))

        options2 = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options2.hccl_config = {"hccl_qos": 5, "hccl_sdma_qos": 5}
        pg = dist.new_group(backend='hccl', ranks=ranks, pg_options=options2)
        y = input2.npu(rank)
        for _ in range(2):
            dist.all_reduce(y, group=pg)

        dist.destroy_process_group(pg)
        dist.destroy_process_group()

    @classmethod
    def _test_options_wrong_value(cls, rank, ranks, world_size, input1, hccl_config, error_expect):
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = hccl_config
        try:
            HcclQosTest._init_dist_hccl(rank, options, world_size, seq=40)
            dist.all_reduce(input1.npu(rank))
        except RuntimeError as e:
            assert error_expect in str(e), f"expected {error_expect!r}, got: {e}"
        else:
            raise RuntimeError(f"expected {error_expect!r}, but init succeeded")

    def _test_multiprocess(self, f, world_size, *args):
        ctx = mp.get_context('spawn')
        # create_common_tensor returns NPU tensors; pass CPU copies to avoid aliasing.
        cpu_args = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]

        ps = []
        rank_pids = []
        with PlogCapture(label=f.__name__) as plog:
            for rank in range(world_size):
                p = ctx.Process(
                    target=_worker_entry,
                    args=(f, plog.stderr_path(rank),
                          (rank, range(world_size), world_size, *cpu_args)))
                p.start()
                ps.append(p)
                rank_pids.append((rank, p.pid))

            for p in ps:
                p.join()

            for p in ps:
                self.assertEqual(p.exitcode, 0)

            evidence = {}
            for rank, pid in rank_pids:
                text = plog.text_for_pid(pid)
                stderr_path = plog.stderr_path(rank)
                stderr_text = ""
                if os.path.exists(stderr_path):
                    with open(stderr_path, "r", errors="replace") as fh:
                        stderr_text = fh.read()
                named, applied = PlogCapture.parse(text)
                evidence[rank] = (named, applied, text, stderr_text, plog.path)
            return evidence

    def _assert_plog_qos(self, evidence, expected, expect_warnings=None):
        r"""HCCL plog must show the hcclQos loaded for each expected comm.
        On A5 the comm name is not logged, so the configured values are matched
        as a subset of the applied values instead.
        expect_warnings: exact number of migration warnings each rank must emit."""
        for rank in sorted(evidence):
            named, applied, text, stderr_text, log_dir = evidence[rank]
            rank_msg = f"rank {rank} (plog: {log_dir})"
            expected_values = sorted(set(expected.values()))
            if named:
                for comm_name, value in expected.items():
                    self.assertIn(
                        comm_name, named,
                        f"{rank_msg}: HCCL did not load a config for [{comm_name}], "
                        f"comms seen in plog: {sorted(named)}")
                    self.assertEqual(
                        named[comm_name], value,
                        f"{rank_msg}: HCCL loaded hcclQos[{named[comm_name]}] for "
                        f"[{comm_name}], expected {value}")
            elif applied:
                missing = [value for value in expected_values if value not in applied]
                self.assertEqual(
                    missing, [],
                    f"{rank_msg}: A5 applied hcclQos values {sorted(set(applied))} "
                    f"do not cover the expected {expected_values}")
            elif HCCL_SKIP_QOS_LOG_RE.search(text):
                self.fail(
                    f"{rank_msg}: A5 skipped hcclQos by comm config version, "
                    f"which is not a passthrough failure")
            else:
                self.fail(
                    f"{rank_msg}: no HCCL QoS evidence found (neither "
                    f"'[Load] comm config of' nor '[ApplyHcclQos]'); log dir: {log_dir}")
            if expect_warnings is not None:
                count = len(DEPRECATION_WARN_RE.findall(stderr_text))
                self.assertEqual(
                    count, expect_warnings,
                    f"{rank_msg}: expected {expect_warnings} migration warning(s), "
                    f"got {count}; stderr was:\n{stderr_text[:2000]}")

    def _assert_no_qos_evidence(self, evidence):
        r"""No HCCL QoS value may be applied after an invalid config value."""
        for rank in sorted(evidence):
            named, applied, _, _, log_dir = evidence[rank]
            self.assertNotIn(
                "group_name_0", named,
                f"rank {rank}: HCCL loaded a config after an invalid value "
                f"(plog: {log_dir})")
            self.assertEqual(
                applied, [],
                f"rank {rank}: A5 applied hcclQos {applied} after an "
                f"invalid value (plog: {log_dir})")

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_qos_config_values(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        # (hccl_config, expected hcclQos in plog, migration warnings per rank)
        cases = (
            ({"hccl_qos": 7}, 7, 0),
            ({"hccl_qos": 0}, 0, 0),
            ({"hccl_sdma_qos": 7}, 7, 0),
            ({"hccl_sdma_qos": 0}, 0, 0),
            ({"hccl_qos": 7, "hccl_sdma_qos": 0}, 7, 1),
            ({"hccl_qos": 5, "hccl_sdma_qos": 5}, 5, 1),
            ({"hccl_buffer_size": 300}, HCCL_QOS_NOT_SET, 0),
        )
        for hccl_config, expected, warnings in cases:
            _, input1 = create_common_tensor(shape, -10, 10)
            evidence = self._test_multiprocess(
                HcclQosTest._test_qos_all_reduce, world_size, input1, hccl_config)
            self._assert_plog_qos(evidence, {"group_name_0": expected},
                                  expect_warnings=warnings)

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_qos_wrong_type(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        _, input1 = create_common_tensor(shape, -10, 10)
        evidence = self._test_multiprocess(
            HcclQosTest._test_options_wrong_value, world_size, input1,
            {"hccl_qos": "1"}, "Value type of hccl_qos should be uint.")
        self._assert_no_qos_evidence(evidence)

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_qos_multi_group(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        _, input1 = create_common_tensor(shape, -10, 10)
        _, input2 = create_common_tensor(shape, -10, 10)
        evidence = self._test_multiprocess(
            HcclQosTest._test_qos_multi_group, world_size, input1, input2)
        # Default group uses the new key (7); the conflict new_group takes 5
        # and warns exactly once despite two collectives.
        self._assert_plog_qos(evidence, {"group_name_0": 7, "group_name_1": 5},
                              expect_warnings=1)


if __name__ == '__main__':
    run_tests()
