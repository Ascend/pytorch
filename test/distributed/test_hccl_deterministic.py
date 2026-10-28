import glob
import os
import re
import tempfile

import numpy as np

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


PLOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hccl_logs")
# HCCL side: value the runtime actually loaded for a comm.
HCCL_LOAD_CONFIG_LOG_RE = re.compile(
    r"\[Load\] comm config of \[([^\]]*)\]:.*?bufferSize\[\d+\], "
    r"deterministic\[(\d+)\]")
# HCCL side: no op_option, so HCCL fell back to env/default.
HCCL_ENV_FALLBACK_LOG_RE = re.compile(
    r"The hcclDeterministic is not configured, use the env config \[(\d+)\] as default")


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
        # Do not mirror the logs to stdout/stderr; the assertion is on plog.
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

    @staticmethod
    def parse(text):
        hccl = {}
        for comm_name, value in HCCL_LOAD_CONFIG_LOG_RE.findall(text):
            hccl[comm_name] = int(value)
        return hccl


class HcclDeterministicTest(TestCase):
    r"""Test per-communication-domain hcclDeterministic via hccl_config:
    op_option > env HCCL_DETERMINISTIC > HCCL default (resolved by HCCL)."""

    @classmethod
    def _init_dist_hccl(cls, rank, options, world_size, seq):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(seq + 29500)
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', pg_options=options, world_size=world_size, rank=rank)

    @classmethod
    def _test_deterministic_all_reduce(cls, rank, ranks, world_size, input1, deterministic):
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_config = {"hccl_deterministic": deterministic}
        options.hccl_config = hccl_config
        HcclDeterministicTest._init_dist_hccl(rank, options, world_size, seq=deterministic)

        original = input1
        input1 = input1.npu(rank)
        dist.all_reduce(input1)

        default_pg = c10d._get_default_group()._get_backend(torch.device(f'npu:{rank}'))
        test_case = TestCase()
        test_case.assertEqual(default_pg.options.hccl_config.get("hccl_deterministic", -1), deterministic,
                              "hccl_deterministic config value mismatch")
        test_case.assertEqual(input1.cpu(), world_size * original,
                              "all_reduce result is incorrect with hccl_deterministic = {}.".format(deterministic))

    @classmethod
    def _test_deterministic_multi_group(cls, rank, ranks, world_size, input1, input2):
        # Default group (1) and new_group (0) on the same devices must stay isolated.
        options1 = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options1.hccl_config = {"hccl_deterministic": 1}
        HcclDeterministicTest._init_dist_hccl(rank, options1, world_size, seq=30)

        original1 = input1
        original2 = input2
        input1 = input1.npu(rank)
        input2 = input2.npu(rank)
        dist.all_reduce(input1)

        options2 = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options2.hccl_config = {"hccl_deterministic": 0}
        pg = dist.new_group(backend='hccl', ranks=ranks, pg_options=options2)
        dist.all_reduce(input2, group=pg)

        test_case = TestCase()
        default_backend = c10d._get_default_group()._get_backend(torch.device(f'npu:{rank}'))
        pg_backend = pg._get_backend(torch.device(f'npu:{rank}'))
        test_case.assertEqual(default_backend.options.hccl_config.get("hccl_deterministic", -1), 1,
                              "hccl_deterministic of default group should keep 1 after new_group")
        test_case.assertEqual(pg_backend.options.hccl_config.get("hccl_deterministic", -1), 0,
                              "hccl_deterministic of new group should be 0")
        test_case.assertEqual(input1.cpu(), world_size * original1,
                              "all_reduce on default group (deterministic=1) is incorrect.")
        test_case.assertEqual(input2.cpu(), world_size * original2,
                              "all_reduce on new group (deterministic=0) is incorrect.")

        dist.destroy_process_group(pg)
        dist.destroy_process_group()

    @classmethod
    def _test_options_wrong_value(cls, rank, hccl_config, error_expect, world_size, input1):
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = hccl_config
        input1 = input1.npu(rank)
        test_case = TestCase()
        with test_case.assertRaises(RuntimeError) as cm:
            HcclDeterministicTest._init_dist_hccl(rank, options, world_size, seq=40)
            dist.all_reduce(input1)

        test_case.assertTrue(
            error_expect in str(cm.exception),
            f"Expected error messages '{error_expect}' not found in actual error: {str(cm.exception)}")

    @classmethod
    def _test_hccl_deterministic_wrong_type(cls, rank, ranks, world_size, input1):
        HcclDeterministicTest._test_options_wrong_value(
            rank, {"hccl_deterministic": "1"}, "Value type of hccl_deterministic should be int.",
            world_size, input1)

    @classmethod
    def _test_deterministic_env_only(cls, rank, ranks, world_size, input1):
        # No hccl_config: HCCL resolves HCCL_DETERMINISTIC or its default itself.
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        HcclDeterministicTest._init_dist_hccl(rank, options, world_size, seq=80)

        original = input1
        input1 = input1.npu(rank)
        dist.all_reduce(input1)

        test_case = TestCase()
        test_case.assertEqual(input1.cpu(), world_size * original,
                              "all_reduce result is incorrect when hccl_deterministic is unset.")

    @classmethod
    def _test_deterministic_other_config_only(cls, rank, ranks, world_size, input1):
        # Non-empty hccl_config without the key must leave HCCL resolution untouched.
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {"hccl_buffer_size": 300}
        HcclDeterministicTest._init_dist_hccl(rank, options, world_size, seq=90)

        original = input1
        input1 = input1.npu(rank)
        dist.all_reduce(input1)

        test_case = TestCase()
        test_case.assertEqual(input1.cpu(), world_size * original,
                              "all_reduce result is incorrect with other hccl_config keys.")

    @classmethod
    def _test_deterministic_comm_creation_paths(cls, rank, ranks, world_size, input1):
        # The two comm-creation paths collective() never takes: DEFAULT + nullptr and P2P.
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {"hccl_deterministic": 2}
        HcclDeterministicTest._init_dist_hccl(rank, options, world_size, seq=70)

        backend = c10d._get_default_group()._get_backend(torch.device(f'npu:{rank}'))
        backend.get_hccl_comm(rank)                     # path 1)

        original = input1
        tensor = input1.npu(rank)
        if rank == 0:
            dist.send(tensor, dst=1)                    # path 2)
        else:
            dist.recv(tensor, src=0)

        test_case = TestCase()
        # path 1) the config must survive the commConfig == nullptr creation path
        test_case.assertEqual(backend.options.hccl_config.get("hccl_deterministic", -1), 2,
                              "hccl_deterministic must be kept on the commConfig==nullptr creation path")
        # path 2) the P2P communicator must still be built and the payload transferred
        test_case.assertEqual(tensor.cpu(), original,
                              "P2P send/recv payload mismatch with hccl_config present.")
        dist.destroy_process_group()

    @classmethod
    def _test_deterministic_two_subdomains(cls, rank, ranks, world_size, input1, input2, input3):
        # Three comm domains on one node: default(0..7)->1, A(0..3)->0, B(4..7)->2.
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {"hccl_deterministic": 1}
        HcclDeterministicTest._init_dist_hccl(rank, options, world_size, seq=60)

        original1 = input1
        original2 = input2
        original3 = input3

        t1 = input1.npu(rank)
        dist.all_reduce(t1)

        group_a = list(range(0, 4))
        group_b = list(range(4, 8))

        opts_a = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        opts_a.hccl_config = {"hccl_deterministic": 0}
        pg_a = dist.new_group(backend='hccl', ranks=group_a, pg_options=opts_a)

        opts_b = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        opts_b.hccl_config = {"hccl_deterministic": 2}
        pg_b = dist.new_group(backend='hccl', ranks=group_b, pg_options=opts_b)

        test_case = TestCase()
        device = torch.device(f'npu:{rank}')

        default_backend = c10d._get_default_group()._get_backend(device)
        default_value = default_backend.options.hccl_config.get("hccl_deterministic", -1)
        test_case.assertEqual(default_value, 1,
                              "hccl_deterministic of default group should keep 1 after new_group x2")
        test_case.assertEqual(t1.cpu(), world_size * original1,
                              "all_reduce on default group (deterministic=1) is incorrect.")

        # A rank only belongs to one sub-group; the other one is an int placeholder.
        if rank in group_a:
            own_pg, own_expect, own_ranks, own_input, own_original = pg_a, 0, group_a, input2, original2
        else:
            own_pg, own_expect, own_ranks, own_input, own_original = pg_b, 2, group_b, input3, original3

        t = own_input.npu(rank)
        dist.all_reduce(t, group=own_pg)

        own_backend = own_pg._get_backend(device)
        test_case.assertEqual(own_backend.options.hccl_config.get("hccl_deterministic", -1), own_expect,
                              f"hccl_deterministic of the sub-group should be {own_expect}")
        test_case.assertEqual(t.cpu(), len(own_ranks) * own_original,
                              f"all_reduce on a {len(own_ranks)}-rank sub-group "
                              f"(deterministic={own_expect}) is incorrect.")
        # The sub-domain must neither inherit nor clobber the default domain's value.
        test_case.assertNotEqual(default_value, own_expect,
                                 "the sub-domain must keep a value different from the default domain")

        dist.destroy_process_group()

    def _test_multiprocess(self, f, world_size, *args, collect_plog=False, extra_env=None):
        ctx = mp.get_context('spawn')
        # create_common_tensor returns NPU tensors; pass CPU copies to avoid aliasing.
        cpu_args = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]

        ps = []
        rank_pids = []
        ranks = range(world_size)
        with PlogCapture(label=f.__name__) as plog:
            # Workers read env at spawn/import time; set it only around start().
            saved_env = {}
            for key, value in (extra_env or {}).items():
                saved_env[key] = os.environ.get(key)
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            try:
                for rank in ranks:
                    p = ctx.Process(
                        target=f,
                        args=(rank, ranks, world_size, *cpu_args))
                    p.start()
                    ps.append(p)
                    rank_pids.append((rank, p.pid))
            finally:
                for key, value in saved_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            for p in ps:
                p.join()

            for p in ps:
                self.assertEqual(p.exitcode, 0)

            if not collect_plog:
                return None

            evidence = {}
            for rank, pid in rank_pids:
                text = plog.text_for_pid(pid)
                hccl = PlogCapture.parse(text)
                evidence[rank] = (hccl, text, plog.path)
            return evidence

    def _assert_plog_evidence(self, evidence, expected):
        r"""HCCL plog must show the value loaded for each expected comm.
        expected may be a dict, or a callable rank -> dict for per-rank comms."""
        self.assertIsNotNone(evidence, "plog capture was not enabled for this case")
        for rank in sorted(evidence):
            hccl, _, log_dir = evidence[rank]
            rank_msg = f"rank {rank} (plog: {log_dir})"
            rank_expected = expected(rank) if callable(expected) else expected
            for comm_name, value in rank_expected.items():
                self.assertIn(
                    comm_name, hccl,
                    f"{rank_msg}: HCCL did not load a config for [{comm_name}], "
                    f"comms seen in plog: {sorted(hccl)}")
                self.assertEqual(
                    hccl[comm_name], value,
                    f"{rank_msg}: HCCL loaded deterministic[{hccl[comm_name]}] for "
                    f"[{comm_name}], expected {value}")

    def _assert_env_fallback_evidence(self, evidence, expected_value):
        r"""No op_option: HCCL must log and load the env/default value."""
        self.assertIsNotNone(evidence, "plog capture was not enabled for this case")
        for rank in sorted(evidence):
            hccl, text, log_dir = evidence[rank]
            rank_msg = f"rank {rank} (plog: {log_dir})"
            fallback = HCCL_ENV_FALLBACK_LOG_RE.findall(text)
            self.assertTrue(
                fallback, f"{rank_msg}: HCCL did not log that it fell back to the "
                          f"env/default value")
            self.assertEqual(
                int(fallback[-1]), expected_value,
                f"{rank_msg}: HCCL used env/default hcclDeterministic "
                f"{fallback[-1]}, expected {expected_value}")
            self.assertIn(
                "group_name_0", hccl,
                f"{rank_msg}: HCCL did not load a config for [group_name_0], "
                f"comms seen in plog: {sorted(hccl)}")
            self.assertEqual(
                hccl["group_name_0"], expected_value,
                f"{rank_msg}: HCCL loaded deterministic[{hccl['group_name_0']}], "
                f"expected {expected_value}")

    def _assert_no_config_log(self, evidence):
        r"""A non-int value must fail before HCCL loads a config."""
        for rank in sorted(evidence):
            hccl, _, log_dir = evidence[rank]
            self.assertNotIn(
                "group_name_0", hccl,
                f"rank {rank}: HCCL loaded a config after an invalid value (plog: {log_dir})")

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_deterministic_all_reduce(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        for deterministic in (0, 1, 2):
            _, input1 = create_common_tensor(shape, -10, 10)
            evidence = self._test_multiprocess(HcclDeterministicTest._test_deterministic_all_reduce,
                                               world_size, input1, deterministic,
                                               collect_plog=True)
            self._assert_plog_evidence(evidence, {"group_name_0": deterministic})

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_deterministic_multi_group(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        _, input1 = create_common_tensor(shape, -10, 10)
        _, input2 = create_common_tensor(shape, -10, 10)
        evidence = self._test_multiprocess(HcclDeterministicTest._test_deterministic_multi_group,
                                           world_size, input1, input2, collect_plog=True)
        # Both default (1) and new_group (0) must appear in plog, not collapse.
        self._assert_plog_evidence(evidence, {"group_name_0": 1, "group_name_1": 0})

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_deterministic_wrong_type(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        _, input1 = create_common_tensor(shape, -10, 10)
        evidence = self._test_multiprocess(HcclDeterministicTest._test_hccl_deterministic_wrong_type,
                                           world_size, input1, collect_plog=True)
        self._assert_no_config_log(evidence)

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_deterministic_option_overrides_env(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        # op_option must win over env: (env, op) = (true, 0), (true, 2), (strict, 1).
        for env_value, option_value in (("true", 0), ("true", 2), ("strict", 1)):
            _, input1 = create_common_tensor(shape, -10, 10)
            evidence = self._test_multiprocess(
                HcclDeterministicTest._test_deterministic_all_reduce,
                world_size, input1, option_value,
                collect_plog=True, extra_env={"HCCL_DETERMINISTIC": env_value})
            self._assert_plog_evidence(evidence, {"group_name_0": option_value})

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_deterministic_env_fallback(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        # No op_option: HCCL must resolve the value from the env var.
        for env_value, expected in (("true", 1), ("strict", 2), ("false", 0)):
            _, input1 = create_common_tensor(shape, -10, 10)
            evidence = self._test_multiprocess(
                HcclDeterministicTest._test_deterministic_env_only,
                world_size, input1,
                collect_plog=True, extra_env={"HCCL_DETERMINISTIC": env_value})
            self._assert_env_fallback_evidence(evidence, expected)

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_deterministic_default(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        _, input1 = create_common_tensor(shape, -10, 10)
        # Neither op_option nor env: HCCL default (0) must be used.
        evidence = self._test_multiprocess(
            HcclDeterministicTest._test_deterministic_env_only,
            world_size, input1,
            collect_plog=True, extra_env={"HCCL_DETERMINISTIC": None})
        self._assert_env_fallback_evidence(evidence, 0)

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_deterministic_unset_key_with_other_config(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        _, input1 = create_common_tensor(shape, -10, 10)
        # Other keys present, hccl_deterministic absent: env must still win.
        evidence = self._test_multiprocess(
            HcclDeterministicTest._test_deterministic_other_config_only,
            world_size, input1,
            collect_plog=True, extra_env={"HCCL_DETERMINISTIC": "true"})
        self._assert_env_fallback_evidence(evidence, 1)

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_deterministic_comm_creation_paths(self):
        world_size = 2
        shape = [np.float32, 0, [2, 3, 16]]
        _, input1 = create_common_tensor(shape, -10, 10)
        evidence = self._test_multiprocess(HcclDeterministicTest._test_deterministic_comm_creation_paths,
                                           world_size, input1, collect_plog=True)
        # get_hccl_comm() path must carry deterministic=2 into HCCL.
        self._assert_plog_evidence(evidence, {"group_name_0": 2})

    @skipIfUnsupportMultiNPU(8)
    def test_hccl_deterministic_two_subdomains(self):
        world_size = 8
        shape = [np.float32, 0, [2, 3, 16]]
        _, input1 = create_common_tensor(shape, -10, 10)
        _, input2 = create_common_tensor(shape, -10, 10)
        _, input3 = create_common_tensor(shape, -10, 10)
        evidence = self._test_multiprocess(HcclDeterministicTest._test_deterministic_two_subdomains,
                                           world_size, input1, input2, input3, collect_plog=True)

        def expected(rank):
            # Default group is 1; a rank only joins its own sub-group (A=0 or B=2).
            if rank < 4:
                return {"group_name_0": 1, "group_name_1": 0}
            return {"group_name_0": 1, "group_name_2": 2}

        self._assert_plog_evidence(evidence, expected)


if __name__ == '__main__':
    run_tests()
