import os
import re
import socket
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


RUN_SCALABLE_ROOTINFO_TESTS = os.getenv("TORCH_NPU_RUN_SCALABLE_ROOTINFO_TESTS") == "1"
SCALABLE_ROOTINFO_SKIP_REASON = (
    "manual-only Scalable RootInfo test; "
    "set TORCH_NPU_RUN_SCALABLE_ROOTINFO_TESTS=1 to run with a supported HCCL library"
)
SCALABLE_COMM_LOG_KEYWORD = "Create hccl comm by hcclCommInitRootInfoScalable success"
SCALABLE_LAYOUT_LOG_KEYWORD = "Scalable HCCL RootInfo path selected"
ROOTINFO_EXCHANGE_LOG_KEYWORD = "All-gather scalable HCCL root infos through store success"
ROOTINFO_START_LOG_KEYWORD = "Start generating scalable HCCL root info"
ROOTINFO_GENERATE_LOG_KEYWORD = "Generate and publish scalable HCCL root info success"
ORIGINAL_COMM_LOG_KEYWORD = "Create hccl comm by hcclCommInitRootInfoConfig success"
UNSUPPORTED_SOC_WARNING_KEYWORD = "Scalable RootInfo initialization is supported only on Atlas A2 and A3"
STDERR_FD = 2


def _run_collectives(rank, world_size, run_all_collectives):
    tensor = torch.tensor([rank + 1], dtype=torch.float32, device=f"npu:{rank}")
    dist.all_reduce(tensor)

    expected = world_size * (world_size + 1) / 2
    actual = tensor.cpu().item()
    if actual != expected:
        raise AssertionError(f"rank {rank}: expected all_reduce result {expected}, got {actual}")

    if not run_all_collectives:
        return

    tensor = torch.tensor([123.0 if rank == 0 else -1.0], device=f"npu:{rank}")
    dist.broadcast(tensor, src=0)
    actual = tensor.cpu().item()
    if actual != 123.0:
        raise AssertionError(f"rank {rank}: expected broadcast result 123, got {actual}")

    tensor = torch.tensor([rank], dtype=torch.int32, device=f"npu:{rank}")
    output = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor)
    actual = [item.cpu().item() for item in output]
    expected = list(range(world_size))
    if actual != expected:
        raise AssertionError(f"rank {rank}: expected all_gather result {expected}, got {actual}")

    dist.barrier()


def _run_scalable_rootinfo_case(
    rank,
    world_size,
    ranks_per_root,
    scalable_enabled,
    master_ports,
    socket_port_range,
    log_dir,
    run_all_collectives,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = socket_port_range
    os.environ["ROOTINFO_SCALABLE_ENABLE"] = "1" if scalable_enabled else "0"
    if ranks_per_root is None:
        os.environ.pop("TORCH_HCCL_RANKS_PER_ROOT", None)
    else:
        os.environ["TORCH_HCCL_RANKS_PER_ROOT"] = str(ranks_per_root)
    os.environ.pop("RANK_TABLE_FILE", None)

    log_path = os.path.join(log_dir, f"rank_{rank}.log")
    original_stderr_fd = os.dup(STDERR_FD)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log_fd, STDERR_FD)

    try:
        torch_npu._C._logging._LogContext.GetInstance().setLogs({"torch.distributed": 20})
        torch_npu.npu.set_device(rank)
        for master_port in master_ports:
            os.environ["MASTER_PORT"] = str(master_port)
            dist.init_process_group(
                backend="hccl",
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=120),
            )
            _run_collectives(rank, world_size, run_all_collectives)
            dist.destroy_process_group()
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            os.fsync(STDERR_FD)
            os.dup2(original_stderr_fd, STDERR_FD)
            os.close(original_stderr_fd)
            os.close(log_fd)


@unittest.skipUnless(RUN_SCALABLE_ROOTINFO_TESTS, SCALABLE_ROOTINFO_SKIP_REASON)
class ScalableRootInfoTest(TestCase):
    world_size = 4
    socket_port_counter = 0

    def setUp(self):
        if torch_npu.npu.device_count() < self.world_size:
            raise unittest.SkipTest(f"Scalable RootInfo test requires {self.world_size}+ NPUs")

    @staticmethod
    def _find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    @classmethod
    def _alloc_socket_port_range(cls):
        cls.socket_port_counter += 1
        base = 30000 + cls.socket_port_counter * 100
        return f"{base}-{base + 99}"

    @staticmethod
    def _is_scalable_supported_soc():
        device_name = torch_npu.npu.get_device_name(0)
        is_a2 = re.fullmatch(r"Ascend910B(?:1|2|2C|3|4|4-1)", device_name) is not None
        is_a3 = re.fullmatch(r"Ascend910_9(?:391|392|381|382|372|362|363)", device_name) is not None
        return is_a2 or is_a3

    @staticmethod
    def _expected_roots(world_size, ranks_per_root):
        if world_size <= 0 or ranks_per_root <= 0:
            raise ValueError("world_size and ranks_per_root must be greater than 0")
        actual_root_num = (world_size + ranks_per_root - 1) // ranks_per_root
        ranks_per_root = world_size // actual_root_num
        remainder = world_size % actual_root_num
        roots = {}
        for root_index in range(actual_root_num):
            if root_index < remainder:
                root_rank = root_index * (ranks_per_root + 1)
            else:
                root_rank = remainder * (ranks_per_root + 1) + (root_index - remainder) * ranks_per_root
            roots[root_rank] = root_index
        return roots

    def _assert_path_logs(self, log_dir, world_size, ranks_per_root, scalable_enabled, rounds):
        effective_ranks_per_root = 128 if ranks_per_root is None else ranks_per_root
        valid_ranks_per_root = isinstance(effective_ranks_per_root, int) and effective_ranks_per_root > 0
        scalable_requested = scalable_enabled and valid_ranks_per_root and world_size > effective_ranks_per_root
        scalable_supported_soc = self._is_scalable_supported_soc()
        use_scalable_path = scalable_requested and scalable_supported_soc
        expected_roots = (
            self._expected_roots(world_size, effective_ranks_per_root) if use_scalable_path else {}
        )
        actual_root_num = len(expected_roots) if use_scalable_path else None

        for rank in range(world_size):
            log_path = os.path.join(log_dir, f"rank_{rank}.log")
            with open(log_path, encoding="utf-8") as log_file:
                log_text = log_file.read()

            if not use_scalable_path:
                self.assertEqual(log_text.count(ORIGINAL_COMM_LOG_KEYWORD), rounds)
                self.assertNotIn(SCALABLE_COMM_LOG_KEYWORD, log_text)
                self.assertNotIn(SCALABLE_LAYOUT_LOG_KEYWORD, log_text)
                self.assertNotIn(ROOTINFO_EXCHANGE_LOG_KEYWORD, log_text)
                self.assertNotIn(ROOTINFO_START_LOG_KEYWORD, log_text)
                self.assertNotIn(ROOTINFO_GENERATE_LOG_KEYWORD, log_text)
                if scalable_enabled and not scalable_supported_soc:
                    self.assertEqual(log_text.count(UNSUPPORTED_SOC_WARNING_KEYWORD), 1)
                else:
                    self.assertNotIn(UNSUPPORTED_SOC_WARNING_KEYWORD, log_text)
                continue

            self.assertEqual(log_text.count(SCALABLE_COMM_LOG_KEYWORD), rounds)
            self.assertEqual(log_text.count(ROOTINFO_EXCHANGE_LOG_KEYWORD), rounds)
            self.assertIn(f"root num is {actual_root_num}", log_text)
            self.assertNotIn(ORIGINAL_COMM_LOG_KEYWORD, log_text)
            base_group_size = world_size // actual_root_num
            larger_group_count = world_size % actual_root_num
            if rank == 0:
                expected_layout_log = (
                    f"num ranks is {world_size}, "
                    f"root num is {actual_root_num}, base group rank num is {base_group_size}, "
                    f"groups with one extra rank is {larger_group_count}"
                )
                self.assertEqual(log_text.count(SCALABLE_LAYOUT_LOG_KEYWORD), rounds)
                self.assertEqual(log_text.count(expected_layout_log), rounds)
            else:
                self.assertNotIn(SCALABLE_LAYOUT_LOG_KEYWORD, log_text)
            if rank in expected_roots:
                root_index = expected_roots[rank]
                root_group_size = base_group_size + (1 if root_index < larger_group_count else 0)
                expected_start_log = (
                    f"{ROOTINFO_START_LOG_KEYWORD}, rank is {rank}, root index is {root_index}, "
                    f"group rank range is [{rank}, {rank + root_group_size - 1}], "
                    f"group rank num is {root_group_size}"
                )
                self.assertEqual(log_text.count(expected_start_log), rounds)
                expected_log = (
                    f"{ROOTINFO_GENERATE_LOG_KEYWORD}, rank is {rank}, "
                    f"root index is {root_index}"
                )
                self.assertEqual(log_text.count(expected_log), rounds)
                self.assertEqual(log_text.count(ROOTINFO_GENERATE_LOG_KEYWORD), rounds)
            else:
                self.assertNotIn(ROOTINFO_START_LOG_KEYWORD, log_text)
                self.assertNotIn(ROOTINFO_GENERATE_LOG_KEYWORD, log_text)

    def _run_case(
        self,
        ranks_per_root,
        scalable_enabled=True,
        run_all_collectives=False,
        rounds=1,
        expected_error=None,
        world_size=None,
    ):
        world_size = self.world_size if world_size is None else world_size
        if torch_npu.npu.device_count() < world_size:
            raise unittest.SkipTest(f"Scalable RootInfo test requires {world_size}+ NPUs")
        master_ports = []
        while len(master_ports) < rounds:
            master_port = self._find_free_port()
            if master_port not in master_ports:
                master_ports.append(master_port)
        socket_port_range = self._alloc_socket_port_range()
        with tempfile.TemporaryDirectory(prefix="scalable_rootinfo_") as log_dir:
            spawn_args = (
                world_size,
                ranks_per_root,
                scalable_enabled,
                master_ports,
                socket_port_range,
                log_dir,
                run_all_collectives,
            )
            if expected_error is not None:
                with self.assertRaisesRegex(Exception, expected_error):
                    mp.spawn(_run_scalable_rootinfo_case, args=spawn_args, nprocs=world_size, join=True)
                return

            mp.spawn(_run_scalable_rootinfo_case, args=spawn_args, nprocs=world_size, join=True)
            self._assert_path_logs(log_dir, world_size, ranks_per_root, scalable_enabled, rounds)

    def test_balanced_roots_and_collectives(self):
        self._run_case(ranks_per_root=2, run_all_collectives=True)

    def test_every_rank_is_root_and_collectives(self):
        self._run_case(ranks_per_root=1, run_all_collectives=True)

    def test_eight_rank_balanced_roots_and_collectives(self):
        self._run_case(ranks_per_root=2, run_all_collectives=True, world_size=8)

    def test_eight_rank_non_divisible_roots_and_collectives(self):
        self._run_case(ranks_per_root=3, run_all_collectives=True, world_size=8)

    def test_ranks_per_root_equal_to_world_size_uses_original_path(self):
        self._run_case(ranks_per_root=4)

    def test_switch_disabled_uses_original_path(self):
        self._run_case(ranks_per_root=2, scalable_enabled=False)

    def test_default_ranks_per_root_uses_original_path(self):
        self._run_case(ranks_per_root=None)

    def test_repeated_create_and_destroy(self):
        self._run_case(ranks_per_root=2, rounds=2)

    def test_zero_ranks_per_root_is_rejected(self):
        expected_error = (
            "TORCH_HCCL_RANKS_PER_ROOT must be a positive uint32_t"
            if self._is_scalable_supported_soc()
            else None
        )
        self._run_case(ranks_per_root=0, expected_error=expected_error)

    def test_wrong_ranks_per_root_type_is_rejected(self):
        expected_error = (
            "TORCH_HCCL_RANKS_PER_ROOT must be a positive uint32_t"
            if self._is_scalable_supported_soc()
            else None
        )
        self._run_case(ranks_per_root="invalid", expected_error=expected_error)


if __name__ == "__main__":
    run_tests()
