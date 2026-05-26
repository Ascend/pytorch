import inspect
import os
import time
import types
import unittest
from functools import wraps

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.pipelining import Schedule1F1B, SplitPoint
from torch_npu.distributed.pipelining import pipeline
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import run_tests, TestCase


d_hid = 64
batch_size = 32
chunks = 4
device_type = "npu"
total_steps = 15
warmup_steps = 5
join_timeout = 120


class TinyTransformerBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(dim)
        self.attention = torch.nn.Linear(dim, dim)
        self.ffn_norm = torch.nn.LayerNorm(dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TinyTransformer(torch.nn.Module):
    def __init__(self, dim: int = d_hid, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [TinyTransformerBlock(dim) for _ in range(n_layers)]
        )
        self.norm = torch.nn.LayerNorm(dim)
        self.output = torch.nn.Linear(dim, dim)
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))


class FXTracerPublicApiTest(TestCase):
    def test_public_pipeline_is_patched(self):
        import torch.distributed.pipelining as pipelining

        self.assertIsNot(pipeline, pipelining.pipeline)
        self.assertTrue(getattr(pipeline, "_torch_npu_pipelining_patch_applied", False))
        signature = inspect.signature(pipeline)
        self.assertIn("mode", signature.parameters)
        self.assertIn("atomic_units", signature.parameters)
        self.assertEqual(pipeline.__module__, "torch_npu.distributed.pipelining")


class FXTracerScheduleTest(TestCase):
    MAIN_PROCESS_RANK = -1
    world_size = 2

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                for process in self.processes:
                    process.join(join_timeout)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                    self.assertEqual(process.exitcode, 0)
            else:
                fn()

        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self):
        super(TestCase, self).setUp()
        if (
            not torch.npu.is_available()
            or torch.npu.device_count() < self.world_size
        ):
            raise unittest.SkipTest("Multi-NPU 2 condition not satisfied")

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29589"
        os.environ["BACKEND"] = dist.Backend.HCCL
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        proc = torch.multiprocessing.get_context("spawn").Process

        for rank in range(self.world_size):
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(rank, self._current_test_name()),
            )
            process.start()
            self.processes.append(process)

    def tearDown(self):
        super().tearDown()
        for process in self.processes:
            process.terminate()
        self.processes = []

    def _current_test_name(self) -> str:
        return self.id().split(".")[-1]

    @classmethod
    def _run(cls, rank: int, test_name: str) -> None:
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(cls.world_size)
        self = cls(test_name)
        self.rank = rank
        getattr(self, test_name)()

    def dist_init(self):
        torch.npu.set_device(self.rank)
        dist.init_process_group(
            backend="hccl",
            rank=self.rank,
            world_size=self.world_size,
        )

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank % torch.npu.device_count())

    def _seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _set_device(self) -> None:
        torch.npu.set_device(self.device)

    def _make_model_and_input(self):
        self._seed_all(2026)
        self._set_device()
        model = TinyTransformer(n_layers=self.world_size).to(self.device)
        x = torch.randn(batch_size, d_hid, device=self.device)
        return model, x

    def _build_pipe_and_stage(self, mode: str, atomic_units=None):
        model, x = self._make_model_and_input()
        x_mb = x.chunk(chunks)[0]

        kwargs = {
            "module": model,
            "mb_args": (x_mb,),
            "split_spec": model.split_spec,
            "mode": mode,
        }
        if atomic_units is not None:
            kwargs["atomic_units"] = atomic_units

        pipe = pipeline(**kwargs)
        stage = pipe.build_stage(self.rank, self.device)
        return model, pipe, stage, x

    def _schedule_step(self, schedule, *args, **kwargs):
        original_fork_rng = torch.random.fork_rng

        def npu_fork_rng(*fork_args, **fork_kwargs):
            if fork_kwargs.get("device_type", "cuda") == "cuda":
                fork_kwargs["device_type"] = "npu"
            return original_fork_rng(*fork_args, **fork_kwargs)

        torch.random.fork_rng = npu_fork_rng
        try:
            return schedule.step(*args, **kwargs)
        finally:
            torch.random.fork_rng = original_fork_rng

    def _run_1f1b(self, mode: str, atomic_units=None):
        model, pipe, stage, _ = self._build_pipe_and_stage(mode, atomic_units)
        self._set_device()
        self._seed_all(2027)
        inputs = [
            torch.randn(batch_size, d_hid, device=self.device)
            for _ in range(total_steps)
        ]
        targets = [
            torch.randn(batch_size, d_hid, device=self.device)
            for _ in range(total_steps)
        ]
        loss_fn = torch.nn.MSELoss(reduction="sum")
        schedule = Schedule1F1B(stage, chunks, loss_fn=loss_fn, scale_grads=False)
        stage_module = pipe.get_stage_module(self.rank)

        dist.barrier()
        self._sync_device()
        total_elapsed = 0.0
        step_losses = []
        final_out = None

        for step_idx in range(total_steps):
            stage_module.zero_grad(set_to_none=True)
            self._set_device()
            self._sync_device()

            start_time = time.perf_counter()
            losses = []
            out = self._schedule_step(
                schedule,
                inputs[step_idx], target=targets[step_idx], losses=losses
            )

            self._sync_device()
            if step_idx >= warmup_steps:
                total_elapsed += time.perf_counter() - start_time

            if self.rank == self.world_size - 1:
                final_out = out.detach()
                step_losses.append(sum(losses).detach())

        dist.barrier()

        if self.rank == self.world_size - 1:
            return model, pipe, final_out, step_losses, total_elapsed
        return model, pipe, None, None, total_elapsed

    def _assert_fx_close_to_export(self, atomic_units=None):
        _, _, export_out, export_losses, export_elapsed = self._run_1f1b("export")
        _, _, fx_out, fx_losses, fx_elapsed = self._run_1f1b(
            "fx",
            atomic_units=atomic_units,
        )

        if self.rank == self.world_size - 1:
            torch.testing.assert_close(fx_out, export_out, rtol=1e-3, atol=1e-3)
            self.assertEqual(len(fx_losses), total_steps)
            self.assertEqual(len(export_losses), total_steps)
            for fx_loss, export_loss in zip(fx_losses, export_losses):
                torch.testing.assert_close(
                    fx_loss,
                    export_loss,
                    rtol=1e-3,
                    atol=1e-3,
                )

        self.assertLessEqual(fx_elapsed, export_elapsed * 1.2)

    @skipIfUnsupportMultiNPU(2)
    def test_fx_tracer_builds_pipe_and_stage(self):
        self.dist_init()
        _, pipe, stage, _ = self._build_pipe_and_stage("fx")

        self.assertEqual(pipe.num_stages, self.world_size)
        self.assertIsNotNone(stage.submod)
        self.assertEqual(stage.stage_index, self.rank)

    @skipIfUnsupportMultiNPU(2)
    def test_fx_tracer_matches_export_with_1f1b(self):
        self.dist_init()
        self._assert_fx_close_to_export()

    @skipIfUnsupportMultiNPU(2)
    def test_fx_tracer_atomic_units_match_export_with_1f1b(self):
        self.dist_init()
        self._assert_fx_close_to_export(atomic_units=["layers.0"])


if __name__ == "__main__":
    run_tests()
