import copy
import json
import os
import pickle
import tempfile
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch_npu
from torch_npu.profiler import _add_native_entry_patch as entry
from torch_npu.profiler._kineto_config import _custom_profiler_config, _NPUExperimentalConfig
from torch_npu.profiler.profiler import tensorboard_trace_handler
from torch_npu.profiler.analysis._profiler_config import ProfilerConfig
from torch_npu.profiler.analysis.prof_common_func._path_manager import ProfilerPathManager
from torch_npu.testing.testcase import TestCase, run_tests


class TestAclprofNative(TestCase):
    @staticmethod
    def _stub_capture(stack, prof, prepare, start, stop):
        import torch.autograd.profiler as native
        import torch.profiler.profiler as public

        prof.use_device = "npu"
        stack.enter_context(patch.object(public, "_get_privateuse1_backend_name", return_value="npu"))
        stack.enter_context(patch.object(torch.npu, "synchronize"))
        stack.enter_context(patch.multiple(
            native, _get_privateuse1_backend_name=lambda: "npu",
            _supported_activities=lambda: {torch.profiler.ProfilerActivity.PrivateUse1},
            _prepare_profiler=prepare, _enable_profiler=start, _disable_profiler=stop,
        ))

    def test_constructor_is_the_only_capture_patch(self):
        target = torch.profiler.profile
        with patch.object(target, "__init__", entry.original_profile_init):
            before = dict(vars(target))
            entry._apply_native_entry_patch()
            after = dict(vars(target))
            self.assertEqual({key for key in after if before.get(key) is not after[key]}, {"__init__"})
            self.assertIs(after["__init__"], entry._custom_profile_init)

    def test_native_and_mspti_configs_pass_through(self):
        with patch.object(entry, "original_profile_init") as original:
            for config in (None, torch._C._profiler._ExperimentalConfig(), _NPUExperimentalConfig(profiler_level="Level1")):
                entry._custom_profile_init(SimpleNamespace(), experimental_config=config)
                self.assertIs(original.call_args.kwargs["experimental_config"], config)

    def test_handler_paths_are_unique_and_device_settings_survive_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            first = tensorboard_trace_handler(directory, analyse_flag=False)
            second = tensorboard_trace_handler(directory, analyse_flag=False)
            self.assertNotEqual(first.prof_path, second.prof_path)
        validate = _NPUExperimentalConfig._get_device_config
        with patch.object(_NPUExperimentalConfig, "_get_device_config", autospec=True, side_effect=validate) as build:
            cfg = _NPUExperimentalConfig(profiler_level="Level1", l2_cache=True, op_attr=True)
            cfg.extra = Lock()
            copies = (cfg, copy.copy(cfg), pickle.loads(pickle.dumps(cfg)))
            build.assert_not_called()
            for candidate in copies:
                build.reset_mock()
                carrier = candidate.to_custom_profiler_config({"backend": "aclprof", "export_type": ["db"]})
                build.assert_called_once_with(candidate, "Level1", ["db"])
                payload = json.loads(_custom_profiler_config(carrier))
                self.assertEqual(payload["profiler_level"], "Level1")
                self.assertEqual((payload["l2_cache"], payload["op_attr"]), (True, True))

    def test_binding_preserves_native_config_state(self):
        cfg = _NPUExperimentalConfig(profiler_level="Level1", enable_cuda_sync_events=True)
        before = cfg.__getstate__()
        with tempfile.TemporaryDirectory() as directory:
            handler = tensorboard_trace_handler(directory, analyse_flag=False)
            self.assertFalse(os.path.exists(handler.prof_path))
            prof = torch.profiler.profile(experimental_config=cfg, on_trace_ready=handler)
            state = prof.experimental_config.__getstate__()
            payload = json.loads(_custom_profiler_config(prof.experimental_config))
            self.assertEqual(payload["backend"], "aclprof")
            self.assertEqual(payload["output_path"], handler.prof_path)
            self.assertEqual(prof.activities, {torch.profiler.ProfilerActivity.PrivateUse1})
            self.assertEqual([v for v in state if not isinstance(v, str)],
                             [v for v in before if not isinstance(v, str)])
            self.assertEqual(copy.copy(prof.experimental_config).__getstate__(), state)
            self.assertEqual(cfg.__getstate__(), before)
            with self.assertRaisesRegex(ValueError, "one profiler session"):
                torch.profiler.profile(experimental_config=cfg, on_trace_ready=handler)

    def test_unsupported_framework_collection_fails_before_native_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = tensorboard_trace_handler(directory, analyse_flag=False)
            for option in ("with_stack", "with_modules", "with_flops"):
                with self.assertRaisesRegex(ValueError, "native capture"):
                    torch.profiler.profile(experimental_config=_NPUExperimentalConfig(),
                                           on_trace_ready=handler, **{option: True})
                self.assertIsNone(handler._profile_ref)

    def test_automatic_analysis_is_rejected_before_native_preparation(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(entry, "original_profile_init") as original,
            patch("torch.autograd.profiler._prepare_profiler") as prepare,
        ):
            for options in ({}, {"analyse_flag": True}):
                handler = tensorboard_trace_handler(directory, **options)
                with self.assertRaisesRegex(ValueError, "set analyse_flag=False"):
                    torch.profiler.profile(experimental_config=_NPUExperimentalConfig(), on_trace_ready=handler)
                self.assertIsNone(handler._profile_ref)
                self.assertFalse(os.path.exists(handler.prof_path))
            original.assert_not_called()
            prepare.assert_not_called()

    def test_invalid_aclprof_payload_state_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = tensorboard_trace_handler(directory, analyse_flag=False)
            with patch.object(_NPUExperimentalConfig, "__getstate__", return_value=()), \
                    patch.object(entry, "original_profile_init") as original:
                with self.assertRaisesRegex(RuntimeError, "Incompatible PyTorch"):
                    torch.profiler.profile(experimental_config=_NPUExperimentalConfig(), on_trace_ready=handler)
                original.assert_not_called()

    def test_handler_writes_each_window_without_analysis_or_event_export(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(torch.distributed, "is_initialized", return_value=False),
            patch("torch_npu.profiler.profiler.NpuProfiler.analyse") as analyse,
            patch.object(torch_npu._C._profiler, "_get_completed_aclprof_window") as completed,
        ):
            handler = tensorboard_trace_handler(directory, analyse_flag=False)
            prof = torch.profiler.profile(
                experimental_config=_NPUExperimentalConfig(profiler_level="Level1", export_type="db", mstx=True),
                on_trace_ready=handler, schedule=lambda _: torch.profiler.ProfilerAction.RECORD,
            )
            prof.schedule.lock = Lock()
            prof.events = Mock(side_effect=AssertionError("events must not be materialized"))
            written = {}
            for window, rank in ((1, None), (2, "3")):
                path = Path(handler.prof_path) / f"window-{window}"
                host = path / "PROF_1_2_test" / "host"
                host.mkdir(parents=True)
                (host / "start_info").write_text('{"collectionTimeBegin":"10","clockMonotonicRaw":1}')
                env = {} if rank is None else {"RANK": rank}
                completed.return_value = (str(path), (True, 100, window, 1000 * window))
                with patch.dict(os.environ, env, clear=True, ASCEND_HOME_PATH=directory):
                    handler(prof)
                info = path / ("profiler_info.json" if rank is None else f"profiler_info_{rank}.json")
                written[info] = info.read_bytes()
                config = json.loads(written[info])["config"]
                self.assertEqual(config["common_config"]["schedule"], {})
                experimental = config["experimental_config"]
                self.assertEqual((experimental["_mstx"], experimental["_msprof_tx"]), (True, False))
                reader = ProfilerConfig()
                reader.reset_is_load()
                reader.load_info(str(path))
                self.assertEqual((reader.activities, reader.export_type, reader.rank_id),
                                 (["ProfilerActivity.NPU"], ["db"], rank if rank is not None else -1))
                self.assertEqual(reader.get_timestamp_from_syscnt(window), 1000 * window)
                reader.reset_is_load()
            self.assertEqual({info: info.read_bytes() for info in written}, written)
            analyse.assert_not_called()
            prof.events.assert_not_called()

    def test_body_error_survives_missing_window_or_metadata_failure(self):
        for complete in (False, True):
            with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                handler = tensorboard_trace_handler(directory, analyse_flag=False, use_gzip=True)
                prof = torch.profiler.profile(experimental_config=_NPUExperimentalConfig(), on_trace_ready=handler)
                path = Path(handler.prof_path) / "window"
                raw = path / "PROF_1_2_test"
                raw.mkdir(parents=True)
                if not complete:
                    (path / ".aclprof_incomplete").touch()
                self._stub_capture(stack, prof, Mock(), Mock(), Mock())
                native = torch_npu._C._profiler
                output = (str(path), (False, 100, 0, 1)) if complete else None
                window = stack.enter_context(patch.object(native, "_get_completed_aclprof_window", return_value=output))
                finish = stack.enter_context(patch.object(native, "FinalizeAclprofBackend"))
                writer = stack.enter_context(patch(
                    "torch_npu.profiler.profiler.FileManager.create_json_file_by_path",
                    side_effect=RuntimeError("writer failed"),
                ))
                archive = stack.enter_context(patch("torch_npu.profiler.profiler.FileManager.compress_to_tar_gz"))
                with self.assertRaisesRegex(ValueError, "body failed"):
                    with prof:
                        raise ValueError("body failed")
                window.assert_called_once_with(handler.prof_path, prof.get_trace_id())
                finish.assert_not_called()
                archive.assert_not_called()
                self.assertEqual(writer.call_count, int(complete))
                for input_path in (path, path.parent):
                    self.assertEqual(ProfilerPathManager.get_profiler_path_list(str(input_path)),
                                     [str(path)] if complete else [])
                self.assertTrue(raw.is_dir())

    def test_schedule_continues_after_an_incomplete_window(self):
        calls, paths = [], []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            handler = tensorboard_trace_handler(directory, analyse_flag=False)
            prof = torch.profiler.profile(
                experimental_config=_NPUExperimentalConfig(), on_trace_ready=handler,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=2),
            )

            def prepare(*args, **kwargs):
                calls.append("prepare")
                path = Path(handler.prof_path) / f"window-{len(paths)}"
                path.mkdir(parents=True)
                paths.append(path)

            self._stub_capture(stack, prof, prepare, lambda *a: calls.append("start"), lambda: calls.append("stop"))
            start_info = (False, 100, 0, 1)
            output = stack.enter_context(patch.object(
                torch_npu._C._profiler, "_get_completed_aclprof_window",
                side_effect=lambda *_: (str(paths[-1]), start_info) if len(paths) == 2 else None,
            ))
            with prof:
                self.assertEqual(calls, [])
                for _ in range(6):
                    prof.step()
            self.assertEqual(calls, ["prepare", "start", "stop"] * 2)
            self.assertEqual(output.call_count, 2)
            self.assertNotEqual(output.call_args_list[0].args[1], output.call_args_list[1].args[1])
            self.assertEqual(list(paths[0].iterdir()), [])
            self.assertTrue(any(paths[1].glob("profiler_info*.json")))

    def test_standard_warmup_exit_drains_without_handler(self):
        action = torch.profiler.ProfilerAction
        for step_to_none in (False, True):
            with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                handler = tensorboard_trace_handler(directory, analyse_flag=False)
                prof = torch.profiler.profile(
                    experimental_config=_NPUExperimentalConfig(backend="aclprof"), on_trace_ready=handler,
                    schedule=lambda step: action.NONE if step else action.WARMUP,
                )
                prepare, start, stop = Mock(), Mock(), Mock()
                self._stub_capture(stack, prof, prepare, start, stop)
                window = stack.enter_context(patch.object(torch_npu._C._profiler, "_get_completed_aclprof_window"))
                with prof:
                    if step_to_none:
                        with self.assertWarnsRegex(UserWarning, "WARMUP followed by NONE"):
                            prof.step()
                for phase in (prepare, start, stop):
                    phase.assert_called_once()
                window.assert_not_called()

    def test_legacy_handler_still_uses_legacy_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            for analyse in (True, False):
                prof = SimpleNamespace(prof_if=Mock())
                tensorboard_trace_handler(directory, analyse_flag=analyse)(prof)
                self.assertEqual(prof.prof_if.analyse.call_count, int(analyse))
                if analyse:
                    prof.prof_if.analyse.assert_called_with(async_mode=False)


if __name__ == "__main__":
    run_tests()
