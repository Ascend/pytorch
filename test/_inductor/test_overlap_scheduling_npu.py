# Owner(s): ["module: inductor"]

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch
from torch._inductor import fx_utils as inductor_fx_utils
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import torch_npu  # noqa: F401
from torch_npu._inductor.fx_passes import overlap_scheduling as npu_overlap


TestCase = unittest.TestCase


class _FakeEvent:
    def __init__(self, factory, event_index):
        self.factory = factory
        self.event_index = event_index
        self.run_index = event_index // 2

    def record(self):
        self.factory.calls.append(("record", self.event_index))

    def synchronize(self):
        self.factory.calls.append(("event_synchronize", self.event_index))

    def elapsed_time(self, end_event):
        self.factory.calls.append(
            ("elapsed_time", self.event_index, end_event.event_index)
        )
        return self.factory.durations[self.run_index]


class _EventFactory:
    def __init__(self, durations):
        self.durations = durations
        self.calls = []
        self.created = 0

    def __call__(self, *, enable_timing):
        if not enable_timing:
            raise AssertionError("timing events must enable timing")
        event = _FakeEvent(self, self.created)
        self.created += 1
        return event


class _CallableTarget:
    def __init__(self, packet, name="torch.ops.npu.npu_grouped_matmul.default"):
        self.overloadpacket = packet
        self.name = name
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return torch.empty(0)

    def __str__(self):
        return self.name


class TestOverlapSchedulingHelpers(TestCase):
    def test_median_uses_lower_value_for_even_samples(self):
        self.assertEqual(npu_overlap._median([5.0, 1.0, 3.0]), 3.0)
        self.assertEqual(npu_overlap._median([9.0, 1.0, 5.0, 3.0]), 3.0)
        with self.assertRaisesRegex(ValueError, "empty list"):
            npu_overlap._median([])

    def test_fake_tensor_materialization_preserves_tensor_metadata(self):
        mode = FakeTensorMode()
        with mode:
            fake_float = torch.empty_strided(
                (2, 3), (4, 1), dtype=torch.float32, device="cpu"
            )
            fake_index = torch.empty(3, dtype=torch.int64, device="cpu")

        args, kwargs = npu_overlap._fake_tensors_to_real(
            ([fake_float],), {"index": fake_index, "constant": 7}
        )
        real_float = args[0][0]
        real_index = kwargs["index"]

        self.assertNotIsInstance(real_float, FakeTensor)
        self.assertNotIsInstance(real_index, FakeTensor)
        self.assertEqual(real_float.shape, fake_float.shape)
        self.assertEqual(real_float.stride(), fake_float.stride())
        self.assertEqual(real_float.dtype, fake_float.dtype)
        self.assertEqual(real_float.device, fake_float.device)
        self.assertEqual(real_index.dtype, torch.int64)
        self.assertEqual(kwargs["constant"], 7)

    def test_fake_tensor_materialization_rejects_unbacked_dimensions(self):
        mode = FakeTensorMode()
        with mode:
            fake = torch.empty(2, 3)

        runtime_estimation = torch._inductor.fx_passes.node_runtime_estimation
        with mock.patch.object(runtime_estimation, "get_hint", return_value=None):
            with self.assertRaisesRegex(ValueError, "unbacked dimensions"):
                npu_overlap._fake_tensors_to_real((fake,), {})

    def test_event_benchmark_warmup_order_and_median(self):
        events = _EventFactory([3.0, 1.0, 2.0])
        calls = []
        fake_npu = types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("device_synchronize"),
            Event=events,
        )

        with mock.patch.object(npu_overlap.torch, "npu", fake_npu, create=True):
            runtime = npu_overlap._benchmark_callable_with_npu_events(
                lambda: calls.append("fn"), warmup=2, nruns=3
            )

        self.assertEqual(runtime, 2.0)
        self.assertEqual(calls.count("fn"), 5)
        self.assertEqual(calls[:3], ["fn", "fn", "device_synchronize"])
        self.assertEqual(events.created, 6)
        self.assertEqual(
            [call[0] for call in events.calls],
            [
                "record",
                "record",
                "event_synchronize",
                "elapsed_time",
                "record",
                "record",
                "event_synchronize",
                "elapsed_time",
                "record",
                "record",
                "event_synchronize",
                "elapsed_time",
            ],
        )

    def test_event_benchmark_requires_npu(self):
        fake_npu = types.SimpleNamespace(is_available=lambda: False)
        with mock.patch.object(npu_overlap.torch, "npu", fake_npu, create=True):
            with self.assertRaisesRegex(RuntimeError, "available NPU"):
                npu_overlap._benchmark_callable_with_npu_events(lambda: None)


class TestCollectiveBenchmarkFunctionality(TestCase):
    def test_collective_benchmark_returns_none_without_npu(self):
        node = types.SimpleNamespace(target=mock.Mock())
        fake_npu = types.SimpleNamespace(is_available=lambda: False)
        with mock.patch.object(npu_overlap.torch, "npu", fake_npu, create=True):
            self.assertIsNone(
                npu_overlap._benchmark_collective_with_npu_events_impl(
                    node, (torch.empty(2),), {}, nruns=3
                )
            )
        node.target.assert_not_called()

    def test_collective_benchmark_materializes_and_waits_before_end_event(self):
        events = _EventFactory([0.7, 0.5, 0.6])
        device_calls = []
        fake_npu = types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: device_calls.append("device_synchronize"),
            Event=events,
        )
        target = mock.Mock(side_effect=lambda tensor, *, scale: tensor * scale)
        node = types.SimpleNamespace(target=target)
        real_args = (torch.ones(2),)
        real_kwargs = {"scale": 2}

        with (
            mock.patch.object(npu_overlap.torch, "npu", fake_npu, create=True),
            mock.patch.object(
                npu_overlap,
                "_fake_tensors_to_real",
                return_value=(real_args, real_kwargs),
            ) as materialize,
            mock.patch.object(npu_overlap, "_wait_collective_result") as wait,
        ):
            runtime = npu_overlap._benchmark_collective_with_npu_events_impl(
                node, ("fake",), {"scale": 9}, nruns=3
            )

        self.assertAlmostEqual(runtime, 0.6, places=6)
        materialize.assert_called_once_with(("fake",), {"scale": 9})
        self.assertEqual(target.call_count, 4)  # one warmup and three samples
        self.assertEqual(wait.call_count, 4)
        self.assertEqual(device_calls, ["device_synchronize", "device_synchronize"])
        for call in target.call_args_list:
            self.assertIs(call.args[0], real_args[0])
            self.assertEqual(call.kwargs, real_kwargs)

    def test_wait_collective_result_walks_nested_tensor_outputs(self):
        first = torch.ones(1)
        second = torch.ones(2)
        waited = []

        def wait_tensor(tensor):
            waited.append(tensor)
            return tensor

        with mock.patch.object(
            torch.ops._c10d_functional, "wait_tensor", side_effect=wait_tensor
        ):
            npu_overlap._wait_collective_result(
                {"a": first, "nested": ["untouched", (second,)]}
            )

        self.assertEqual(len(waited), 2)
        self.assertIs(waited[0], first)
        self.assertIs(waited[1], second)


class TestComputeNodeClassification(TestCase):
    def test_registered_compute_ops_and_upstream_fallback(self):
        packets = {
            "npu_grouped_matmul": object(),
            "npu_fusion_attention_v3": object(),
            "npu_fusion_attention_grad_v3": object(),
        }
        upstream_packet = object()

        def lookup(name):
            return packets[name]

        def upstream(node):
            return getattr(node.target, "overloadpacket", None) is upstream_packet

        with mock.patch.object(
            npu_overlap, "_get_registered_npu_op_packet", side_effect=lookup
        ):
            is_compute = npu_overlap._build_npu_is_compute_node(upstream)

        for packet in packets.values():
            node = types.SimpleNamespace(
                target=types.SimpleNamespace(overloadpacket=packet)
            )
            self.assertTrue(is_compute(node))

        upstream_node = types.SimpleNamespace(
            target=types.SimpleNamespace(overloadpacket=upstream_packet)
        )
        other_node = types.SimpleNamespace(
            target=types.SimpleNamespace(overloadpacket=object())
        )
        self.assertTrue(is_compute(upstream_node))
        self.assertFalse(is_compute(other_node))


class TestBalancedGroupList(TestCase):
    @staticmethod
    def _hint(value):
        return int(value)

    def test_group_list_type_zero_is_balanced_cumulative_offsets(self):
        group_list = torch.empty(3, dtype=torch.int64)
        result = npu_overlap._balanced_group_list_for_benchmark(
            group_list,
            ([torch.empty(10, 8)],),
            {"group_type": 0, "group_list_type": 0},
            self._hint,
        )
        self.assertEqual(result.tolist(), [4, 7, 10])
        self.assertEqual(result.dtype, group_list.dtype)
        self.assertEqual(result.device, group_list.device)

    def test_group_list_type_one_is_balanced_counts(self):
        result = npu_overlap._balanced_group_list_for_benchmark(
            torch.empty(3, dtype=torch.int64),
            ([torch.empty(10, 8)],),
            {"group_type": 0, "group_list_type": 1},
            self._hint,
        )
        self.assertEqual(result.tolist(), [4, 3, 3])
        self.assertEqual(sum(result.tolist()), 10)

    def test_group_list_type_two_is_id_and_count_matrix(self):
        result = npu_overlap._balanced_group_list_for_benchmark(
            torch.empty(3, 2, dtype=torch.int64),
            ([torch.empty(10, 8)],),
            {"group_type": 0, "group_list_type": 2},
            self._hint,
        )
        self.assertEqual(result.tolist(), [[0, 4], [1, 3], [2, 3]])

    def test_k_split_uses_contraction_dimension(self):
        result = npu_overlap._balanced_group_list_for_benchmark(
            torch.empty(4, dtype=torch.int64),
            ([torch.empty(16, 5)],),
            {"group_type": 2, "group_list_type": 0},
            self._hint,
        )
        self.assertEqual(result.tolist(), [2, 3, 4, 5])

    def test_more_groups_than_tokens_keeps_total_work(self):
        counts = npu_overlap._balanced_group_list_for_benchmark(
            torch.empty(5, dtype=torch.int64),
            ([torch.empty(2, 8)],),
            {"group_type": 0, "group_list_type": 1},
            self._hint,
        )
        self.assertEqual(counts.tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(sum(counts.tolist()), 2)

    def test_invalid_group_list_inputs_fall_back_to_generic_materialization(self):
        valid_x = ([torch.empty(10, 8)],)
        cases = [
            (torch.empty(2, 2, 2, dtype=torch.int64), valid_x, {"group_type": 0}),
            (torch.empty(0, dtype=torch.int64), valid_x, {"group_type": 0}),
            (torch.empty(3, dtype=torch.int64), (), {"group_type": 0}),
            (torch.empty(3, dtype=torch.int64), valid_x, {}),
            (torch.empty(3, dtype=torch.int64), valid_x, {"group_type": None}),
            (torch.empty(3, dtype=torch.int64), valid_x, {"group_type": -1}),
            (torch.empty(3, dtype=torch.int64), valid_x, {"group_type": False}),
            (torch.empty(3, dtype=torch.int64), valid_x, {"group_type": 1}),
            (
                torch.empty(3, dtype=torch.int64),
                valid_x,
                {"group_type": 0, "group_list_type": 2},
            ),
        ]
        for tensor, args, kwargs in cases:
            with self.subTest(shape=tensor.shape, kwargs=kwargs):
                self.assertIsNone(
                    npu_overlap._balanced_group_list_for_benchmark(
                        tensor, args, kwargs, self._hint
                    )
                )


class TestGroupedMatmulBenchmarkWrapper(TestCase):
    def setUp(self):
        super().setUp()
        self.packet = object()
        self.target = _CallableTarget(self.packet)
        self.node = types.SimpleNamespace(target=self.target)
        self.group_list = torch.tensor([0, 0, 0], dtype=torch.int64)
        self.fake_args = (
            [torch.empty(10, 8)],
            [torch.empty(3, 8, 4)],
        )
        self.fake_kwargs = {
            "group_list": self.group_list,
            "split_item": 3,
            "group_type": 0,
            "group_list_type": 1,
        }
        self.upstream = mock.Mock(return_value=(9.0, "upstream-key"))
        self.overlap = types.SimpleNamespace(
            get_custom_estimation=mock.Mock(return_value=None),
            get_hint=lambda value: int(value),
            get_cached_node_time=mock.Mock(return_value=None),
            set_cached_node_time=mock.Mock(),
            get_collective_do_bench=mock.Mock(),
        )

    def _build_wrapper(self):
        with mock.patch.object(
            npu_overlap,
            "_get_registered_npu_op_packet",
            return_value=self.packet,
        ):
            return npu_overlap._build_npu_benchmark_node_with_cache_key(
                self.upstream, self.overlap
            )

    def _fake_args_context(self):
        return mock.patch.object(
            inductor_fx_utils,
            "get_fake_args_kwargs",
            return_value=(True, self.fake_args, self.fake_kwargs),
        )

    def test_non_gmm_node_delegates_to_upstream_unchanged(self):
        wrapper = self._build_wrapper()
        other_node = types.SimpleNamespace(
            target=_CallableTarget(object(), name="other")
        )
        estimator = mock.Mock()
        self.assertEqual(wrapper(other_node, estimator), (9.0, "upstream-key"))
        self.upstream.assert_called_once_with(other_node, estimator)

    def test_custom_estimation_short_circuits_input_generation(self):
        wrapper = self._build_wrapper()
        estimator = mock.Mock()
        self.overlap.get_custom_estimation.return_value = 4.25
        with mock.patch.object(
            inductor_fx_utils, "get_fake_args_kwargs"
        ) as get_fake:
            self.assertEqual(wrapper(self.node, estimator), (4.25, None))
        get_fake.assert_not_called()
        self.overlap.get_custom_estimation.assert_called_once_with(
            self.node, estimator, None
        )

    def test_invalid_fake_inputs_return_zero_without_benchmark(self):
        wrapper = self._build_wrapper()
        with mock.patch.object(
            inductor_fx_utils,
            "get_fake_args_kwargs",
            return_value=(False, (), {}),
        ):
            self.assertEqual(wrapper(self.node), (0.0, None))
        self.overlap.get_collective_do_bench.assert_not_called()

    def test_cache_hit_reuses_upstream_style_key(self):
        wrapper = self._build_wrapper()
        self.overlap.get_cached_node_time.return_value = 1.75
        with self._fake_args_context():
            runtime, key = wrapper(self.node)

        self.assertEqual(runtime, 1.75)
        self.assertTrue(key.startswith(f"{self.target}: "))
        self.assertIn("T: ([10, 8], [8, 1], torch.float32)", key)
        self.assertNotIn("gmm_balanced", key)
        self.target.calls.clear()
        self.overlap.get_collective_do_bench.assert_not_called()
        self.overlap.set_cached_node_time.assert_not_called()

    def test_cache_miss_benchmarks_nonzero_balanced_group_list(self):
        wrapper = self._build_wrapper()

        def bench(fn):
            fn()
            return 1.25

        self.overlap.get_collective_do_bench.return_value = bench
        with self._fake_args_context():
            runtime, key = wrapper(self.node)

        self.assertEqual(runtime, 1.25)
        self.assertEqual(len(self.target.calls), 1)
        _, call_kwargs = self.target.calls[0]
        generated = call_kwargs["group_list"]
        self.assertEqual(generated.tolist(), [4, 3, 3])
        self.assertEqual(sum(generated.tolist()), 10)
        self.overlap.set_cached_node_time.assert_called_once_with(key, 1.25)

    def test_cache_key_depends_on_metadata_not_group_list_values(self):
        wrapper = self._build_wrapper()
        self.overlap.get_collective_do_bench.return_value = lambda fn: 1.0
        keys = []
        for values in ([0, 0, 0], [4, 3, 3]):
            kwargs = dict(self.fake_kwargs)
            kwargs["group_list"] = torch.tensor(values, dtype=torch.int64)
            with mock.patch.object(
                inductor_fx_utils,
                "get_fake_args_kwargs",
                return_value=(True, self.fake_args, kwargs),
            ):
                _, key = wrapper(self.node)
                keys.append(key)
        self.assertEqual(keys[0], keys[1])

    def test_unbacked_tensor_does_not_run_target(self):
        wrapper = self._build_wrapper()
        self.overlap.get_hint = lambda value: None
        with self._fake_args_context():
            runtime, key = wrapper(self.node)
        self.assertEqual(runtime, 0.0)
        self.assertIsNotNone(key)
        self.assertEqual(self.target.calls, [])
        self.overlap.get_collective_do_bench.assert_not_called()


class TestEstimatorAndScheduleGuards(TestCase):
    def test_collective_estimator_prefers_custom_then_benchmark(self):
        overlap = types.SimpleNamespace(
            get_custom_estimation=mock.Mock(return_value=0.25)
        )
        runtime_estimation = types.SimpleNamespace(
            benchmark_collective_with_cuda_events=mock.Mock(return_value=(0.5, "k"))
        )
        estimate = npu_overlap._build_npu_estimate_collective_time(
            overlap, runtime_estimation
        )
        node = object()
        custom = mock.Mock()

        self.assertEqual(estimate(node, 123, custom, "benchmark"), 0.25)
        overlap.get_custom_estimation.assert_called_once_with(node, custom, 123)
        runtime_estimation.benchmark_collective_with_cuda_events.assert_not_called()

        overlap.get_custom_estimation.return_value = None
        self.assertEqual(estimate(node, None, None, "benchmark"), 0.5)
        runtime_estimation.benchmark_collective_with_cuda_events.assert_called_once_with(
            node, nruns=5
        )

    def test_collective_estimator_returns_safe_zero_without_hccl_model(self):
        overlap = types.SimpleNamespace(get_custom_estimation=lambda *args: None)
        runtime_estimation = types.SimpleNamespace(
            benchmark_collective_with_cuda_events=mock.Mock(
                return_value=(None, "")
            )
        )
        estimate = npu_overlap._build_npu_estimate_collective_time(
            overlap, runtime_estimation
        )
        self.assertEqual(estimate(object(), collective_estimator="analytical"), 0.0)
        self.assertEqual(estimate(object(), collective_estimator="benchmark"), 0.0)

    def test_collective_logger_emits_supplied_aligned_medians(self):
        runtime = types.SimpleNamespace(
            _get_collective_key=lambda node: f"key-{node}",
            _format_csv=lambda headers, rows: f"{headers!r}\n{rows!r}",
            trace_structured=mock.Mock(),
        )
        logger = npu_overlap._build_npu_collective_logger(runtime)
        logger(
            ["a", "b"],
            collective_keys=["ka", "kb"],
            benchmarked_medians=[0.125, 2.5],
            world_size=8,
            artifact_name="test_collectives",
        )

        call = runtime.trace_structured.call_args
        self.assertEqual(call.args[0], "artifact")
        self.assertEqual(
            call.kwargs["metadata_fn"](),
            {"name": "test_collectives", "encoding": "string"},
        )
        payload = call.kwargs["payload_fn"]()
        self.assertIn("# World size: 8", payload)
        self.assertIn("0.1250", payload)
        self.assertIn("2.5000", payload)

    def test_schedule_entry_rejects_unsupported_estimators(self):
        upstream = mock.Mock(return_value="gm")

        def schedule(gm, compute_estimator, collective_estimator):
            return upstream(gm, compute_estimator, collective_estimator)

        wrapped = npu_overlap._build_npu_schedule_entry(schedule)
        with torch._inductor.config.patch(deterministic=False):
            with self.assertRaisesRegex(NotImplementedError, "analytical compute"):
                wrapped("gm", "analytical", "benchmark")
            with self.assertRaisesRegex(NotImplementedError, "analytical collective"):
                wrapped("gm", "benchmark", "analytical")
            self.assertEqual(wrapped("gm", "benchmark", "benchmark"), "gm")

    def test_schedule_entry_rejects_deterministic_mode(self):
        def schedule(gm, compute_estimator, collective_estimator):
            return gm

        wrapped = npu_overlap._build_npu_schedule_entry(schedule)
        with torch._inductor.config.patch(deterministic=True):
            with self.assertRaisesRegex(NotImplementedError, "deterministic mode"):
                wrapped("gm", "benchmark", "benchmark")

    def test_unsupported_npu_roofline_is_zero(self):
        self.assertEqual(npu_overlap._unsupported_npu_roofline_estimation(object()), 0.0)


if __name__ == "__main__":
    unittest.main()
