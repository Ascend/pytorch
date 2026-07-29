from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu._inductor.lowering_patch as lowering_patch
from torch_npu._inductor.lowering_common import LOWERING_REGISTRY_ATTRS


class _IRValue:
    def __init__(self, device_type, dtype=torch.float32, size=(1,)):
        self.device = torch.device(device_type)
        self.dtype = dtype
        self.size = size

    def get_device(self):
        return self.device

    def get_dtype(self):
        return self.dtype

    def get_size(self):
        return self.size


class TestLoweringDeviceDispatch(TestCase):
    def setUp(self):
        super().setUp()
        self.target = "target"
        self.upstream_call = mock.Mock(return_value="upstream")
        self.device_call = mock.Mock(return_value="npu")

        def upstream(*args, **kwargs):
            return self.upstream_call(*args, **kwargs)

        def device_handler(*args, **kwargs):
            return self.device_call(*args, **kwargs)

        self.upstream = upstream
        self.device_handler = device_handler
        self.registry = {self.target: self.device_handler}
        self.make_reduction = object()
        self.lowering = SimpleNamespace(
            lowerings=self.registry,
            make_reduction=self.make_reduction,
        )
        registry_copies = {}
        for attr in LOWERING_REGISTRY_ATTRS:
            if attr == "lowerings":
                registry_copies[attr] = {self.target: self.upstream}
                continue
            value = {}
            setattr(self.lowering, attr, value)
            registry_copies[attr] = {}

        self.baseline = lowering_patch.LoweringSnapshot(
            functions={},
            lowerings_ref=self.registry,
            lowerings_copy={self.target: self.upstream},
            registry_copies=registry_copies,
            make_reduction=self.make_reduction,
        )
        self.get_lowering_patch = mock.patch.object(
            lowering_patch,
            "_get_inductor_lowering",
            return_value=self.lowering,
        )
        self.capture_patch = mock.patch.object(
            lowering_patch,
            "capture_lowering_baseline",
            return_value=self.baseline,
        )
        self.get_lowering_patch.start()
        self.capture_patch.start()

    def tearDown(self):
        self.capture_patch.stop()
        self.get_lowering_patch.stop()
        super().tearDown()

    def install(self, targets=None, extra_device_predicates=None):
        targets = targets or [self.target]
        registry_id = id(self.registry)
        lowering_patch.install_device_lowering_dispatch(
            targets,
            extra_device_predicates=extra_device_predicates,
        )
        self.assertEqual(id(self.registry), registry_id)
        return self.registry[targets[0]]

    def _register_handlers(self, target):
        self.baseline.lowerings_copy[target] = self.upstream
        self.registry[target] = self.device_handler

    def _dispatch_cpu_cast(
        self,
        *,
        source_dtype=torch.float64,
        source_size=(),
        destination_dtype=torch.float32,
        graph_device_types=("cpu", "npu"),
        layout_device_type=None,
    ):
        target = torch.ops.prims.convert_element_type
        self._register_handlers(target)
        kwargs = {}
        if layout_device_type is not None:
            kwargs["layout"] = SimpleNamespace(
                device=torch.device(layout_device_type)
            )

        with V.set_graph_handler(
            SimpleNamespace(device_types=graph_device_types)
        ):
            dispatcher = self.install([target])
            return dispatcher(
                _IRValue("cpu", dtype=source_dtype, size=source_size),
                destination_dtype,
                **kwargs,
            )

    def test_cpu_scalar_fp64_cast_in_npu_graph_uses_device_handler(self):
        result = self._dispatch_cpu_cast()

        self.assertEqual(result, "npu")
        self.device_call.assert_called_once()
        self.upstream_call.assert_not_called()

    def test_cpu_scalar_fp64_cast_in_cpu_graph_uses_upstream_handler(self):
        result = self._dispatch_cpu_cast(graph_device_types=("cpu",))

        self.assertEqual(result, "upstream")
        self.upstream_call.assert_called_once()
        self.device_call.assert_not_called()

    def test_custom_predicates_are_merged_with_defaults(self):
        convert = torch.ops.prims.convert_element_type
        custom_target = "custom_target"
        self._register_handlers(convert)
        self._register_handlers(custom_target)

        def custom_predicate(*args, **kwargs):
            return True

        with V.set_graph_handler(SimpleNamespace(device_types=("cpu", "npu"))):
            self.install(
                [convert, custom_target],
                extra_device_predicates={custom_target: custom_predicate},
            )
            convert_result = self.registry[convert](
                _IRValue("cpu", dtype=torch.float64, size=()), torch.float32
            )
            custom_result = self.registry[custom_target](_IRValue("cpu"))

        self.assertEqual(convert_result, "npu")
        self.assertEqual(custom_result, "npu")

    def test_cpu_non_scalar_fp64_cast_in_npu_graph_uses_upstream_handler(self):
        result = self._dispatch_cpu_cast(source_size=(1,))

        self.assertEqual(result, "upstream")
        self.upstream_call.assert_called_once()
        self.device_call.assert_not_called()

    def test_cpu_scalar_non_fp64_cast_in_npu_graph_uses_upstream_handler(self):
        result = self._dispatch_cpu_cast(
            source_dtype=torch.float32,
            destination_dtype=torch.bfloat16,
        )

        self.assertEqual(result, "upstream")
        self.upstream_call.assert_called_once()
        self.device_call.assert_not_called()

    def test_cpu_scalar_fp64_noop_in_npu_graph_uses_upstream_handler(self):
        result = self._dispatch_cpu_cast(destination_dtype=torch.float64)

        self.assertEqual(result, "upstream")
        self.upstream_call.assert_called_once()
        self.device_call.assert_not_called()

    def test_non_npu_layout_overrides_extra_device_predicate(self):
        result = self._dispatch_cpu_cast(layout_device_type="cpu")

        self.assertEqual(result, "upstream")
        self.upstream_call.assert_called_once()
        self.device_call.assert_not_called()

    def test_non_npu_layout_uses_upstream_handler(self):
        dispatcher = self.install()

        for device_type in ("cpu", "cuda", "xpu"):
            with self.subTest(device_type=device_type):
                result = dispatcher(
                    _IRValue(device_type),
                    layout=SimpleNamespace(device=torch.device(device_type)),
                )

                self.assertEqual(result, "upstream")

        self.assertEqual(self.upstream_call.call_count, 3)
        self.device_call.assert_not_called()

    def test_npu_layout_wins_over_cpu_metadata(self):
        dispatcher = self.install()

        result = dispatcher(
            _IRValue("cpu"),
            layout=SimpleNamespace(device=torch.device("npu")),
        )

        self.assertEqual(result, "npu")
        self.device_call.assert_called_once()
        self.upstream_call.assert_not_called()

    def test_non_npu_layout_wins_over_npu_input(self):
        dispatcher = self.install()

        result = dispatcher(
            _IRValue("npu"),
            layout=SimpleNamespace(device=torch.device("cpu")),
        )

        self.assertEqual(result, "upstream")
        self.upstream_call.assert_called_once()
        self.device_call.assert_not_called()

    def test_npu_input_without_layout_uses_device_handler(self):
        dispatcher = self.install()

        result = dispatcher({"nested": [_IRValue("npu")]})

        self.assertEqual(result, "npu")
        self.device_call.assert_called_once()
        self.upstream_call.assert_not_called()

    def test_missing_device_information_uses_upstream(self):
        dispatcher = self.install()

        result = dispatcher(1, flag=True)

        self.assertEqual(result, "upstream")
        self.upstream_call.assert_called_once()
        self.device_call.assert_not_called()

    def test_missing_upstream_handler_leaves_target_unchanged(self):
        self.baseline.lowerings_copy.clear()

        dispatcher = self.install()

        self.assertIs(dispatcher, self.device_handler)

    def test_repeated_install_is_idempotent(self):
        first = self.install()

        second = self.install()

        self.assertIs(second, first)
        self.assertTrue(
            getattr(second, "_torch_npu_device_lowering_dispatch", False)
        )

    def test_multiple_targets_capture_their_own_handlers(self):
        second_target = "second_target"
        second_upstream_call = mock.Mock(return_value="second_upstream")
        second_device_call = mock.Mock(return_value="second_npu")

        def second_upstream(*args, **kwargs):
            return second_upstream_call(*args, **kwargs)

        def second_device(*args, **kwargs):
            return second_device_call(*args, **kwargs)

        self.baseline.lowerings_copy[second_target] = second_upstream
        self.registry[second_target] = second_device

        self.install([self.target, second_target])

        self.assertEqual(self.registry[self.target](_IRValue("cpu")), "upstream")
        self.assertEqual(
            self.registry[second_target](_IRValue("npu")), "second_npu"
        )
        self.upstream_call.assert_called_once()
        second_device_call.assert_called_once()

    def test_op_packet_expands_registered_overloads(self):
        packet = torch.ops.aten.mm
        overload = packet.default

        def predicate(*args, **kwargs):
            return True

        def packet_upstream(*args, **kwargs):
            return "packet_upstream"

        def packet_device(*args, **kwargs):
            return "packet_npu"

        def overload_upstream(*args, **kwargs):
            return "overload_upstream"

        def overload_device(*args, **kwargs):
            return "overload_npu"

        self.baseline.lowerings_copy.update(
            {
                packet: packet_upstream,
                overload: overload_upstream,
            }
        )
        self.registry.update(
            {
                packet: packet_device,
                overload: overload_device,
            }
        )

        lowering_patch.install_device_lowering_dispatch(
            [packet], extra_device_predicates={packet: predicate}
        )

        self.assertTrue(
            getattr(
                self.registry[packet],
                "_torch_npu_device_lowering_dispatch",
                False,
            )
        )
        self.assertTrue(
            getattr(
                self.registry[overload],
                "_torch_npu_device_lowering_dispatch",
                False,
            )
        )
        self.assertEqual(self.registry[overload](_IRValue("cpu")), "overload_npu")

    def test_restore_removes_dispatcher_in_place(self):
        registry_id = id(self.registry)
        self.install()

        lowering_patch.restore_lowering_baseline()

        self.assertEqual(id(self.registry), registry_id)
        self.assertIs(self.registry[self.target], self.upstream)


if __name__ == "__main__":
    run_tests()
