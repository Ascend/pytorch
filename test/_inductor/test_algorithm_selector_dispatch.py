import contextlib
import inspect
from types import SimpleNamespace
from unittest import mock

from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch.testing._internal.common_utils import TestCase, run_tests


_UPSTREAM_CALL = AlgorithmSelectorCache.__call__
_UPSTREAM_MAKE_BENCHMARK_FN = AlgorithmSelectorCache.__dict__[
    "make_benchmark_fn"
].__func__

import torch_npu._inductor.select_algorithm as npu_select_algorithm  # noqa: E402


def _layout(device_type):
    return SimpleNamespace(device=SimpleNamespace(type=device_type))


@contextlib.contextmanager
def _install_with_parents(parent_call, parent_make_benchmark_fn):
    with (
        mock.patch.object(AlgorithmSelectorCache, "__call__", parent_call),
        mock.patch.object(
            AlgorithmSelectorCache,
            "make_benchmark_fn",
            classmethod(parent_make_benchmark_fn),
        ),
    ):
        npu_select_algorithm.patch_algorithm_selector()
        yield (
            AlgorithmSelectorCache.__call__,
            AlgorithmSelectorCache.__dict__["make_benchmark_fn"].__func__,
        )


class TestAlgorithmSelectorDispatch(TestCase):
    def test_wrappers_preserve_v213_signatures(self):
        with _install_with_parents(
            _UPSTREAM_CALL, _UPSTREAM_MAKE_BENCHMARK_FN
        ) as (patched_call, patched_make_benchmark_fn):
            self.assertIs(patched_call.__wrapped__, _UPSTREAM_CALL)
            self.assertIs(
                patched_make_benchmark_fn.__wrapped__,
                _UPSTREAM_MAKE_BENCHMARK_FN,
            )
            self.assertEqual(
                inspect.signature(patched_call),
                inspect.signature(_UPSTREAM_CALL),
            )
            self.assertEqual(
                inspect.signature(patched_make_benchmark_fn),
                inspect.signature(_UPSTREAM_MAKE_BENCHMARK_FN),
            )

    def test_non_npu_call_delegates_and_preserves_parent_result(self):
        choice = SimpleNamespace(output_node=mock.Mock())
        parent_result = (object(), object())

        for device_type in ("cpu", "cuda", "xpu"):
            with self.subTest(device_type=device_type):
                parent_call = mock.Mock(return_value=parent_result)
                layout = _layout(device_type)

                with _install_with_parents(parent_call, mock.Mock()) as (
                    patched_call,
                    _,
                ):
                    result = patched_call(
                        object(),
                        "test",
                        [choice],
                        [],
                        layout,
                        best_config_future=object(),
                    )

                self.assertIs(result, parent_result)
                parent_call.assert_called_once()
                choice.output_node.assert_not_called()

    def test_plain_npu_choice_returns_tuple_abi(self):
        selected_node = object()
        choice = SimpleNamespace(output_node=mock.Mock(return_value=selected_node))
        parent_call = mock.Mock()

        with _install_with_parents(parent_call, mock.Mock()) as (patched_call, _):
            result = patched_call(
                object(),
                "test",
                [choice],
                [],
                _layout("npu"),
                best_config_future=object(),
                is_collective=True,
                min_speedup_threshold=1.5,
                benchmark_with_cudagraphs=True,
            )

        self.assertIs(result[0], selected_node)
        self.assertIs(result[1], choice)
        parent_call.assert_not_called()


if __name__ == "__main__":
    run_tests()
