# Owner(s): ["module: dynamo"]
import copy
import functools
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch._dynamo.test_case import TestCase
from torch.overrides import TorchFunctionMode


FilterFn = Callable[[list], list[bool]]

_DICT_GUARD_TYPES = frozenset(
    {
        "DICT_VERSION",
        "DICT_KEYS",
        "DICT_KEYS_MATCH",
        "DICT_CONTAINS",
    }
)

_OPTIONAL_TYPE_GUARD_TYPES = frozenset(
    {
        "TYPE_MATCH",
        "OPTIONAL_TENSOR",
    }
)

_HASATTR_GUARD_TYPES = frozenset(
    {
        "HASATTR",
        "NOT_PRESENT_IN_GENERIC_DICT",
    }
)

_RUNTIME_STATE_GUARD_TYPES = frozenset(
    {
        "GRAD_MODE",
        "TORCH_FUNCTION_STATE",
        "GLOBAL_STATE",
        "DEFAULT_DEVICE",
        "DETERMINISTIC_ALGORITHMS",
        "AUTOCAST_STATE",
        "FSDP_TRAINING_STATE",
    }
)

_ORIGINAL_TORCH_COMPILE = None
_TORCH_COMPILE_STACK = []


def _entry_type(entry) -> str:
    return getattr(entry, "guard_type", "") or ""


def make_npu_guard_filter(
    *,
    disable_dict_version: bool = False,
    disable_optional_type: bool = False,
    disable_hasattr: bool = False,
    disable_runtime_state: bool = False,
) -> FilterFn:
    def _filter(entries) -> list[bool]:
        keep = [True] * len(entries)
        for i, e in enumerate(entries):
            t = _entry_type(e)
            if disable_dict_version and t in _DICT_GUARD_TYPES:
                keep[i] = False
                continue
            if disable_optional_type and t in _OPTIONAL_TYPE_GUARD_TYPES:
                keep[i] = False
                continue
            if disable_hasattr and t in _HASATTR_GUARD_TYPES:
                keep[i] = False
                continue
            if disable_runtime_state and t in _RUNTIME_STATE_GUARD_TYPES:
                keep[i] = False
                continue
        return keep

    return _filter


def _supports_dynamo_guard_filter_config() -> bool:
    import torch._dynamo.config as cfg

    return hasattr(cfg, "_config") and "guard_filter_fn" in cfg._config


def _set_wrapped_torch_compile(filter_fn: FilterFn | None) -> None:
    global _ORIGINAL_TORCH_COMPILE
    if _ORIGINAL_TORCH_COMPILE is None:
        _ORIGINAL_TORCH_COMPILE = torch.compile

    if filter_fn is None:
        torch.compile = _ORIGINAL_TORCH_COMPILE
        return

    @functools.wraps(_ORIGINAL_TORCH_COMPILE)
    def wrapped_compile(model=None, *args, **kwargs):
        options = kwargs.get("options")
        if options is None:
            options = {}
        else:
            options = dict(options)
        options.setdefault("guard_filter_fn", filter_fn)
        kwargs["options"] = options
        return _ORIGINAL_TORCH_COMPILE(model, *args, **kwargs)

    torch.compile = wrapped_compile


def _install_global_guard_filter(filter_fn: FilterFn):
    if _supports_dynamo_guard_filter_config():
        import torch._dynamo.config as cfg

        prev = getattr(cfg, "guard_filter_fn", None)
        cfg.guard_filter_fn = filter_fn
        return ("dynamo_config", prev)

    previous = _TORCH_COMPILE_STACK[-1] if _TORCH_COMPILE_STACK else None
    _TORCH_COMPILE_STACK.append(filter_fn)
    _set_wrapped_torch_compile(filter_fn)
    return ("torch_compile", previous)


def _restore_global_guard_filter(state) -> None:
    kind, prev = state
    if kind == "dynamo_config":
        import torch._dynamo.config as cfg

        cfg.guard_filter_fn = prev
        return

    if _TORCH_COMPILE_STACK:
        _TORCH_COMPILE_STACK.pop()
    if _TORCH_COMPILE_STACK:
        _set_wrapped_torch_compile(_TORCH_COMPILE_STACK[-1])
    else:
        _set_wrapped_torch_compile(None)


def _get_installed_guard_filter():
    if _supports_dynamo_guard_filter_config():
        import torch._dynamo.config as cfg

        return getattr(cfg, "guard_filter_fn", None)

    compile_fn = torch.compile
    closure = getattr(compile_fn, "__closure__", None) or ()
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value) and getattr(value, "__name__", "") == "_filter":
            return value
    return None


class NpuGuardPolicy:
    def __init__(
        self,
        *,
        filter_fn: FilterFn | None = None,
        disable_dict_version: bool = False,
        disable_optional_type: bool = False,
        disable_hasattr: bool = False,
        disable_runtime_state: bool = False,
    ):
        switches_used = any(
            [
                disable_dict_version,
                disable_optional_type,
                disable_hasattr,
                disable_runtime_state,
            ]
        )
        if filter_fn is not None and switches_used:
            raise ValueError("filter_fn and disable_* switches are mutually exclusive")

        if filter_fn is not None:
            self._fn = filter_fn
        else:
            self._fn = make_npu_guard_filter(
                disable_dict_version=disable_dict_version,
                disable_optional_type=disable_optional_type,
                disable_hasattr=disable_hasattr,
                disable_runtime_state=disable_runtime_state,
            )
        self._prev = None
        self._active = False

    def __enter__(self) -> "NpuGuardPolicy":
        self._prev = _install_global_guard_filter(self._fn)
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._active:
            return
        _restore_global_guard_filter(self._prev)
        self._prev = None
        self._active = False


@dataclass
class FakeEntry:
    guard_type: str
    name: str = ""
    is_global: bool = False
    has_value: bool = False
    value: Any = None


def _entries(*types: str):
    return [FakeEntry(guard_type=t, name=f"L[{i}]") for i, t in enumerate(types)]


def _make_mlp(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


def _recompile_count() -> int:
    from torch._dynamo.utils import counters

    return sum(counters["recompiles"].values())


def _reset_dynamo() -> None:
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()


class GlobalTorchFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class TestDictDimension(TestCase):
    def test_drops_dict_family(self):
        f = make_npu_guard_filter(disable_dict_version=True)
        es = _entries(
            "DICT_VERSION", "DICT_KEYS", "DICT_CONTAINS", "TENSOR_MATCH", "TYPE_MATCH"
        )
        self.assertEqual(f(es), [False, False, False, True, True])

    def test_inactive_keeps_all(self):
        f = make_npu_guard_filter()
        es = _entries("DICT_VERSION", "TENSOR_MATCH")
        self.assertEqual(f(es), [True, True])


class TestOptionalTypeDimension(TestCase):
    def test_drops_type_family(self):
        f = make_npu_guard_filter(disable_optional_type=True)
        es = _entries("TYPE_MATCH", "OPTIONAL_TENSOR", "TENSOR_MATCH", "DICT_VERSION")
        self.assertEqual(f(es), [False, False, True, True])


class TestHasattrDimension(TestCase):
    def test_drops_hasattr_family(self):
        f = make_npu_guard_filter(disable_hasattr=True)
        es = _entries("HASATTR", "NOT_PRESENT_IN_GENERIC_DICT", "TENSOR_MATCH")
        self.assertEqual(f(es), [False, False, True])


class TestRuntimeStateDimension(TestCase):
    def test_drops_runtime_state_family(self):
        f = make_npu_guard_filter(disable_runtime_state=True)
        es = _entries(
            "GRAD_MODE",
            "TORCH_FUNCTION_STATE",
            "GLOBAL_STATE",
            "DEFAULT_DEVICE",
            "DETERMINISTIC_ALGORITHMS",
            "AUTOCAST_STATE",
            "FSDP_TRAINING_STATE",
            "TENSOR_MATCH",
        )
        self.assertEqual(f(es), [False] * 7 + [True])


class TestUpstreamGuardFilterCoverage(TestCase):
    def setUp(self):
        super().setUp()
        _reset_dynamo()

    def test_guard_filter_fn_by_id(self):
        def guard_filter_fn(entries):
            return [entry.guard_type != "ID_MATCH" for entry in entries]

        @torch.compile(fullgraph=True, options={"guard_filter_fn": guard_filter_fn})
        def fn(x):
            return id(x)

        inputs = (torch.randn(3, 2),)
        fn(*inputs)

        inputs_1 = (torch.randn(3, 2),)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs_1), id(inputs[0]))

    def test_guard_filter_fn_by_is_global(self):
        def guard_filter_fn(entries):
            return [not entry.is_global for entry in entries]

        global GLOBAL_INT

        @torch.compile(fullgraph=True, options={"guard_filter_fn": guard_filter_fn})
        def fn(x):
            return x + GLOBAL_INT

        GLOBAL_INT = 1
        fn(torch.randn(3, 2))

        GLOBAL_INT = 2
        inputs = (torch.randn(3, 2),)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertTrue(torch.equal(fn(*inputs), inputs[0] + 1))

    def test_guard_filter_fn_by_name_and_value(self):
        def guard_filter_fn(entries):
            return [
                not (entry.name == "y" and entry.value is None) for entry in entries
            ]

        @torch.compile(fullgraph=True, options={"guard_filter_fn": guard_filter_fn})
        def fn(x, y):
            if y is not None:
                x += y
            return x

        fn(torch.randn(3, 2), None)

        inputs = (torch.randn(3, 2), torch.tensor(1))
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertTrue(torch.equal(fn(*inputs), inputs[0]))

    def test_guard_filter_inbuilt_nn_modules(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                return self.norm(x)

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.norm.eps = 1e-02
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)


class TestCombinedSwitches(TestCase):
    def test_union_drop(self):
        f = make_npu_guard_filter(
            disable_dict_version=True,
            disable_optional_type=True,
            disable_hasattr=True,
            disable_runtime_state=True,
        )
        es = _entries(
            "DICT_VERSION",
            "TYPE_MATCH",
            "HASATTR",
            "GRAD_MODE",
            "TENSOR_MATCH",
            "ID_MATCH",
        )
        self.assertEqual(f(es), [False, False, False, False, True, True])

    def test_returns_list_of_correct_length(self):
        f = make_npu_guard_filter(disable_dict_version=True)
        es = _entries("DICT_VERSION", "TENSOR_MATCH", "TYPE_MATCH")
        keep = f(es)
        self.assertIsInstance(keep, list)
        self.assertEqual(len(keep), len(es))


class TestPolicy(TestCase):
    def setUp(self):
        super().setUp()
        _reset_dynamo()

    def test_installs_and_restores(self):
        prev = _get_installed_guard_filter()
        with NpuGuardPolicy(disable_dict_version=True):
            self.assertIsNotNone(_get_installed_guard_filter())
            self.assertNotEqual(_get_installed_guard_filter(), prev)
        self.assertEqual(_get_installed_guard_filter(), prev)

    def test_restores_on_exception(self):
        prev = _get_installed_guard_filter()
        with self.assertRaises(RuntimeError):
            with NpuGuardPolicy(disable_runtime_state=True):
                raise RuntimeError("boom")
        self.assertEqual(_get_installed_guard_filter(), prev)

    def test_nesting(self):
        prev = _get_installed_guard_filter()
        with NpuGuardPolicy(disable_dict_version=True):
            outer_fn = _get_installed_guard_filter()
            with NpuGuardPolicy(disable_runtime_state=True):
                self.assertNotEqual(_get_installed_guard_filter(), outer_fn)
            self.assertEqual(_get_installed_guard_filter(), outer_fn)
        self.assertEqual(_get_installed_guard_filter(), prev)

    def test_with_custom_filter_fn(self):
        called = {"n": 0}

        def my_filter(entries):
            called["n"] += 1
            return [True] * len(entries)

        with NpuGuardPolicy(filter_fn=my_filter):
            c = torch.compile(_make_mlp())
            c(torch.randn(2, 8))
        self.assertGreaterEqual(called["n"], 1)

    def test_filter_fn_and_switches_are_exclusive(self):
        with self.assertRaises(ValueError):
            NpuGuardPolicy(
                filter_fn=lambda e: [True] * len(e), disable_dict_version=True
            )


class TestParity(TestCase):
    def setUp(self):
        super().setUp()
        _reset_dynamo()

    def _parity(self, **switches):
        torch.manual_seed(0)
        m = _make_mlp().eval()
        x = torch.randn(2, 8)
        with torch.no_grad():
            ref = m(x)

        m2 = copy.deepcopy(m)
        c = torch.compile(
            m2, options={"guard_filter_fn": make_npu_guard_filter(**switches)}
        )
        with torch.no_grad():
            got = c(x)
        self.assertTrue(torch.allclose(got, ref, atol=1e-5, rtol=1e-5))

    def test_parity_dict(self):
        self._parity(disable_dict_version=True)

    def test_parity_optional_type(self):
        self._parity(disable_optional_type=True)

    def test_parity_hasattr(self):
        self._parity(disable_hasattr=True)

    def test_parity_runtime_state(self):
        self._parity(disable_runtime_state=True)

    def test_parity_all(self):
        self._parity(
            disable_dict_version=True,
            disable_optional_type=True,
            disable_hasattr=True,
            disable_runtime_state=True,
        )


class TestRuntimeStateRecompileBehavior(TestCase):
    def setUp(self):
        super().setUp()
        _reset_dynamo()

    def test_no_recompile_across_grad_mode_change(self):
        def foo(x):
            return x + 1

        x = torch.randn(3, 2)
        compiled_fn = torch.compile(
            foo,
            options={
                "guard_filter_fn": make_npu_guard_filter(disable_runtime_state=True)
            },
        )

        with torch.no_grad():
            compiled_fn(x)

        with torch.enable_grad(), torch.compiler.set_stance("fail_on_recompile"):
            self.assertTrue(torch.equal(compiled_fn(x), foo(x)))

    def test_no_recompile_across_torch_function_mode_change(self):
        def foo(x):
            return x + 1

        x = torch.randn(3, 2)
        with GlobalTorchFunctionMode():
            compiled_fn = torch.compile(
                foo,
                options={
                    "guard_filter_fn": make_npu_guard_filter(disable_runtime_state=True)
                },
            )
            compiled_fn(x)

        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertTrue(torch.equal(compiled_fn(x), foo(x)))


class TestRecompileCount(TestCase):
    def setUp(self):
        super().setUp()
        _reset_dynamo()

    def test_no_recompile_on_repeated_same_shape(self):
        m = _make_mlp()
        c = torch.compile(
            m,
            options={
                "guard_filter_fn": make_npu_guard_filter(
                    disable_dict_version=True,
                    disable_optional_type=True,
                    disable_hasattr=True,
                    disable_runtime_state=True,
                )
            },
        )
        for _ in range(5):
            c(torch.randn(2, 8))
        self.assertEqual(_recompile_count(), 0)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
