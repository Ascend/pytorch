"""CPU-only tests for the lazy-fusion Python wrappers, using a fake binding."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class TestLazyFusionAPI(unittest.TestCase):
    def setUp(self):
        self.state = {"disabled": False, "dump": True}
        self.calls = []

        def setter(key):
            def update(value):
                previous = self.state[key]
                self.state[key] = value
                self.calls.append((key, value))
                return previous
            return update

        binding = types.SimpleNamespace(
            _set_disabled=setter("disabled"),
            _set_dump_enabled=setter("dump"),
        )
        # Deliberately provide no _C.dvm/Inductor namespace.
        package = types.ModuleType("torch_npu")
        package._C = types.SimpleNamespace(_lazy_fusion=binding)
        source = Path(__file__).resolve().parents[1] / "torch_npu/npu/lazy_fusion.py"
        spec = importlib.util.spec_from_file_location("lazy_fusion_under_test", source)
        self.api = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch_npu": package}):
            spec.loader.exec_module(self.api)
        self.assertEqual(self.calls, [])  # Import must not change runtime state.

    def test_fusion_nested_exception_restores_state(self):
        with self.api.disabled():
            self.assertTrue(self.state["disabled"])
            with self.assertRaisesRegex(RuntimeError, "test"):
                with self.api.enabled():
                    self.assertFalse(self.state["disabled"])
                    raise RuntimeError("test")
            self.assertTrue(self.state["disabled"])
        self.assertFalse(self.state["disabled"])

    def test_no_argument_setters(self):
        for enable_name, disable_name, key, enabled_value in (
            ("set_enable", "set_disable", "disabled", False),
            ("set_dump_enable", "set_dump_disable", "dump", True),
        ):
            self.assertIn(enable_name, self.api.__all__)
            self.assertIn(disable_name, self.api.__all__)
            for enabled in (False, False, True, True, False):
                setter = getattr(self.api, enable_name if enabled else disable_name)
                self.assertIsNone(setter())
                self.assertEqual(self.state[key], enabled_value if enabled else not enabled_value)
        self.assertFalse(hasattr(self.api, "set_enabled"))
        self.assertFalse(hasattr(self.api, "set_dump_enabled"))

    def test_set_enabled_and_context_restore(self):
        self.api.set_disable()
        with self.api.enabled():
            self.assertFalse(self.state["disabled"])
        self.assertTrue(self.state["disabled"])
        self.api.set_enable()
        with self.assertRaisesRegex(RuntimeError, "test"):
            with self.api.disabled():
                self.api.set_enable()
                self.api.set_disable()
                raise RuntimeError("test")
        self.assertFalse(self.state["disabled"])

    def test_dump_selected_step(self):
        self.api.set_dump_disable()
        observed = []
        for step in range(3):
            with self.api.dump_enabled(step == 1):
                observed.append(self.state["dump"])
            self.assertFalse(self.state["dump"])
        self.assertEqual(observed, [False, True, False])

    def test_dump_nested_exception_restores_state(self):
        with self.api.dump_enabled(False):
            with self.assertRaisesRegex(RuntimeError, "test"):
                with self.api.dump_enabled():
                    self.assertTrue(self.state["dump"])
                    raise RuntimeError("test")
            self.assertFalse(self.state["dump"])
        self.assertTrue(self.state["dump"])

    def test_dump_setters_and_context_restore(self):
        self.api.set_dump_disable()
        with self.assertRaisesRegex(RuntimeError, "test"):
            with self.api.dump_enabled():
                self.api.set_dump_disable()
                self.api.set_dump_enable()
                raise RuntimeError("test")
        self.assertFalse(self.state["dump"])

    def test_dump_disabled_restores_state(self):
        self.assertIn("dump_disabled", self.api.__all__)
        for initial in (False, True):
            for raises in (False, True):
                with self.subTest(initial=initial, raises=raises):
                    self.state["dump"] = initial
                    context = self.api.dump_disabled()
                    self.assertEqual(self.state["dump"], initial)
                    try:
                        with context:
                            self.assertFalse(self.state["dump"])
                            with self.api.dump_enabled():
                                self.assertTrue(self.state["dump"])
                                with self.api.dump_disabled():
                                    self.assertFalse(self.state["dump"])
                                self.assertTrue(self.state["dump"])
                            self.assertFalse(self.state["dump"])
                            if raises:
                                raise RuntimeError("test")
                    except RuntimeError as error:
                        self.assertTrue(raises)
                        self.assertEqual(str(error), "test")
                    self.assertEqual(self.state["dump"], initial)
                    self.assertFalse(self.state["disabled"])

    def test_dump_disabled_rejects_arguments(self):
        with self.assertRaises(TypeError):
            self.api.dump_disabled(False)
        self.assertEqual(self.calls, [])

    def test_setters_reject_arguments(self):
        for name in ("set_enable", "set_disable", "set_dump_enable", "set_dump_disable"):
            setter = getattr(self.api, name)
            for value in (True, False, None, 0, 1, "true"):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(TypeError):
                        setter(value)
                    with self.assertRaises(TypeError):
                        setter(enabled=value)
        self.assertEqual(self.calls, [])

    def test_invalid_arguments_do_not_change_state(self):
        for value in (None, 0, 1, "true"):
            for context in (self.api.enabled, self.api.dump_enabled):
                with self.subTest(value=value, context=context):
                    with self.assertRaises(TypeError):
                        with context(value):
                            pass
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
