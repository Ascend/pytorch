# Copyright (c) 2026, Huawei Technologies Co., Ltd

"""NPU compile options must work before an Inductor backend is imported.

Use fresh interpreters: an earlier config copy or backend import otherwise
hides the registration-order regression. These tests do not need an NPU.
"""

import os
import subprocess
import sys
import textwrap

from torch_npu.testing.testcase import TestCase, run_tests


class TestCompileOptionsColdStart(TestCase):
    def _run_fresh(self, code):
        env = dict(os.environ)
        env.pop("TORCHINDUCTOR_NPU_BACKEND", None)
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    def test_first_compile_keeps_backend_and_device_lazy(self):
        self._run_fresh("""
            import sys
            import torch
            import torch_npu

            assert "torch._inductor" not in sys.modules
            assert "torch_npu._inductor" not in sys.modules
            assert not torch.npu.is_initialized()
            compiled = torch.compile(
                lambda x: x + 1,
                options={"npu_backend": "triton_experimental"},
            )
            assert callable(compiled)
            assert "torch_npu._inductor" not in sys.modules
            assert not torch.npu.is_initialized()
        """)

    def test_all_npu_options_are_registered_before_validation(self):
        self._run_fresh("""
            import sys
            import torch
            import torch_npu

            options = {
                "npu_backend": "triton_experimental",
                "enable_shape_handling": False,
                "shape_handling_configs": [{"type": "BATCHSIZE"}],
                "shape_handling_dict": {},
            }
            wrapper = torch._TorchCompileInductorWrapper(
                mode=None, options=options, dynamic=False,
            )
            for key, value in options.items():
                assert wrapper.config[key] == value, (key, wrapper.config)
            assert "torch_npu._inductor" not in sys.modules
            assert not torch.npu.is_initialized()
        """)

    def test_repeated_registration_preserves_config_overrides(self):
        self._run_fresh("""
            import sys
            import torch
            import torch_npu
            from torch._inductor import config

            # Register through the historical config-copy path first.
            config.get_config_copy()
            options = {
                "npu_backend": "dvm",
                "enable_shape_handling": False,
                "shape_handling_configs": [{"type": "BATCHSIZE"}],
                "shape_handling_dict": {},
            }
            with config.patch(options):
                entries = {key: config._config[key] for key in options}
                for _ in range(2):
                    torch.compile(
                        lambda x: x + 1,
                        options={"npu_backend": "triton_experimental"},
                    )
                    for key, value in options.items():
                        assert getattr(config, key) == value, key
                        assert config._config[key] is entries[key], key
            assert config.npu_backend == "default"
            assert "torch_npu._inductor" not in sys.modules
            assert not torch.npu.is_initialized()
        """)

    def test_first_compile_rejects_invalid_npu_option_types(self):
        invalid_options = (
            ("npu_backend", 1),
            ("enable_shape_handling", "false"),
            ("shape_handling_configs", "invalid"),
            ("shape_handling_dict", []),
        )
        for key, value in invalid_options:
            with self.subTest(option=key):
                self._run_fresh(f"""
                    import sys
                    import torch
                    import torch_npu

                    try:
                        torch.compile(lambda x: x, options={{{key!r}: {value!r}}})
                    except RuntimeError as exc:
                        assert "Unexpected type of attr {key}" in str(exc), str(exc)
                    else:
                        raise AssertionError("invalid option type was accepted")
                    assert "torch_npu._inductor" not in sys.modules
                    assert not torch.npu.is_initialized()
                """)

    def test_first_compile_rejects_unknown_options(self):
        self._run_fresh("""
            import torch
            import torch_npu

            try:
                torch.compile(lambda x: x, options={"unknown_npu_compile_option": True})
            except RuntimeError as exc:
                assert "Unexpected optimization option unknown_npu_compile_option" in str(exc), str(exc)
            else:
                raise AssertionError("unknown compile option was accepted")
        """)

    def test_shape_handling_is_enabled_once_after_validation(self):
        self._run_fresh("""
            import torch
            import torch_npu
            from unittest.mock import patch

            options = {"npu_backend": "triton_experimental", "enable_shape_handling": True}
            with patch("torch_npu.utils._dynamo._patch_shape_handling") as enable:
                wrapper = torch._TorchCompileInductorWrapper(
                    mode=None, options=options, dynamic=False,
                )
                enable.assert_called_once_with()
            assert wrapper.config["enable_shape_handling"] is True
        """)

    def test_invalid_options_do_not_enable_shape_handling(self):
        self._run_fresh("""
            import torch
            import torch_npu
            from unittest.mock import patch

            with patch("torch_npu.utils._dynamo._patch_shape_handling") as enable:
                try:
                    torch._TorchCompileInductorWrapper(
                        mode=None,
                        options={"npu_backend": 1, "enable_shape_handling": True},
                        dynamic=False,
                    )
                except RuntimeError as exc:
                    assert "Unexpected type of attr npu_backend" in str(exc), str(exc)
                else:
                    raise AssertionError("invalid option type was accepted")
                enable.assert_not_called()
        """)


if __name__ == "__main__":
    run_tests()
