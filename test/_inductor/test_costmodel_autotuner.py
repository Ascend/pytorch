from types import SimpleNamespace

from torch.testing._internal.common_utils import run_tests, TestCase

from torch_npu._inductor.runtime import triton_heuristics


class TestCostModelAutotuner(TestCase):
    def test_costmodel_reloads_jit_function_after_parallel_compile(self):
        autotuner = object.__new__(triton_heuristics.NPUCostModelAutotuner)
        autotuner.fn = SimpleNamespace(fn=None)
        reloaded_jit_fn = SimpleNamespace(fn=lambda: None)
        autotuner._reload_kernel = lambda: SimpleNamespace(fn=reloaded_jit_fn)
        autotuner._resolve_costmodel_runtime_inputs = lambda: ((381,), {})
        calls = []
        autotuner._apply_costmodel_to_configs = (
            lambda *args, **kwargs: calls.append((args, kwargs))
        )

        autotuner._prepare_configs_for_precompile()

        self.assertIs(autotuner.fn, reloaded_jit_fn)
        self.assertEqual(calls, [((381,), {})])

    def test_costmodel_builds_bindings_for_named_ttir_args(self):
        autotuner = object.__new__(triton_heuristics.NPUCostModelAutotuner)
        autotuner.triton_meta = {
            "signature": {
                "in_out_ptr0": "*fp16",
                "in_ptr0": "*i1",
                "y0_numel": "i32",
                "x1_numel": "i32",
                "Y0BLOCK": "i32",
            }
        }
        ttir = """
        tt.func public @kernel(
            %in_out_ptr0: !tt.ptr<f16>, %in_ptr0: !tt.ptr<i1>,
            %y0_numel: i32, %x1_numel: i32, %Y0BLOCK: i32
        ) attributes {noinline = false} { return }
        """

        bindings = autotuner._build_costmodel_arg_bindings(
            ttir,
            (object(), object(), 381, 12),
            {"Y0BLOCK": 47},
        )

        self.assertEqual(bindings, "arg2=381,arg3=12,arg4=47,pid_x=0")

    def test_costmodel_uses_config_kwargs_for_named_ttir_args(self):
        autotuner = object.__new__(triton_heuristics.NPUCostModelAutotuner)
        autotuner.triton_meta = {
            "signature": {
                "ptr": "*fp32",
                "x0_numel": "i32",
                "X0BLOCK": "i32",
            }
        }
        ttir = """
        tt.func public @kernel(
            %ptr: !tt.ptr<f32>, %x0_numel: i32, %X0BLOCK: i32
        ) attributes {noinline = false} { return }
        """
        cfg = SimpleNamespace(kwargs={"X0BLOCK": 128})

        bindings = autotuner._build_costmodel_arg_bindings(
            ttir,
            (object(), 1024),
            cfg=cfg,
        )

        self.assertEqual(bindings, "arg1=1024,arg2=128,pid_x=0")

    def test_costmodel_keeps_legacy_positional_ttir_bindings(self):
        autotuner = object.__new__(triton_heuristics.NPUCostModelAutotuner)
        autotuner.triton_meta = {
            "signature": {
                "ptr": "*fp32",
                "x0_numel": "i32",
                "X0BLOCK": "i32",
            }
        }
        ttir = """
        tt.func public @kernel(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32)
        attributes {noinline = false} { return }
        """

        bindings = autotuner._build_costmodel_arg_bindings(
            ttir,
            (object(), 1024),
            {"X0BLOCK": 128},
        )

        self.assertEqual(bindings, "arg1=1024,arg2=128,pid_x=0")


if __name__ == "__main__":
    run_tests()
