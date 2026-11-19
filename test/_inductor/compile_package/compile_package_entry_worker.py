# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""One bounded process for a cross-process cache stage; writes result.json.

A producer compiles, executes and persists; a consumer compiles the same callable
in a fresh process against the same cache root. Each side reports the counters it
actually observed, so the assertions live in the test rather than here. Only
trusted compiler artifacts are loaded: the upstream package format uses pickle.

The three entries are isolated on purpose. ``AOTAutogradCache`` is tried before
the FX cache and its hit would hide what the FX layer did, so the ``fx`` entry
turns the AOT cache off and ``aot`` leaves the package path off.

  python3 compile_package_entry_worker.py --phase producer --entry package \\
          --artifact result.json
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import time
import traceback

ENTRIES = ("package", "fx", "aot")


def _workload():
    """One small DVM graph: a matmul the backend lowers itself, plus a relu."""
    import torch

    def fn(x, w):
        return torch.relu(x @ w + 1.0)

    return fn, (torch.randn(64, 64).npu(), torch.randn(64, 64).npu())


def _counters():
    from torch._dynamo.utils import counters

    return {
        "dynamo_cache": dict(counters["dynamo_cache"]),
        "aot_autograd": dict(counters["aot_autograd"]),
        "fxgraph": {k: v for k, v in counters["inductor"].items() if "fxgraph" in k},
        "frames_ok": counters["frames"]["ok"],
        "npu_package": dict(counters["npu_package"]),
    }


def _cache_dirs():
    """What landed on disk, so a test can tell which layer persisted."""
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    root = Path(cache_dir())
    return {name: len(list((root / name).iterdir())) if (root / name).is_dir() else 0
            for name in ("dynamo", "fxgraph", "aotautograd")}


@contextlib.contextmanager
def _entry_scope(entry):
    """Enable exactly the cache under test.

    ``AOTAutogradCache`` is consulted before ``FxGraphCache`` and is enabled by
    ``torch._functorch.config.enable_autograd_cache`` (default on). Leaving it on
    for the ``fx`` entry would report an AOT hit and hide what the FX layer did,
    so that entry turns it off; the ``aot`` entry keeps it and leaves the package
    path off.
    """
    import torch._dynamo.config as dynamo_config
    from torch._functorch import config as functorch_config

    with contextlib.ExitStack() as stack:
        stack.enter_context(dynamo_config.patch(caching_precompile=entry == "package"))
        if entry == "fx":
            stack.enter_context(functorch_config.patch(enable_autograd_cache=False))
        yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("producer", "consumer"), required=True)
    parser.add_argument("--entry", choices=ENTRIES, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("result.json"))
    parser.add_argument("--change-config", action="store_true",
                        help="Flip a DVM codegen knob before compiling")
    parser.add_argument("--backward", action="store_true",
                        help="Run forward and backward, and check gradients too")
    parser.add_argument("--flip-init", action="store_true",
                        help="With --backward: flip an initialization-time knob "
                             "between forward and backward")
    parser.add_argument("--fx-fallback", action="store_true",
                        help="Force the fused kernel through the FX fallback "
                             "path, archiving a real FX asset")
    args = parser.parse_args()

    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "started": time.time()}
    try:
        import torch
        import torch_npu  # noqa: F401  - registers torch.npu in this worker process

        fn, inputs = _workload()
        if args.fx_fallback:
            # The only fused kernel this workload produces is
            # ``dvm_fused_add_relu_<n>``, and on a fresh cache root it is the
            # first, so the name is stable. Forcing it through the fallback
            # path is what makes the compilation archive a real FX asset;
            # without an asset on disk the resilience entries would assert
            # on nothing.
            from torch_npu._inductor.dvm import config as dvm_config

            dvm_config.force_fallback_kernel_names = {"dvm_fused_add_relu_0"}
        if args.backward:
            # The eager baseline gets its own tensors so the compiled run
            # cannot accumulate onto its graph.
            base = tuple(t.detach().clone().requires_grad_(True) for t in inputs)
            expect = fn(*base)
            expect.sum().backward()
        else:
            expect = fn(*inputs)

        with _entry_scope(args.entry):
            if args.change_config:
                from torch_npu._inductor.dvm import config as dvm_config

                # A codegen knob that feeds the backend pass UUID: the lower keys
                # move while the package itself must stay usable.
                dvm_config.disable_addmm_fusion = not dvm_config.disable_addmm_fusion
                result["changed_config"] = {
                    "disable_addmm_fusion": dvm_config.disable_addmm_fusion}

            run_inputs = (tuple(t.detach().clone().requires_grad_(True) for t in inputs)
                          if args.backward else inputs)
            started = time.monotonic()
            out = torch.compile(fn)(*run_inputs)
            if args.flip_init:
                from torch_npu._inductor.dvm import config as dvm_config

                # An initialization-time policy: a deferred backward must
                # notice the drift and stay out of the caches instead of
                # reusing the forward's identity for another save.
                dvm_config.enable_decomp_list = [
                    *(dvm_config.enable_decomp_list or []), "_flip_init_probe_"]
                result["flipped_init"] = list(dvm_config.enable_decomp_list)
            if args.backward:
                out.sum().backward()
                result["correct"] = bool(
                    torch.equal(out, expect)
                    and all(a is not None and b is not None and torch.equal(a, b)
                            for a, b in zip((t.grad for t in run_inputs),
                                            (t.grad for t in base))))
            else:
                result["correct"] = bool(torch.equal(out, expect))
            result["compile_seconds"] = round(time.monotonic() - started, 1)

            result.update(_counters())
            if args.phase == "producer" and args.entry == "package":
                # Package persistence is explicit; FX and AOT save themselves.
                result["saved"] = torch.compiler.save_cache_artifacts() is not None
            result["cache_dirs"] = _cache_dirs()
        result["status"] = "pass" if result["correct"] else "fail"
    except Exception:
        result.update(status="fail", exception=traceback.format_exc())

    args.artifact.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "exception"}, indent=2))
    return int(result["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
