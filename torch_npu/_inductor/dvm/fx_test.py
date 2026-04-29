import os
import sys
import importlib
import hashlib
import torch


DEFAULT_OUTPUT_DIR = "./dvm_fx_regression_cases"


def generate_dvm_fx_case(
    gm: torch.fx.GraphModule,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    fusion_type: str = "graph",
):
    if fusion_type not in ("graph", "mlir"):
        raise ValueError(f"unsupported fusion_type: {fusion_type}")

    def _indent(code: str, spaces: int) -> str:
        pad = " " * spaces
        return "\n".join(
            pad + line if line.strip() else line
            for line in code.split("\n")
        )

    os.makedirs(output_dir, exist_ok=True)

    readable = gm.print_readable(print_output=False)

    sig = []
    inputs = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for i, n in enumerate(inputs):
        v = n.meta["val"]
        sig.append(f"arg{i}:{tuple(v.shape)},{tuple(v.stride())},{v.dtype}")

    h = hashlib.sha256(
        (fusion_type + "\n" + readable + "\n" + "\n".join(sig)).encode("utf-8")
    ).hexdigest()[:16]
    case_name = f"test_{fusion_type}_{h}"

    file_path = os.path.join(output_dir, f"{case_name}.py")
    if os.path.exists(file_path):
        print(f"[skip] {file_path}")
        return None
    class_name = "TestModel"

    input_lines = []
    for i, n in enumerate(inputs):
        v = n.meta["val"]
        fill = "random_()" if v.dtype == torch.bool else "uniform_(0, 1)"
        input_lines.append(
            f"arg{i} = torch.empty_strided("
            f"torch.Size({tuple(v.shape)}), "
            f"{tuple(v.stride())}, "
            f"dtype={v.dtype}, device='npu').{fill}"
        )

    input_code = "\n    ".join(input_lines)
    fwd_args = ", ".join(f"arg{i}" for i in range(len(inputs)))

    if fusion_type == "graph":
        fusion_env = ""
        fusion_imports = (
            "from torch_npu._inductor.dvm.graph_fusion "
            "import DvmGraphFusionPatch"
        )
        compile_lines = [
            "with DvmGraphFusionPatch():",
            "    compiled = torch.compile(model, backend=\"inductor\", dynamic=False)",
            f"    out = compiled({fwd_args})",
            "    deterministic_state = torch.are_deterministic_algorithms_enabled()",
            "    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()",
            "    try:",
            "        torch.use_deterministic_algorithms(True)",
            "        deterministic_compiled = torch.compile(model, backend=\"inductor\", dynamic=False)",
            f"        deterministic_out = deterministic_compiled({fwd_args})",
            "    finally:",
            "        torch.use_deterministic_algorithms(deterministic_state, warn_only=deterministic_warn_only)",
        ]
    else:
        fusion_env = 'os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"'
        fusion_imports = "from torch_npu._inductor.dvm import mlir_fusion"
        compile_lines = [
            "compiled = torch.compile(model, backend=\"inductor\", dynamic=False)",
            f"out = compiled({fwd_args})",
            "deterministic_state = torch.are_deterministic_algorithms_enabled()",
            "deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()",
            "try:",
            "    torch.use_deterministic_algorithms(True)",
            "    deterministic_compiled = torch.compile(model, backend=\"inductor\", dynamic=False)",
            f"    deterministic_out = deterministic_compiled({fwd_args})",
            "finally:",
            "    torch.use_deterministic_algorithms(deterministic_state, warn_only=deterministic_warn_only)",
        ]
    compile_code = "\n    ".join(compile_lines)
    env_lines = fusion_env

    test_code = f"""import torch
import torch_npu
from torch import device
from torch.utils._pytree import tree_flatten
import os
{env_lines}


class {class_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
{_indent(gm.code, 4)}


def _assert_close(ref, out, atol=2e-3, rtol=2e-3):
    rf, rs = tree_flatten(ref)
    of, os = tree_flatten(out)
    assert rs == os, f"pytree mismatch\\nref={{rs}}\\nout={{os}}"
    for r, o in zip(rf, of):
        torch.testing.assert_close(r, o, atol=atol, rtol=rtol, equal_nan=True)


def test_case():
    {input_code}

    model = {class_name}().npu()
    ref = model({fwd_args})

    {fusion_imports}

    {compile_code}

    _assert_close(ref, out)
    _assert_close(ref, deterministic_out)


if __name__ == "__main__":
    test_case()
    print("PASS")
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(test_code)

    print(f"[ok] generated: {file_path}")
    return file_path


def _load_fx_model(acc_meta):
    """Load the traced FX GraphModule from disk for accuracy comparison."""
    if acc_meta.get('_fx_model') is not None:
        return acc_meta['_fx_model']
    dump_path = os.path.join(
        os.getenv("TORCHINDUCTOR_CACHE_DIR"),
        acc_meta['traced_graph_cache'],
        str(acc_meta['device_index']),
        acc_meta['traced_graph_hash'],
    )
    sys.path.insert(0, dump_path)
    try:
        module = importlib.import_module(acc_meta['traced_graph_hash'])
    finally:
        sys.path.remove(dump_path)
    Model = getattr(module, acc_meta['traced_graph_hash'])
    model = Model()
    acc_meta['_fx_model'] = model
    return model
 
 
def _accuracy_check_run(kobj, acc_meta, kernel_name, args):
    """Run DVM kernel then compare outputs against FX graph reference."""
    from torch._inductor.compile_fx import clone_preserve_strides
    from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import (
        config as anir_config,
    )

    fx_model = _load_fx_model(acc_meta)

    num_outputs = acc_meta['num_outputs']
    num_inputs = len(args) - num_outputs

    fx_inputs = []
    for arg in args[:num_inputs]:
        if isinstance(arg, torch.Tensor):
            inp = clone_preserve_strides(arg)
            if arg.dtype == torch.bfloat16:
                inp = inp.float()
            fx_inputs.append(inp)
        else:
            fx_inputs.append(arg)

    fx_outputs = fx_model.forward(*fx_inputs)
    if not isinstance(fx_outputs, (tuple, list)):
        fx_outputs = (fx_outputs,)

    kobj.run(*args)

    for idx, (actual, expected) in enumerate(
        zip(args[num_inputs:], fx_outputs)
    ):
        if not isinstance(actual, torch.Tensor):
            continue
        if actual.dtype != expected.dtype:
            expected = expected.to(actual.dtype)
        tol = anir_config.acc_comp_tol.get(
            actual.dtype, anir_config.acc_comp_tol["default"]
        )
        rtol, atol = tol["rtol"], tol["atol"]
        matches = torch.isclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )
        if not matches.all():
            abs_diff = torch.abs(actual - expected)
            rel_diff = abs_diff / torch.clamp(torch.abs(expected), min=1e-20)
            rel_diff.masked_fill_(matches, 0)
            num_el = matches.numel()
            num_mis = num_el - int(torch.sum(matches))
            print(
                f"CHECK ACCURACY FAILED! "
                f"Kernel: {kernel_name}, "
                f"Output idx: {idx}, "
                f"Mismatched: {num_mis}/{num_el} ({num_mis / num_el:.1%}), "
                f"Greatest Rel Diff: {rel_diff.max().item()}, "
                f"Greatest Abs Diff: {abs_diff.max().item()}",
                flush=True,
            )

            del abs_diff, rel_diff
        del matches
