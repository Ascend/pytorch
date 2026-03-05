import os
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
        input_lines.append(
            f"arg{i} = torch.empty_strided("
            f"torch.Size({tuple(v.shape)}), "
            f"{tuple(v.stride())}, "
            f"dtype={v.dtype}).uniform_(0, 1).npu()"
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
        ]
    else:
        fusion_env = 'os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"'
        fusion_imports = "from torch_npu._inductor.dvm import mlir_fusion"
        compile_lines = [
            "compiled = torch.compile(model, backend=\"inductor\", dynamic=False)",
            f"out = compiled({fwd_args})",
        ]
    compile_code = "\n    ".join(compile_lines)
    env_lines = fusion_env

    test_code = f"""import torch
import torch_npu
from torch.utils._pytree import tree_flatten
import os
{env_lines}


class {class_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
{_indent(gm.code, 4)}


def _assert_close(ref, out, atol=1e-4, rtol=1e-4):
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


if __name__ == "__main__":
    test_case()
    print("PASS")
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(test_code)

    print(f"[ok] generated: {file_path}")
    return file_path
