# FXRT

FXRT is a lightweight, high-performance runtime for the inference phase. It lowers PyTorch graphs (via `torch.compile`) onto its own IR and runtime, executing them on Ascend / CPU backends. It focuses on inference latency and memory reuse, with native support for zero-copy view operators.

> 中文文档：[README_CN.md](./README_CN.md)

## Table of Contents

- [Build and Install](#build-and-install)
- [Run Your First Example](#run-your-first-example)
- [Environment Variables](#environment-variables)

## Build and Install

FXRT is built and shipped as part of torch_npu; it is not packaged on its own.
Build torch_npu from source as usual and the wheel carries FXRT under
`torch_npu/fxrt`, with the Ascend and CPU backends both enabled:

```bash
bash ci/build.sh --python=3.10
```

Pass `--disable_fxrt` to build torch_npu without FXRT. Its only extra build
dependency, nanobind, is declared in torch_npu's `requirements*-build.txt`, and
it is needed at build time only. See the
[build-from-source guide](../docs/en/installation_guide/compilation_installation_using_source_code.md).

FXRT is imported as `torch_npu.fxrt`. `import torch_npu` alone does not load it;
once `torch_npu.fxrt` is imported, the same package is also reachable as `fxrt`.

## Run Your First Example

FXRT plugs in as a `torch.compile` backend. Once a model or function is compiled with `backend`, it runs on the FXRT runtime:

```python
# quickstart.py
import torch
from torch_npu.fxrt import backend


def model(x, y):
    z = x + y
    z = z.view(4, -1)        # view operators take the zero-copy path
    return torch.relu(z)


# For the Ascend backend, move tensors to npu; drop .npu() for the CPU backend
x = torch.randn(2, 8).npu()
y = torch.randn(2, 8).npu()

compiled = torch.compile(model, backend=backend)
out = compiled(x, y)
print(out.shape)
```

Key points:

- `from torch_npu.fxrt import backend` provides the compile backend.
- Call the function as usual after `torch.compile(fn, backend=backend)`.
- `view` / `reshape` / `permute` / `slice` and similar operators take the zero-copy view path by default.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `FXRT_DISABLE_VIEW_OPS` | Disable the zero-copy implementation of specific view operators and fall back to the non-view path, for debugging and comparison. Accepts a comma-separated list of operator names, or `all`. |
| `FXRT_DEV_DUMP_IR` | When set to `1`, dumps the compiled IR for debugging |
