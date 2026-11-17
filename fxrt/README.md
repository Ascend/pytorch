# FXRT

FXRT is a lightweight, high-performance runtime for the inference phase. It lowers PyTorch graphs (via `torch.compile`) onto its own IR and runtime, executing them on the Ascend backend. It focuses on inference latency and memory reuse, with native support for zero-copy view operators.

> 中文文档：[README_CN.md](./README_CN.md)

## Table of Contents

- [Build and Install](#build-and-install)
- [Run Your First Example](#run-your-first-example)
- [Enable FXRT in Inductor](#enable-fxrt-in-inductor)
- [Environment Variables](#environment-variables)

## Build and Install

FXRT is built and shipped as part of torch_npu; it is not packaged on its own.
Build torch_npu from source as usual and the wheel carries FXRT under
`torch_npu/fxrt`:

```bash
bash ci/build.sh --python=3.10
```

Pass `--disable_fxrt` to build torch_npu without FXRT. Its only extra build
dependency, nanobind, is declared in torch_npu's `requirements*-build.txt`, and
it is needed at build time only. See the
[build-from-source guide](../docs/en/installation_guide/building_from_source.md).

FXRT is imported as `torch_npu.fxrt`. `import torch_npu` alone does not load it;
once `torch_npu.fxrt` is imported, the same package is also reachable as `fxrt`.

## Run Your First Example

FXRT plugs in as a `torch.compile` backend. Once a model or function is compiled with `backend="fxrt"`, it runs on the FXRT runtime:

```python
# quickstart.py
import torch
import torch_npu


def model(x, y):
    z = x + y
    z = z.view(4, -1)        # view operators take the zero-copy path
    return torch.relu(z)


# Move the inputs to npu
x = torch.randn(2, 8).npu()
y = torch.randn(2, 8).npu()

compiled = torch.compile(model, backend="fxrt")
out = compiled(x, y)
print(out.shape)
```

Key points:

- The installed wheel registers `"fxrt"` as a compile backend and loads FXRT on demand.
- Call the function as usual after `torch.compile(fn, backend="fxrt")`.
- `view` / `reshape` / `permute` / `slice` and similar operators take the zero-copy view path by default.

## Enable FXRT in Inductor

The Inductor `fxrt` option defaults to `False`. When enabled, Inductor uses FX wrappers for all graphs, including the FXRT wrapper for NPU graphs.

PyTorch 2.15 introduced device-namespaced compile options. Follow the pre-2.15 examples for early 2.15 nightly builds without device namespace support.

### Enable FXRT for one compiled model

```python
# PyTorch 2.15+ with device namespace support
compiled = torch.compile(model, backend="inductor", options={"npu.fxrt": True})
out = compiled(x, y)
```

```python
# PyTorch before 2.15 (also accepted by TorchNPU on newer versions)
compiled = torch.compile(model, backend="inductor", options={"fxrt": True})
out = compiled(x, y)
```

### Enable FXRT within a scope

Call `torch.compile` within the `config.patch()` scope. The config is restored on exit. The compiled model can run outside the scope.

```python
# PyTorch 2.15+ with device namespace support
from torch_npu._inductor import compile_options as config

with config.patch({"fxrt": True}):
    compiled = torch.compile(model, backend="inductor")

out = compiled(x, y)
```

```python
# PyTorch before 2.15
import torch_npu._inductor  # Register the FXRT option.
from torch._inductor import config

with config.patch({"fxrt": True}):
    compiled = torch.compile(model, backend="inductor")

out = compiled(x, y)
```

### Set the default for subsequent compilations

Set `config.fxrt` directly. Inductor reads that value when creating subsequent compiled models in the current context, until you change it again.

```python
# PyTorch 2.15+ with device namespace support
from torch_npu._inductor import compile_options as config

config.fxrt = True
compiled = torch.compile(model, backend="inductor")
out = compiled(x, y)
```

```python
# PyTorch before 2.15
import torch_npu._inductor  # Register the FXRT option.
from torch._inductor import config

config.fxrt = True
compiled = torch.compile(model, backend="inductor")
out = compiled(x, y)
```

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `FXRT_DISABLE_VIEW_OPS` | Disable the zero-copy implementation of specific view operators and fall back to the non-view path, for debugging and comparison. Accepts a comma-separated list drawn from `permute`, `transpose`, `t`, `movedim`, `split` and `chunk`, or `all` to cover them in one go. |
| `FXRT_DEV_DUMP_IR` | When set to `1`, dumps the compiled IR for debugging |
