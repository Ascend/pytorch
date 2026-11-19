# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Backend-neutral compile scope for NPU ``torch.compile`` calls.

Every NPU backend enters this module's scope around one compilation, so it must
stay independent of ``torch_npu._inductor`` at import time: loading a cache does
not choose a target or a backend.

What lives here is only the routing:

* the context variables that tell a nested compile which backend the caller
  asked for (:data:`SELECTED_BACKEND`) and which request is active
  (:data:`ACTIVE`);
* :func:`backend_request_scope`, which keeps one backend's compile from
  borrowing an enclosing one's identity;
* :func:`is_dvm_request`, the predicate the package adapters route on;
* :func:`npu_devices`, target discovery performed before FX/AOT hashing;
* :func:`preserve_device`, so an imported wrapper cannot leak a device change.

The DVM producer identity and the request itself live in
``torch_npu._inductor.dvm.persistence``: they read DVM/ANIR configuration and
therefore load the backend, which must happen only after DVM is confirmed.
"""
import contextlib
import contextvars

# Only an explicit compile scope authorizes cache reuse.
ACTIVE = contextvars.ContextVar("torch_npu_compile_package_request", default=None)
# The backend the current compile asked for, set before a request exists.
SELECTED_BACKEND = contextvars.ContextVar("torch_npu_compile_package_backend", default=None)


def is_dvm_request(context=None):
    """Resolve the backend from the compile caller's request."""
    selected = SELECTED_BACKEND.get()
    if selected is not None:
        return selected == "dvm"
    context = context if context is not None else ACTIVE.get()
    if context is not None:
        backend = getattr(context, "backend", None)
        if backend is None:
            backend = context.manifest.request.get("backend")
        return backend == "dvm"
    from torch_npu.utils._dynamo import _resolve_npu_backend

    return _resolve_npu_backend(None) == "dvm"


@contextlib.contextmanager
def backend_request_scope(backend, *, isolate=False):
    """Keep one backend's compilation from borrowing an enclosing one's identity."""
    selected_token = SELECTED_BACKEND.set(backend)
    active_token = None
    if isolate or backend != "dvm":
        active_token = ACTIVE.set(None)
    try:
        yield
    finally:
        if active_token is not None:
            ACTIVE.reset(active_token)
        SELECTED_BACKEND.reset(selected_token)


def npu_devices(args, kwargs=None, module=None):
    """Discover targets before hashing, including tensors created inside FX graphs."""
    import torch
    from torch.utils._pytree import tree_leaves

    def tensor_devices(values):
        return {value.device for value in tree_leaves(values)
                if isinstance(value, torch.Tensor) and value.device.type == "npu"}

    devices = tensor_devices((args, kwargs or {}))
    if isinstance(module, torch.nn.Module):
        devices.update(tensor_devices(tuple(module.parameters())))
        devices.update(tensor_devices(tuple(module.buffers())))
        for child in module.modules():
            if not isinstance(child, torch.fx.GraphModule):
                continue
            for node in child.graph.nodes:
                # Scan intermediate nodes too: the graph may return CPU tensors
                # after creating/using NPU tensors. FakeTensor.device is concrete
                # metadata and does not require executing the graph.
                targets = tensor_devices((node.meta.get("example_value"), node.meta.get("val")))
                devices.update(targets)
                if targets or node.op not in ("call_function", "call_method"):
                    continue
                # Missing tensor metadata must not turn a known NPU operation
                # into a CPU request. Keep an unspecified ordinal unresolved;
                # make_context will bypass persistence instead of guessing it.
                specs = [node.kwargs.get("device")]
                specs.extend(value for value in tree_leaves((node.args, node.kwargs))
                             if isinstance(value, torch.device))
                if node.op == "call_method" and node.target == "to":
                    specs.extend(node.args[1:])
                elif node.op == "call_method" and node.target == "npu":
                    target = node.args[1] if len(node.args) > 1 else node.kwargs.get("device")
                    if type(target) is int:
                        target = torch.device("npu", target)
                    specs.append(target if isinstance(target, (str, torch.device)) else "npu")
                for spec in specs:
                    if isinstance(spec, str) and (spec == "npu" or spec.startswith("npu:")):
                        spec = torch.device(spec)
                    if isinstance(spec, torch.device) and spec.type == "npu":
                        devices.add(spec)
    return tuple(sorted(devices, key=str))


@contextlib.contextmanager
def preserve_device():
    """Generated wrappers may set a device at import time; don't leak it."""
    import torch

    previous = torch.npu.current_device() if torch.npu.is_initialized() else None
    try:
        yield
    finally:
        if previous is not None:
            torch.npu.set_device(previous)
