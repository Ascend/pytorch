"""Runner-owned Dynamo backend identity, separate from ordinary code caches."""

import torch


class IsolatedGraphBackend:
    """A per-runner backend object with identity-based Dynamo guards.

    Object identity is never equal to an ordinary backend or to another
    runner's backend, so Dynamo guards root frames and graph-break
    continuations separately.  Both lanes of one runner share this backend,
    which lets them share compiled code while the graphify adapter keeps
    their graph-tree state apart.
    """

    compiler_name = "npu_dual_stream_graph"

    def __init__(self, backend="inductor", *, options=None, dynamic=True):
        if backend != "inductor" and not callable(backend):
            raise TypeError(
                "backend must be 'inductor' or an explicitly adapted backend callable"
            )
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary")
        self._backend = backend
        self._options = dict(options or {})
        # Splitting belongs to the caller of run(), not to each lane's
        # compilation; strip shape-handling knobs so lanes never re-split.
        for key in (
            "enable_shape_handling", "shape_handling_configs", "shape_handling_dict",
        ):
            self._options.pop(key, None)
        if backend == "inductor":
            for key in ("triton.cudagraphs", "triton.cudagraph_trees"):
                if self._options.get(key, True) is not True:
                    raise ValueError(f"Dual-stream Inductor requires {key}=True")
                self._options[key] = True
        elif self._options:
            raise ValueError("options are only supported with the Inductor backend")
        self._dynamic = dynamic
        self._compiler = None
        self.compile_count = 0

    def __call__(self, gm, example_inputs):
        if self._compiler is None:
            if self._backend == "inductor":
                self._compiler = torch._TorchCompileInductorWrapper(
                    mode=None,
                    options={"triton.cudagraphs": True, "triton.cudagraph_trees": True},
                    dynamic=self._dynamic,
                )
            else:
                self._compiler = self._backend
        self.compile_count += 1
        if self._backend == "inductor":
            from torch._inductor import config

            # The caller may pass a full validated compiler-config snapshot;
            # apply it as a compilation scope instead of re-validating here.
            with config.patch(self._options):
                return self._compiler(gm, example_inputs)
        return self._compiler(gm, example_inputs)

    def reset(self):
        if self._compiler is not None:
            compiler_reset = getattr(self._compiler, "reset", None)
            if compiler_reset is not None:
                compiler_reset()


def compile_region(fn, *, backend="inductor", options=None, dynamic=True):
    """Compile a raw callable with an isolated backend identity.

    Nested precompiled callables are outside this contract: the whole region
    must be compiled by the returned backend, including graph breaks.
    Ordinary Dynamo caches are neither cleared nor reused.
    """
    if not callable(fn):
        raise TypeError("compile_region requires a callable raw function")
    if isinstance(fn, torch.nn.Module):
        raise ValueError(
            "Pass a raw function (e.g. a def calling model(x)), not an nn.Module"
        )
    if hasattr(fn, "_torchdynamo_orig_callable") or hasattr(fn, "_orig_mod"):
        raise ValueError(
            "Pass the raw function, not a torch.compile wrapper or nn.Module"
        )
    isolated = IsolatedGraphBackend(backend, options=options, dynamic=dynamic)
    return torch.compile(fn, backend=isolated, dynamic=dynamic), isolated
