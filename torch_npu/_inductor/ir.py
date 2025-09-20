import itertools
import torch
from torch._inductor.virtualized import ops, OpsValue, V
from torch._inductor.ir import log, Layout
from torch._inductor import config


def patch_fallback_kernel_codegen():
    def codegen_npu(self, wrapper) -> None:  # type: ignore[no-untyped-def]
        kernel = self.op_overload
        if kernel.namespace == "aten":  # type: ignore[union-attr]
            if not isinstance(kernel, torch._ops.OpOverload):
                raise AssertionError(f"kernel should be OpOverload, but got {type(kernel)}")
            if V.graph.cpp_wrapper:
                # Fallback all npu op to proxy executor and warn when gpu do not.
                from torchgen.aoti.fallback_ops import inductor_fallback_ops
                self.use_runtime_dispatch = True
                if str(kernel) in inductor_fallback_ops:
                    log.warning(
                        "%s is using proxy executor as fallback instead of aoti shim.",
                        kernel,
                    )

        elif kernel.namespace == "_quantized":  # type: ignore[union-attr]
            # Internal Quantized Fallback Ops
            if not isinstance(kernel, torch._ops.OpOverload):
                raise AssertionError
        elif V.graph.cpp_wrapper:
            # For non-aten OpOverload, i.e. custom ops
            # If the op is in custom_ops_to_c_shims, generate direct function call
            self.use_runtime_dispatch = (
                kernel not in config.aot_inductor.custom_ops_to_c_shims
            )

        # Handle the special case where a complex number is input to a C-shim kernel for
        # a scalar input.  The torchgen'ed shim API will use type "double", which is
        # incompatible with complex numbers, forcing a fallback to runtime dispatch.
        if (
            V.graph.cpp_wrapper
            and isinstance(kernel, torch._ops.OpOverload)
            and not self.use_runtime_dispatch
        ):

            def is_number(t: torch.JitType) -> bool:
                if isinstance(t, torch.OptionalType):
                    return is_number(t.getElementType())
                return isinstance(t, torch.NumberType)

            # Using unflatten_args is a bit of a hack, but all the complex arguments we
            # care about are in self.constant_args, and calling unflatten_args puts them
            # in the correct order without triggering codegen.
            args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
            # Append kwarg values to args.  ordered_kwargs_for_cpp_kernel is guaranteed
            # to be set, since this is an OpOverload kernel.
            args_iter = itertools.chain(
                args,
                (
                    self.get_kwargs_value(k, **kwargs)
                    for k in self.ordered_kwargs_for_cpp_kernel
                ),
            )
            self.use_runtime_dispatch = any(
                isinstance(v, complex) and is_number(a.real_type)
                for v, a in zip(args_iter, kernel._schema.arguments)
            )

        self.codegen_comment(wrapper)
        if self.use_runtime_dispatch:
            exported_args = self.export_extern_kernel_node()
            wrapper.generate_fallback_kernel_with_runtime_lookup(
                self.get_name(),
                self.python_kernel_name,
                lambda: [*self.codegen_args(), *self.codegen_kwargs()],
                self.op_overload,
                exported_args,
                # NOTE: [special handling of all_reduce_coalesced_'s return value]
                self.outputs if self.outputs else self.mutation_outputs,
            )
        else:
            wrapper.generate_fallback_kernel(self)
            if isinstance(self.layout, Layout):
                self.codegen_size_asserts(wrapper)
                self.codegen_alignment_asserts(wrapper)

        self.codegen_unbacked_symbol_defs(wrapper)

    from torch._inductor.ir import FallbackKernel
    FallbackKernel.codegen = codegen_npu