from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    overload,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from collections.abc import Sequence
import torch
from torch._inductor.ir import (NopKernel, SliceView, IRNode, StorageBox, FlexibleLayout, FixedLayout, NonOwningLayout, Pointwise, TensorBox, ComputedBuffer, View, log, Layout, Scatter)
from torch._inductor.virtualized import ops, OpsValue, V
from torch._inductor.utils import ir_dataclass
from torch._inductor import lowering
from sympy import Expr, Integer, Symbol
from typing_extensions import Never

from torch.utils._sympy.functions import Identity
from torch_npu._inductor import config


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
        else:
            # For non-aten OpOverload, i.e. custom ops
            if V.graph.cpp_wrapper:
                self.use_runtime_dispatch = True

        if self.use_runtime_dispatch:
            self.codegen_comment(wrapper)

            exported_args = None
            args = None
            exported_args = self.export_extern_kernel_node()

            wrapper.generate_fallback_kernel_with_runtime_lookup(
                self.get_name(),
                self.python_kernel_name,
                self.cpp_kernel_name,
                args,
                self.op_overload,
                exported_args,
                # NOTE: [special handling of all_reduce_coalesced_'s return value]
                self.outputs if self.outputs else self.mutation_outputs,
            )
        else:
            self.codegen_comment(wrapper)
            args = [*self.codegen_args(), *self.codegen_kwargs()]
            V.graph.wrapper_code.generate_fallback_kernel(self, args)
            if isinstance(self.layout, Layout):
                self.codegen_size_asserts(wrapper)

        self.codegen_unbacked_symbol_defs(wrapper)
    
    from torch._inductor.ir import FallbackKernel
    FallbackKernel.codegen = codegen_npu


@ir_dataclass
class IndexputTemplate(Scatter):
    boundary: Optional[int] = None

    def store_output(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        store_vars: Sequence[Expr],
    ) -> None:
        loader = self.make_loader()
        if output_name is None:
            output_name = "unnamed"
        output_indexer = self.output_indexer(store_vars)
        indirect_indexer = None
        for var in output_indexer:
            if str(var).startswith("indirect"):
                indirect_indexer = var
                break

        return ops.indexput_template(
            output_name,
            indexer(output_indexer),
            loader(store_vars),
            indirect_indexer,
            self.boundary
        )


class ScatterTemplate(Scatter):
    def store_output(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        store_vars: Sequence[Expr],
    ) -> None:
        loader = self.make_loader()
        if output_name is None:
            output_name = "unnamed"
        output_indexer, boundary = self.output_indexer(store_vars)
        indirect_indexer = None
        for var in output_indexer:
            if str(var).startswith("indirect"):
                indirect_indexer = var
                break

        return ops.scatter_template(
            output_name,
            indexer(output_indexer),
            loader(store_vars),
            indirect_indexer,
            int(boundary),
        )