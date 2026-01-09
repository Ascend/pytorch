import copy
import logging
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Literal,
                    Optional, Tuple, Union)

import torch
from sympy import Expr, symbols
from torch import dtype as torch_dtype
from torch.utils._sympy.value_ranges import ValueRanges
from torch._inductor.codegen.common import (CSEVariable, IndentedBuffer, Kernel, KernelArgs,
                                            OpOverrides, WorkspaceArg,
                                            WorkspaceZeroMode)
from torch._inductor.codegen.cpp_utils import DTYPE_TO_CPP, CppPrinter
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.ir import (Buffer, ChoiceCaller, IRNode, Layout,
                                PrimitiveInfoType, TemplateBuffer, TensorBox)
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import ops, V

from ...autotune_process import CATLASSBenchmarkRequest

if TYPE_CHECKING:
    from .catlass_template import ArgInfo, CATLASSTemplate


log = logging.getLogger("torch._inductor")

cexpr = CppPrinter().doprint

# NB: for catlass bf16 type
_DTYPE_TO_CPP = copy.deepcopy(DTYPE_TO_CPP)
_DTYPE_TO_CPP[torch.bfloat16] = "bfloat16_t"


def _normalize_idx(index: int, total_length: int) -> int:
    return index if index >= 0 else index + total_length


ValidLayoutSymbols = Literal["M", "N", "K"]
ValidLayoutAttrs = Literal["size"]


@dataclass(frozen=True)
class LayoutArg:
    node: IRNode
    symbol: ValidLayoutSymbols
    attr: ValidLayoutAttrs
    dim: int

    def matches(self, node, attr, dim) -> bool:
        return self.node == node and self.attr == attr and self.dim == dim


class CATLASSTemplateBuffer(TemplateBuffer):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        layout,
        inputs,
        make_kernel_render,
        workspace_size: int,
        template: "CATLASSTemplate",  # type: ignore[name-defined]
        is_mix: bool = False,
        epilogue_fusion_type: int = 0,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template
        self.is_mix = is_mix
        self.epilogue_fusion_type = epilogue_fusion_type

    def get_workspace_size(self) -> int:
        return self.workspace_size if self.workspace_size is not None else 0

    def emulate_store_fn(self) -> None:
        for output in self.get_outputs():
            ops.store(output.get_name(), None, None)


class CATLASSKernelArgs(KernelArgs):
    @staticmethod
    def replace_bf16(s: str) -> str:
        return s.replace("bfloat16", "bfloat16_t")

    # HACK for catlass's bfloat16 type
    # torch.bfloat16 is "bfloat16_t" in catlass while "bfloat16" in other place
    # so we can not modify the DTYPE_TO_CPP dict
    def cpp_argdefs(self):
        arg_defs, call_args, arg_types = super().cpp_argdefs()

        if any("bfloat16" in s for s in arg_types):
            return (
                [self.replace_bf16(s) for s in arg_defs],
                call_args,
                [self.replace_bf16(s) for s in arg_types],
            )
        else:
            return arg_defs, call_args, arg_types


class CATLASSTemplateKernel(Kernel):
    """
    Template kernels defined by Catlass in C++.
    """

    overrides = OpOverrides  # type: ignore[assignment]
    _EXTRA_CPP_ARGS = "size_t* workspace_size, uint8_t* workspace, aclrtStream stream"

    def __init__(
        self,
        kernel_name: str,
    ) -> None:
        catlass_args = CATLASSKernelArgs()
        super().__init__(args=catlass_args)
        self.layout_args: Dict[str, LayoutArg] = {}
        # Mapping from arg name to IRNode.
        self.named_nodes: Dict[str, IRNode] = {}
        self.kernel_name = kernel_name

    def arg_name(self, node: IRNode) -> Optional[str]:
        """
        Returns arg name of a given input or output node.
        """
        if node is None:
            return None
        return {**self.args.input_buffers, **self.args.output_buffers}.get(
            node.get_name(), None
        )

    def check_not_null(self, node: IRNode) -> str:
        """
        Generates code to check that a node is not null.
        """
        if node is None:
            return ""

        size_str = self.size(node, 0, -1)
        name_str = self.arg_name(node)
        if name_str is None:
            return ""

        res = IndentedBuffer(initial_indent=2)
        res.tabwidth = 1
        res.splice(
            f"""
            {{
              if (!{name_str}) {{
                int64_t {name_str}_size = {size_str};
                if ({name_str}_size > 0) {{
                  throw std::runtime_error("input {name_str} is null but size is not 0!");
                }}
              }}
            }}
            """
        )
        return res.getvalue()

    def get_signature(self) -> str:
        return self.signature

    def find_symbol(
        self, node: IRNode, attr: ValidLayoutAttrs, dim: int
    ) -> Optional[str]:
        arg = self.find_layout_arg(node, attr, dim)
        return arg.symbol if arg else None

    def find_layout_arg(
        self, node: IRNode, attr: ValidLayoutAttrs, dim: int
    ) -> Optional[LayoutArg]:
        matches = [arg for arg in self.layout_args.values() if arg.matches(node, attr, dim)]
        if len(matches) >= 1:
            # Verify all matches have the same node, attribute, and dimension
            # And if they come from the same node, whichever symbol we use is fine.
            # if in runtime the logic changes, this would trigger guard
            first_match = matches[0]
            if not all(
                match.node == first_match.node
                and match.attr == first_match.attr
                and match.dim == first_match.dim
                for match in matches
            ):
                raise AssertionError("All matching layout args should be identical")
            return first_match
        return None

    def add_layout_arg(
        self, symbol: ValidLayoutSymbols, node: IRNode, attr: ValidLayoutAttrs, dim: int
    ):
        arg = LayoutArg(node, symbol, attr, dim)
        self.layout_args.setdefault(symbol, arg)

    def init_layout_args(self) -> None:
        X = self.named_nodes["X"]
        W = self.named_nodes["W"]
        Y = self.named_nodes["Y"]
        Bias = self.named_nodes.get("Bias", None)
        mdim = _normalize_idx(-2, len(X.get_size()))
        ndim = _normalize_idx(-1, len(W.get_size()))
        kdim = _normalize_idx(-1, len(X.get_size()))
        self.add_layout_arg("M", X, "size", mdim)
        self.add_layout_arg("N", W, "size", ndim)
        self.add_layout_arg("K", X, "size", kdim)

    def get_layout_args(self) -> Tuple[Union[Expr, int], ...]:
        X = self.named_nodes["X"]
        W = self.named_nodes["W"]
        Y = self.named_nodes["Y"]
        Bias = self.named_nodes.get("Bias", None)
        mdim = _normalize_idx(-2, len(X.get_size()))
        ndim = _normalize_idx(-1, len(W.get_size()))
        kdim = _normalize_idx(-1, len(X.get_size()))

        M = X.get_size()[mdim]
        N = W.get_size()[ndim]
        K = X.get_size()[kdim]
        return (M, N, K)

    @staticmethod
    def find_ld_idx(node: IRNode) -> int:
        strides = node.get_stride()
        # Handle 1D tensor case
        if V.graph.sizevars.statically_known_equals(strides[-1], 1):
            return _normalize_idx(-2, len(strides))

        assert V.graph.sizevars.statically_known_equals(strides[-2], 1), strides[-2]
        return _normalize_idx(-1, len(strides))

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        """
        Hook called from template code to generate function definition and
        needed args.

        Args:
            inputs: List of input IRNodes
            outputs: List of output IRNodes
            names_str: Comma separated list of input + output argument names.
            input_reorder: The actual order of input nodes.
                           e.g. The template might have input argument defined as [X, W, Bias],
                           and the actual input passed into this template could be [Bias, X, W].
                           In this case, the `input_reorder` would be [2, 0, 1].
        """
        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name

        for name, node in zip(names[len(inputs):len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

        arg_defs, *_ = self.args.cpp_argdefs()

        self.init_layout_args()

        size_args = [f"const int {s}" for s in ("M", "N", "K")]

        signature = f"int {self.kernel_name}({', '.join(arg_defs + size_args)}, {self._EXTRA_CPP_ARGS})"
        self.signature = signature
        return signature

    def call_kernel(
        self,
        name: str,
        node: CATLASSTemplateBuffer,
    ) -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.PythonWrapperCodegen

        name: Name of kernel function.
        node: The CATLASSTemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
        wrapper = V.graph.wrapper_code

        if V.graph.cpp_wrapper:
            # Make sure we initialize these kernels since they're exported as
            # C-style symbol names.
            assert isinstance(wrapper, CppWrapperCpu)
            wrapper.initialized_kernels[name] = self
            # We always originally initialize name with "KERNEL_NAME". So, we
            # we replace with the real kernel name passed as an arg to this function.
            self.signature = self.signature.replace("KERNEL_NAME", name)
            _, call_args, arg_types = self.args.cpp_argdefs()
        else:
            _, call_args, _, arg_types = self.args.python_argdefs()

        layout_args = self.get_layout_args()
        call_args.extend(layout_args)  # type: ignore[arg-type]
        arg_types.extend("int" for a in layout_args)
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            elif isinstance(arg_types[i], torch_dtype):
                call_args[i] = (
                    call_args[i]
                    if V.graph.cpp_wrapper
                    else f"c_void_p({call_args[i]}.data_ptr())"
                )

        # workspace_size ptr is NULL to mark this call is not intended for retrieving workspace_size.
        # workspace_size should have already been retrieved prior to this call.
        # workspace_size is here.
        call_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("size_t*")

        if node.get_workspace_size() > 0:
            ws = WorkspaceArg(
                count=node.get_workspace_size(),
                device=V.graph.get_current_device_or_throw(),
                zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                outer_name=WorkspaceArg.unique_name(),
            )
            wrapper.generate_workspace_allocation(ws)
            workspace = str(ws.outer_name)
            call_args.append(
                workspace
                if V.graph.cpp_wrapper
                else f"c_void_p({workspace}.data_ptr())"
            )
        else:
            ws = None
            call_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("uint8_t*")

        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=False,
            arg_types=arg_types,
        )
        if ws:
            wrapper.generate_workspace_deallocation(ws)

    def dtype(self, node: IRNode) -> Optional[str]:
        """
        Generates code which represents dtype of a given node.
        """

        if node is None:
            return "void"
        return _DTYPE_TO_CPP.get(node.get_layout().dtype)

    def catlass_dtype(self, node: IRNode, default_dtype="void") -> Optional[str]:
        # Helper method, called into from CATLASSGemmTemplate
        if node is None:
            return default_dtype
        from .catlass_template import CATLASSTemplate

        return CATLASSTemplate._DTYPE_TO_CATLASS[node.get_layout().dtype]

    def max_valid_index(self, node: IRNode, default=-1):
        # Helper method, called into from CATLASSGemmTemplate
        if node is None:
            return default
        max_valid_offset = 0
        for i in range(len(node.get_size())):
            max_valid_offset += (node.get_size()[i] - 1) * node.get_stride()[i]
        return max_valid_offset

    def offset(self, node: IRNode) -> str:
        """
        Generates code which represents offset of a given node.
        """

        if node is None:
            return "0"
        return str(node.get_layout().offset)  # type: ignore[union-attr]

    def ptr(self, node: IRNode) -> str:
        """
        Generates code which represents pointer of a given node.
        """

        if node is None:
            return "nullptr"
        arg_name = self.arg_name(node)
        if arg_name is None:
            return "nullptr"
        offset = self.offset(node)
        return arg_name if offset == "0" else f"{arg_name} + {offset}"

    def size(
        self,
        node: IRNode,
        start_index: int,
        end_index: Optional[int] = None,
        default_value: int = 0,
    ) -> str:
        """
        Hook called from template code to get the size of an arg.
        Generates code which represents size of a given node in [start_index, end_index).
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        start_index = _normalize_idx(start_index, len(node.get_size()))
        if end_index is None:
            end_index = start_index
        end_index = _normalize_idx(end_index, len(node.get_size()))
        sizes = [
            self.find_symbol(node, "size", dim=i) or node.get_size()[i]
            for i in range(start_index, end_index + 1)
        ]
        if len(sizes) == 0:
            return str(default_value)

        sizes = [symbols(v) if isinstance(v, str) else v for v in sizes]
        val = sympy_product(sizes)
        return val

    def stride(self, node: IRNode, index: int, default_value: int = 0) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Generates code which represents stride of a given node at index.
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        index = _normalize_idx(index, len(node.get_size()))
        if index < 0:
            return str(default_value)

        stride = node.get_stride()[index]
        if V.graph.sizevars.statically_known_leq(stride, 1):
            return str(stride)
        return self.find_symbol(node, "stride", dim=index) or str(stride)

    def load(self, name: str, index: Expr, mode: Any = None) -> CSEVariable:
        """
        Mock load function for memory planning to optimize allocations properly.
        """
        return self.create_cse_var(name, bounds=ValueRanges.unknown())
    
    def store(self, name: str, index: Expr, value: Any, mode: Any = None) -> None:
        """
        Mock store function for memory planning to optimize allocations properly.
        """
        self.store_buffer_names.add(name)


class CATLASSTemplateCaller(ChoiceCaller):
    """
    CATLASSTemplateCaller

    This class represents a caller for CATLASS template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CATLASSBenchmarkRequest): The benchmark request for the caller.
        template_buffer (CATLASSTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: List[Buffer],
        layout: Layout,
        make_kernel_render: Callable[[CATLASSTemplateBuffer, Optional[List[IRNode]]], str],
        bmreq: CATLASSBenchmarkRequest,
        epilogue_fusion_type: int,
        template: "CATLASSTemplate",  # type: ignore[name-defined]
        is_mix: bool,
        info_kwargs: Optional[
            Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]
        ],  # type: ignore[type-arg]
        description: str,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.fbmreq = None  # Store FusedCATLASSBenchmarkRequest
        self.epilogue_fusion_type = epilogue_fusion_type
        self.template = template
        self.is_mix = is_mix
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        assert self.bmreq is not None
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        assert self.bmreq is not None
        return self.bmreq.benchmark(
            *args, output_tensor=out
        )  # @TODO: Hack for ensuring that Catlass Kernel is preferred

    def __str__(self) -> str:
        return f"CATLASSTemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"catlass_template_kernels.{self.name}"

    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        if self.info_kwargs is not None and "op" in self.info_kwargs:
            op: Any = self.info_kwargs["op"]
            return {
                "backend": "NPU",
                "op_type": type(op).__name__,
                "op_conf_name": str(op.configuration_name()),
                "op_arch": str(op.arch_typename()),
                "tile_shape": str(op.tile_description.procedural_name()),
                "dispatch_policy": str(op.dispatch_policy_typename()),
                "swizzling": str(op.swizzle_typename()),
                "element_accumulator": str(op.accumulator_type()),
                "op_name": str(op.procedural_name()),
            }
        else:
            return {"backend": "NPU", "op_type": "unknown"}

    def output_node(self) -> TensorBox:
        self.bmreq.update_workspace_size()
        return TensorBox.create(
            CATLASSTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                workspace_size=self.bmreq.workspace_size,
                epilogue_fusion_type=self.epilogue_fusion_type,
                template=self.template,
                is_mix=self.is_mix,
            )
        )

    def get_make_kernel_render(self):
        return self.make_kernel_render
