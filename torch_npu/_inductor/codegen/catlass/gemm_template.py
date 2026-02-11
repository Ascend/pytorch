from __future__ import annotations

import copy
import enum
import math
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.ir import (
    Buffer,
    ChoiceCaller,
    ComputedBuffer,
    FixedLayout,
    IRNode,
    Layout,
    Pointwise,
    ReinterpretView,
)
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import is_dynamic
from torch._inductor.virtualized import V

from ...config import catlass as catlass_config
from . import catlass_utils
from .catlass_python_evg import CatlassEVGCodegen
from .catlass_kernel import CATLASSTemplateBuffer, CATLASSTemplateKernel
from .catlass_template import CATLASSTemplate

log = logging.getLogger("torch._inductor")


# TLA Matmul template
CATLASS_TEMPLATE_1X = r"""
{{template.header(op).getvalue()}}
{{template.globals().getvalue()}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.

extern "C" {
PT_EXPORT {{kernel_call_signature}} {
    try {
    uint8_t* deviceA = {{template.catlass_type_cast(X, kernel.ptr(X))}};
    uint8_t* deviceB = {{template.catlass_type_cast(W, kernel.ptr(W))}};
    uint8_t* deviceBias = {{template.catlass_type_cast(Bias, kernel.ptr(Bias))}};
    uint8_t* deviceC = {{template.catlass_type_cast(Y, kernel.ptr(Y))}};
    {% if template.is_group_mm %}
    uint8_t* deviceGroupList = (uint8_t*) offsets;
    {% endif %}

    {{op.gen_input_template()}}
    
    {{evg_ptr}}

    {{evg_template}}

    {{op.gen_kernel_template()}}

    {{op.gen_layout_template()}}

    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    typename GemmKernel::Arguments arguments{
        {{op.gen_params_device()}}
    };
    GemmAdapter gemm_op;

    if (workspace_size) {
        *workspace_size = gemm_op.GetWorkspaceSize(arguments);
        return 0;
    }

    {
        auto status = gemm_op.CanImplement(arguments);
        CATLASS_CHECK(status);
    }
    {
        auto status = gemm_op.Initialize(arguments, workspace);
        CATLASS_CHECK(status);
    }
    {
        auto status = gemm_op(stream, aicCoreNum);
        CATLASS_CHECK(status);
    }

    }
    catch (std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        return -1;
    }
    return 0;
}
}
"""


class BiasShape(enum.IntEnum):
    NO_BIAS = 0
    N_BIAS = 1  # bias shape is (N,)
    MN_BIAS = 2  # bias shape is (M, N)


class CATLASSGemmTemplate(CATLASSTemplate, ABC):
    """
    CATLASS GEMM Template, which is used to generate CATLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
    ) -> None:
        from .catlass_utils import _catlass_tensor_from_node, _catlass_tensor_from_node_for_bias

        super().__init__("catlass_gemm", input_nodes, layout, input_reorder)
        self.alpha = alpha
        self.beta = beta
        assert len(input_nodes) == 2 or len(input_nodes) == 3 or len(input_nodes) == 4
        if len(input_nodes) == 4:
            assert self._are_inputs_layout_compatible(
                [node.get_layout() for node in input_nodes[:2]]
            )
        else:
            assert self._are_inputs_layout_compatible(
                [node.get_layout() for node in input_nodes]
            )
        self.is_group_mm = len(input_nodes) == 4
        self.offsets_size = input_nodes[3].get_size()[0] if self.is_group_mm else None
        self.is_batchmm = any(len(node.get_size()) == 3 for node in input_nodes) and not self.is_group_mm
        self.shape_desc = self.get_shape_desc(self.input_nodes)
        self.bias_shape = BiasShape.NO_BIAS
        if len(self.input_nodes) >= 3 and self.input_nodes[2]:
            bias_first_stride = self.input_nodes[2].get_stride()[-2]
            # For N = 1, cannot distinguish bias shape is (M, 1) or (1,)
            # currently use matmulBias for this case
            self.bias_shape = (
                BiasShape.MN_BIAS
                if bias_first_stride != 0 and not (self.shape_desc[1] == 1)
                else BiasShape.N_BIAS
            )
        self.shape_desc = self.shape_desc + (self.bias_shape,)
        if self.is_group_mm:
            self.input_nodes = [node for node in self.input_nodes if node]
            self.op_tensors = [_catlass_tensor_from_node(node) for node in self.input_nodes]
        elif self.bias_shape != BiasShape.NO_BIAS:
            self.op_tensors = [_catlass_tensor_from_node(node) for node in self.input_nodes[:2]]
            self.op_tensors.append(_catlass_tensor_from_node_for_bias(self.input_nodes[2]))
        else:
            self.op_tensors = [_catlass_tensor_from_node(node) for node in self.input_nodes]


    @staticmethod
    def get_shape_desc(input_nodes) -> Tuple[int, int, int]:
        X, W = input_nodes[0], input_nodes[1]
        M = X.get_size()[-2]
        K = X.get_size()[-1]
        N = W.get_size()[-1]
        shape_desc = [M, N, K]

        for i, x in enumerate(shape_desc):
            if isinstance(x, (int, sympy.Integer)):
                shape_desc[i] = int(x)
            elif isinstance(x, (sympy.Symbol, sympy.Expr)):
                x = x.subs(V.graph.sizevars.var_to_val)
                shape_desc[i] = int(x)
            else:
                raise ValueError(f"Unknown shape dim type: {type(x)}, value: {x}")
        return tuple(shape_desc)

    @staticmethod
    @abstractmethod
    def add_catlass_gemm_choices(
        choices: List[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: List[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[List[int]] = None,
        **extra_kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _are_inputs_layout_compatible(self, layouts: List[Layout]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _get_extra_inputs_and_names(
        self,
        op: "GemmKernelBase" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> Tuple[Optional[Buffer], List[Optional[Buffer]], List[str]]:
        raise NotImplementedError

    def _is_standard_matmul(self) -> bool:
        return self.alpha == 1.0 and (self.beta == 0.0 or self.beta == 1.0)

    def _add_catlass_gemm_choices(
        self,
        choices: List[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: List[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[List[int]] = None,
        **extra_kwargs,
    ) -> None:
        """
        Adds Catlass GEMM configurations choices to the auto-tuning list.

        This function mutates the passed list of choices by appending the choices for Catlass GEMM configs to it.

        Args:
            choices (list): The list to which choices are appended.
            layout (ir.Layout): The layout configuration.
            input_nodes (list): The list of input nodes.
            alpha (float,int): Scaling factor, defaults to 1.
            beta (float,int): Offset, defaults to 0.
            input_reorder (list, optional): Order of the inputs, defaults to None.
            **extra_kwargs: Additional keyword arguments.

        """
        ops = self.gen_ops()
        for name, op in ops:
            self.maybe_append_choice(
                choices,
                description=name,
                op=op,
            )
        if len(ops) == 0:
            input_layouts = [node.get_layout() for node in self.input_nodes]
            input_strides = [node.get_stride() for node in self.input_nodes]
            output_layout = layout
            warning_msg = f"No suitable Catlass GEMM configs found, fallbacks used ( {len(ops)=}, {output_layout=}, {input_layouts=}, {input_strides=} )"  # noqa: B950
            log.warning(warning_msg)
        log.debug(
            "Added %d Catlass gemm configs.",
            len(ops),
        )


    def gen_ops(
        self,
    ) -> "List[Tuple[str, GemmKernelBase]]":  # type: ignore[name-defined]  # noqa: F821
        """
        Creates a list of Catlass GemmKernelBase instances that match the operation this template is designed to represent.
        The matching is carried out with respect to the input and output specifications of the operation.

        No function arguments.

        Returns:
            List[Tuple[str, GemmKernelBase]]: A list of (catlass_name, GemmKernelBase)
            tuples that are compatible with the operation requirements of this template.
        """
        from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase

        ops = catlass_utils.gen_ops(self.op_tensors, self.is_group_mm)
        res: Dict[str, GemmKernelBase] = {op.gen_kernel_name(): op for op in ops}

        log.debug("Got catlass configs: total number of ops: %d, ", len(res))
        return list(res.items())[: catlass_config.catlass_max_profiling_configs]

    def header(self, op) -> IndentedBuffer:
        """
        # Returns a buffer containing CUDA C++ code for the header section of the CATLASS GEMM template.
        This section primarily includes the necessary header files.

        Returns:
            IndentedBuffer: An instance of IndentedBuffer that contains the generated C++ header code.
        """
        res = super().header()
        res.splice(op.gen_includes())
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        if not self._is_standard_matmul() or self.bias_shape == BiasShape.MN_BIAS:
            res.splice(
                """
                    // Workspace util funcs
                    layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
                    {
                        if (align == 0) {
                            return layout;
                        }
                        return layout::RowMajor(layout.shape(0), layout.shape(1), RoundUp(layout.shape(1), align));
                    }

                    
                    layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
                    {
                        if (align == 0) {
                            return layout;
                        }
                        return layout::ColumnMajor(layout.shape(0), layout.shape(1), RoundUp(layout.shape(0), align));
                    }

                    
                    size_t GetWorkspaceLen(layout::RowMajor layout)
                    {
                        return layout.shape(0) * layout.stride(0);
                    }

                    
                    size_t GetWorkspaceLen(layout::ColumnMajor layout)
                    {
                        return layout.shape(1) * layout.stride(1);
                    }

                    bool IsSameStride(layout::RowMajor layout1, layout::RowMajor layout2)
                    {
                        return layout1.stride(0) == layout2.stride(0);
                    }

                    bool IsSameStride(layout::ColumnMajor layout1, layout::ColumnMajor layout2)
                    {
                        return layout1.stride(1) == layout2.stride(1);
                    }

                """
            )
        else:
            res.splice(
                """
                    bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
                    {
                        // If the stride is greater than 65536, padding is required to reduce the stride.
                        if (layout.stride(0) < 65536) {
                            return layout.stride(0) % align != 0;
                        } else {
                            return true;
                        }
                    }

                    bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
                    {
                        // If the stride is greater than 65536, padding is required to reduce the stride.
                        if (layout.stride(1) < 65536) {
                            return layout.stride(1) % align != 0;
                        } else {
                            return true;
                        }
                    }

                    bool IsNeedPadding(layout::zN layout, uint32_t align)
                    {
                        return false;
                    }

                    bool IsNeedPadding(layout::nZ layout, uint32_t align)
                    {
                        return false;
                    }
                """
            )
        return res

    @staticmethod
    def catlass_layout(torch_layout: ir.Layout) -> "Optional[catlass_lib_layout.Layout]":  # type: ignore[name-defined]  # noqa: F821
        """
        Converts an ir.Layout instance into the corresponding catlass layout str
        (RowMajor, ColumnMajor, VectorLayout or None if no matching value is found ).

        Args:
            torch_layout (ir.Layout): The layout that needs to be looked up.

        Returns:
            str: The converted layout corresponding to the `torch_layout` or None if no matching
            value is found.
        """
        import catlass_cppgen.catlass.layout as catlass_lib_layout

        # bias stride could be (0, 1), which indicates (n,) bias 
        if len(torch_layout.stride) == 1 or torch_layout.stride[0] == 0:
            return catlass_lib_layout.VectorLayout

        if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
            return catlass_lib_layout.RowMajor
        elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-2], 1):
            return catlass_lib_layout.ColumnMajor
        else:
            return None

    @staticmethod
    def layout_match(
        torch_layout: ir.Layout,
        catlass_layout: "catlass_lib_layout.Layout",  # type: ignore[name-defined] # noqa: F821
    ) -> bool:
        """Helper Method: Determines whether a given torch layout matches a given Catlass layout"""
        return isinstance(catlass_layout, CATLASSGemmTemplate.catlass_layout(torch_layout))

    def render(  # type: ignore[override]
        self,
        kernel: CATLASSTemplateKernel,
        op: "GemmKernelBase" = None,  # type: ignore[name-defined]  # noqa: F821
        template_buffer_node: Optional[CATLASSTemplateBuffer] = None,
        epilogue_nodes: Optional[List[BaseSchedulerNode]] = None,
        **kwargs,
    ) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        Renders the Catlass based C++ code for the GEMM Kernel that this template is designed to implement,
        including potentially fused epilogues.

        Args:
            kernel (CATLASSTemplateKernel): The kernel to be rendered.
            op (GemmKernelBase, optional): A GEMM operation that is required to be compatible with the
                input and output definitions as well as a possible epilogue. Defaults to None.
            **kwargs: Additional keyword arguments. Currently unused.

        Returns:
            str: Catlass based C++ code fragment as a string, to be used by the current
            CATLASSTemplateKernel or autotuning code.

        Note:
            All inputs and their corresponding buffer addresses and names take precedence over previously
            passed inputs to the template at construction time. However, they should be layout compatible.
        """
        from catlass_cppgen.kernel.gemm.gemm_base import GemmKernelBase
        from catlass_cppgen.common.data_type import DataType

        assert isinstance(
            op, GemmKernelBase
        ), "op argument is required and has to be an instance of GemmKernelBase"

        assert len(self.input_nodes) >= 2 and self.output_node is not None
        # check if we can freeze FlexibleLayout
        # since this operation could make the linearization failed
        safe_to_freeze_layout = True
        for input_node in self.input_nodes:
            if (
                not isinstance(input_node.layout, FixedLayout)
                and len(input_node.get_size()) > 2
            ):
                safe_to_freeze_layout = False
                break
        if not safe_to_freeze_layout:
            raise NotImplementedError("Layout is not fixed")

        for input_node in self.input_nodes:
            if not isinstance(input_node.layout, FixedLayout):
                input_node.freeze_layout()
        X, W = self.input_nodes[0], self.input_nodes[1]

        Y = self.output_node
        if template_buffer_node is not None:
            Y = template_buffer_node

        Bias, extra_inputs, extra_names = self._get_extra_inputs_and_names()

        # Define Kernel call signature
        # Important: This step also populates Kernel name to node mapping data structures,
        # which are required further below (for example by the template renderer)
        inputs = [X, W, Bias, *extra_inputs]
        names = ["X", "W", "Bias", *extra_names] + ["Y"]
        names_str = ",".join(names)
        input_reorder = self.input_reorder

        # The layouts might have changed between autotuning and this call if they were FlexibleLayout
        # we need to adapt, which might lead to suboptimal performance.
        op = self.fix_op_layout(op, X, W, Bias, Y)

        relu_enabled = False
        name_to_buffer = {node.get_name(): node for node in self.input_nodes}
        # handle the fake output buffer during lowering
        name_to_buffer[Y.get_name()] = Y  # type: ignore[assignment]
        # Fuse the epilogue nodes on-fly or using Catlass Epilogue Visitor Graph (EVG)
        is_evg_fusion = False
        if epilogue_nodes:
            op = copy.deepcopy(op)
            try:
                (relu_enabled, bias_buffer, output_buffer) = self._try_fast_fusion(
                    epilogue_nodes, Y.get_name()
                )

                from .catlass_library.gemm_autotune import may_adjust_l1_tile_for_bias
                from .catlass_utils import _catlass_tensor_from_node_for_bias

                if bias_buffer:
                    bias_tensor = _catlass_tensor_from_node_for_bias(bias_buffer)
                    op.element_Bias = bias_tensor.dtype
                    op.layout_Bias = bias_tensor.layout
                    # Add bias for fusion may exceed L1 tile
                    # so we will try to adjust l1 tiling before rendering
                    may_adjust_l1_tile_for_bias(op)

                if relu_enabled:
                    if not getattr(op, "is_support_relu", False):
                        raise NotImplementedError(
                            f"{type(op).__name__} does not support passing relu"
                        )
                    op.tune(relu_enable=True)

                Bias = bias_buffer
                Y = output_buffer
                evg_template = ""
                evg_ptr = ""
                inputs = [X, W, Bias, *extra_inputs]
                outputs = [Y]
            except NotImplementedError as e:
                log.debug(
                    f"Cannot fuse epilogue nodes on-fly, reason: {e}, will use EVG to fuse."
                )

                (
                    input_names,
                    output_names,
                    var_name_to_buffer_name,
                    evg_py_code,
                ) = CatlassEVGCodegen.ir_to_evg_python_code(
                    Y.get_name(), epilogue_nodes, V.kernel.removed_buffers
                )

                for name, buf in (
                    V.graph.name_to_buffer | V.graph.graph_inputs
                ).items():
                    if name not in name_to_buffer:
                        name_to_buffer[name] = buf

                D_output_name = var_name_to_buffer_name["D"]
                D_output_buffer = name_to_buffer[D_output_name]
                Y = D_output_buffer  # type: ignore[assignment]
                # Interestingly, I don't think the rest of the layout matters here since we
                # use the properties of the Y buffer to fill in D's properties in the epilogue
                # args. This is needed though because it defines types expected in the epilogue args.
                op.element_C = DataType.from_dtype(D_output_buffer.get_dtype())
                is_evg_fusion = True

                assert output_names, "There should be at least one write"

                epilogue_inputs = [name_to_buffer[name] for name in input_names]
                output_names.remove(Y.get_name())  # remove duplicated output
                outputs = [name_to_buffer[name] for name in output_names]

                op = self._render_evg(
                    op,
                    evg_py_code,
                    var_name_to_buffer_name,
                    name_to_buffer,
                )
                template_buffer_node.is_mix = True
                if Bias is not None:
                    assert Bias.get_layout().dtype == X.get_layout().dtype
                    # This might have been set to void during filtering, when the assumption was still that there's no C
                    # operand
                    op.element_Bias = op.element_A

                evg_template = op.gen_evg_template()

                inputs = [
                    X,
                    W,
                    Bias,
                    *epilogue_inputs,  # type: ignore[list-item]
                    Y,
                    *extra_inputs,
                ]
                evg_ptr = "\n"
                for name, _ in zip(input_names, epilogue_inputs):
                    evg_ptr += f"    uint8_t* {name}_ptr = (uint8_t*)({name});\n"
                names_str = ",".join(
                    ["X", "W", "Bias", *input_names, "Y", *output_names, *extra_names]
                )
        else:
            # no epilogue nodes
            outputs = [Y]
            evg_template = ""
            evg_ptr = ""

        kernel_call_signature = kernel.def_kernel(
            inputs=inputs,
            outputs=outputs,
            names_str=names_str,
            input_reorder=input_reorder,
        )

        options = dict(
            X=X,
            W=W,
            Y=Y,
            kernel_call_signature=kernel_call_signature,
            Bias=Bias,
            template=self,
            evg_template=evg_template,
            evg_ptr=evg_ptr,
            kernel=kernel,
            op=op,
        )
        options.update(dict(zip(extra_names, extra_inputs)))
        res = self._template_from_string(CATLASS_TEMPLATE_1X).render(
            **options
        )

        return res

    def fix_op_layout(
        self,
        op: "GemmKernelBase",  # type: ignore[name-defined] # noqa: F821
        X: Buffer,
        W: Buffer,
        Bias: Optional[Buffer],
        Y: Union[Buffer, ReinterpretView],
    ) -> "GemmKernelBase":  # type: ignore[name-defined]  # noqa: F821
        # This is a workaround to deal with cases where the input layouts have changed
        # between autotuning and rendering. This happens if the inputs layout
        # are FlexibleLayout instances. In this case, we need to update the
        # op's input layouts. It is a hack, because now the op
        # we benchmarked is not the same as the op we render,
        # but there is no simple way to fix this in the autotuner, since that would
        # potentially disable other optimizations.
        from catlass_cppgen.common.utils import infer_layout_from_stride
        from catlass_cppgen.common.data_type import DataType

        a_layout = X.get_layout()
        b_layout = W.get_layout()
        bias_layout = Bias.get_layout() if Bias is not None else None

        c_layout = copy.deepcopy(Y.get_layout())
        match_list = []
        for buf, op_layout in zip(
                (X, W, Bias, Y),
                (op.layout_A, op.layout_B, op.layout_Bias, op.layout_C),
            ):
            if buf is not None:
                result = CATLASSGemmTemplate.layout_match(buf.get_layout(), op_layout)
                match_list.append(result)
        all_match = all(match_list)
        if all_match:
            return op
        log.warning(
            f"Catlass GEMM Layout change: Input and/or output layouts have changed between autotuning/retuning and call to render on {self}. Applying workaround. This can lead to suboptimal performance. Match List: {match_list}"  # noqa: G004, B950
        )
        new_op = copy.deepcopy(op)

        if a_layout is not None:
            new_op.layout_A = infer_layout_from_stride(a_layout.size, a_layout.stride)
        if b_layout is not None:
            new_op.layout_B = infer_layout_from_stride(b_layout.size, b_layout.stride)
        if bias_layout is not None:
            new_op.layout_Bias = infer_layout_from_stride(bias_layout.size, bias_layout.stride)
            new_op.element_Bias = DataType.from_dtype(bias_layout.dtype)
        if c_layout is not None:
            new_op.layout_C = infer_layout_from_stride(c_layout.size, c_layout.stride)
        return new_op

    def test_call_statement(
        self,
        kernel,
        input_nodes,
        names_str: str = "",
    ) -> str:
        """
        Helper method to render the Catlass C++ code required for calling the GEMM operation in the standalone
        test runner that might also be generated along with the rest of the code, if the corresponding config is
        enabled.

        Returns a C++ statement that calls the GEMM operation with the correct arguments.
        """
        _, __, arg_types = kernel.args.cpp_argdefs()
        arg_names = [name.strip() for name in names_str.strip().split(",")]
        if input_nodes[2] is None:
            del arg_names[2]
        arguments = [
            f"(({arg_type}){arg_name}_data.get())"
            for arg_type, arg_name in zip(arg_types, arg_names)
        ]
        return f"{kernel.kernel_name}({', '.join(arguments)}, workspace_size_ptr, (uint8_t*)workspace_data.get(), 0);"

    @staticmethod
    def _try_fast_fusion(
        epilogue_nodes: List[BaseSchedulerNode], template_output_name: str
    ):
        raise NotImplementedError(
            "_try_fast_fusion in CATLASSGemmTemplate not implemented"
        )

    def _render_evg(
        self,
        op: "GemmKernelBase",
        evg_py_code: str,
        buffer_renames: dict[str, str],
        name_to_buffer: dict[str, Buffer],
        output_dtype: torch.dtype,
        accumulator_dtype: torch.dtype,
    ) -> "GemmKernelBase":  # type: ignore[name-defined]  # noqa: F821
        raise NotImplementedError("_render_evg in CATLASSGemmTemplate not implemented")


class CATLASS1xGemmTemplate(CATLASSGemmTemplate):
    def __init__(
        self,
        input_nodes: List[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
    ):
        super().__init__(input_nodes, layout, alpha, beta, input_reorder)

    @staticmethod
    def add_catlass_gemm_choices(
        choices: List[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: List[Buffer],
        alpha: Union[float, int] = 1,
        beta: Union[float, int] = 0,
        input_reorder: Optional[List[int]] = None,
        **extra_kwargs,
    ) -> None:
        template = CATLASS1xGemmTemplate(
            input_nodes, layout, alpha, beta, input_reorder,
        )
        template._add_catlass_gemm_choices(
            choices, layout, input_nodes, alpha, beta, input_reorder, **extra_kwargs
        )


    def _get_extra_inputs_and_names(
        self,
        op: "GemmKernelBase" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> Tuple[Optional[Buffer], List[Optional[Buffer]], List[str]]:
        Bias = None if (len(self.input_nodes) == 2 or self.is_group_mm) else self.input_nodes[2]
        if self.is_group_mm:
            inputs: List[Optional[Buffer]] = [self.input_nodes[2]]
            names: List[str] = ["offsets"]
        else:
            inputs: List[Optional[Buffer]] = []
            names: List[str] = []
        return (Bias, inputs, names)

    @staticmethod
    def is_mixed_template(op: "GemmKernelBase") -> bool:
        return getattr(op, "is_mix", False)
    
    @staticmethod
    def epilogue_fusion_type(op: "GemmKernelBase") -> int:
        fusion_type = 0
        if getattr(op, "is_support_relu", False):
            # support fast fusion
            fusion_type = 1
        if getattr(op, "is_support_evg", False):
            # support EVG fusion
            fusion_type = 2

        return fusion_type

    def _are_inputs_layout_compatible(self, layouts: List[Layout]) -> bool:
        """
        Evaluates whether input layouts are compatible for set of operations supported by this class.

        Args:
            layouts (List[Layout]): List containing Layout objects representing
                                    the input matrices

        Returns:
            bool: True if layouts are GEMM compatible, otherwise False.
        """
        assert len(layouts) == 2 or len(layouts) == 3
        # Check if A and B are compatible
        A_layout, B_layout = layouts[:2]
        if len(A_layout.size) < 1:
            return False
        if len(B_layout.size) < 1:
            return False
        A_size = list(V.graph.sizevars.size_hints(A_layout.size))
        B_size = list(V.graph.sizevars.size_hints(B_layout.size))
        if len(A_size) < 2:
            A_size.insert(0, 1)
        if len(B_size) < 2:
            A_size.insert(1, 1)
        # Are batch dims broadcastable?
        while len(A_size) < len(B_size):
            A_size.insert(0, 1)
        while len(B_size) < len(A_size):
            B_size.insert(0, 1)
        K = max(A_size[-1], B_size[-2])
        M = A_size[-2]
        N = B_size[-1]
        if K != A_size[-1] and A_size[-1] != 1:
            return False
        if K != B_size[-2] and B_size[-1] != 1:
            return False
        # check batch dim broadcastable
        for i in range(len(A_size) - 2):
            if A_size[i] != B_size[i] and A_size[i] != 1 and B_size[i] != 1:
                return False
        if len(layouts) == 3:
            C_layout = layouts[2]
            C_size = [int(i) for i in C_layout.size]
            while len(C_size) < len(A_size):
                C_size.insert(0, 1)
            # check batch dims
            for i in range(len(A_size) - 2):
                bd = max(A_size[i], B_size[i])
                if bd != C_size[i] and C_size[i] != 1:
                    return False
            if len(C_size) > len(A_size):
                # This may happen if the last elements of C are contiguous and
                # their multiplied size equals the last dim size of B
                if M != C_size[len(A_size) - 2] and C_size[len(A_size) - 2] != 1:
                    return False
                remaining_size = 1
                for i in range(len(A_size) - 1, len(C_size)):
                    remaining_size *= C_size[i]
                if N != remaining_size and remaining_size != 1:
                    return False
                return True
            assert len(C_size) == len(A_size)
            if M != C_size[-2] and C_size[-2] != 1:
                return False
            if N != C_size[-1] and C_size[-1] != 1:
                return False
        return True

    @staticmethod
    def _try_fast_fusion(
        epilogue_nodes: List[BaseSchedulerNode], template_output_name: str
    ):
        if len(epilogue_nodes) > 1:
            raise NotImplementedError(
                "Do not support more than one epilogue nodes for fast-fusion."
            )

        node = epilogue_nodes[0]
        cb = node.node
        assert isinstance(cb, ComputedBuffer)
        pw = cb.data
        assert isinstance(pw, Pointwise)
        op_count_res = pw.inner_fn_opcount()
        used_ops = set(op_count_res.used_ops)
        used_ops.discard("load")
        num_ops = op_count_res.num_ops - len(op_count_res.read_buffers)
        supported_on_fly_ops = {"add", "relu"}
        if len(used_ops.difference(supported_on_fly_ops)) != 0:
            raise NotImplementedError(
                "There are ops that are not supported for fast-fusion."
            )
        if len(used_ops) > 2 or len(used_ops) != num_ops:
            raise NotImplementedError(
                "Do not support more than one add or relu for fast-fusion."
            )

        # Only support biasAdd, Relu, biasAdd+Relu on fly
        bias_buffer = None
        relu_enabled = False

        if "add" in used_ops:
            read_names = list(pw.get_read_names())
            for name in read_names:
                if name == template_output_name:
                    continue
                if name in V.graph.name_to_buffer:
                    buf = V.graph.name_to_buffer.get(name)
                elif name in V.graph.graph_inputs:
                    buf = V.graph.graph_inputs.get(name)
                else:
                    raise KeyError(
                        f"Cound not resolve buffer for name {name} (maybe removed)."
                    )

                buf_stride = buf.get_layout().stride
                if len(buf_stride) > 1 or not V.graph.sizevars.statically_known_equals(
                    buf_stride[0], 1
                ):
                    raise NotImplementedError("Do not support matrix-Add for fast-fusion.")
                assert bias_buffer is None
                bias_buffer = buf
        if "relu" in used_ops:
            relu_enabled = True

        output_name = cb.get_name()
        output_buffer = V.graph.name_to_buffer.get(output_name)
        return (relu_enabled, bias_buffer, output_buffer)

    def _render_evg(
        self,
        op: "GemmKernelBase",
        evg_py_code: str,
        var_name_to_buffer_name: dict[str, str],
        name_to_buffer: dict[str, Buffer],
    ) -> "GemmKernelBase":  # type: ignore[name-defined]  # noqa: F821
        from .catlass_library.evg_extension import create_example_tensors

        examples = create_example_tensors(
            var_name_to_buffer_name,
            name_to_buffer,  # type: ignore[arg-type]
            V.graph.sizevars.size_hint,
        )

        evg_config = {
            "fn_src": evg_py_code,
            "example_inputs": examples
        }

        return op.to_evg(evg_config)