import copy
from typing import Optional

import torch
import torch.utils._pytree as pytree
from torch._inductor.utils import IndentedBuffer
from torch.fx.node import Argument, Target

from . import config as dvm_config, is_ascend950
from .fx_pass import annotate_mm_transpose_flags, make_cast_node
from .op_emitter import DVM_OP_REGISTRY, load, store
from .util import codegen_maybe_view_load


aten = torch.ops.aten


def is_fx_dynamic(graph):
    for node in graph.graph.nodes:
        if node.op == "placeholder" or node.op == "call_function":
            val = node.meta.get("val")
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                if any(isinstance(dim, torch.SymInt) for dim in val.shape):
                    return True
            elif isinstance(val, (torch.SymInt, torch.SymFloat)):
                return True
    return False


class DvmCodegenInterpreter(torch.fx.Interpreter):
    KERNEL_NAME_PLACEHOLDER = "__DVM_KERNEL_NAME__"

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        ktype: str,
        is_dynamic: Optional[bool] = None,
    ):
        self.original_gm = gm
        self.gm = self._dvm_pass(torch.fx.GraphModule(gm, copy.deepcopy(gm.graph)))
        super().__init__(self.gm)
        self.view_fusion_level = dvm_config.view_fusion_level
        self.current_node = None
        self.cont_flag_input = []
        self.need_trans_input = []
        self.code = IndentedBuffer()

        self._analyze_graph(is_dynamic)
        self._configure_kernel(ktype)
        self._emit_kernel_header()

    def _dvm_pass(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Apply DVM-only transformations to the private codegen graph."""
        if not is_ascend950 or dvm_config.bf16_vector_keep_promoted:
            self._insert_bf16_concat_cast_pass(gm)
        return gm

    def _insert_bf16_concat_cast_pass(self, gm: torch.fx.GraphModule) -> None:
        graph = gm.graph
        changed = False
        for node in graph.find_nodes(op="call_function", target=aten.cat.default):
            dtype = node.meta["val"].dtype
            if dtype != torch.bfloat16:
                continue
            inputs = []
            for src in node.args[0]:
                # BF16 producers can become FP32 in the generated builder.
                with graph.inserting_before(node):
                    inputs.append(make_cast_node(graph, src, dtype))
                changed = True
            node.args = (inputs, *node.args[1:])
        if changed:
            graph.lint()
            gm.recompile()

    def _analyze_graph(self, is_dynamic: Optional[bool] = None) -> None:
        self.is_mix_kernel = annotate_mm_transpose_flags(self.gm)
        if is_dynamic is not None and not isinstance(is_dynamic, bool):
            raise TypeError("is_dynamic must be bool when provided")
        self.is_dynamic = is_fx_dynamic(self.gm) if is_dynamic is None else is_dynamic
        self.has_concat = False
        self.has_non_unit_inner_stride = False
        self.spec_nodes = set()
        reduction_ops = (
            aten.sum.default,
            aten.sum.dim_IntList,
            aten.amax.default,
            aten.amin.default,
        )
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor) and val.ndim > 0:
                    inner_stride = val.stride()[-1]
                    if isinstance(inner_stride, torch.SymInt) or inner_stride != 1:
                        self.has_non_unit_inner_stride = True
            elif node.op == "call_function":
                if node.target == aten.cat.default:
                    self.has_concat = True
                if node.target in reduction_ops and any(
                    user.op == "call_function" for user in node.users
                ):
                    self.spec_nodes.add(node)

    def _configure_kernel(self, ktype: str) -> None:
        self.ktype = ktype
        if ktype == "split":
            self.spec_nodes.clear()
            self.view_fusion_level = min(self.view_fusion_level, 1)
            return

        if self.is_mix_kernel:
            self.ktype = "mix"
        elif ktype == "vector" and self.spec_nodes:
            self.ktype = "spec"

        if ktype != "vector" or self.is_mix_kernel:
            self.spec_nodes.clear()

        if self.ktype == "mix":
            self.view_fusion_level = 0
            return

        # These kernels only support level-1 view fusion.
        if self.ktype == "spec" or self.has_concat:
            self.view_fusion_level = min(self.view_fusion_level, 1)
            return

        # Fractal layout is only needed for level-2 vector views.
        if (
            self.ktype == "vector"
            and self.view_fusion_level == 2
            and self.has_non_unit_inner_stride
        ):
            self.ktype = "vector:opt_fractal"

    def _emit_kernel_header(self) -> None:
        self.code.splice(f'\n"""\n{self.gm.print_readable(print_output=False)}\n"""')
        self.code.splice(
            f"{chr(64)}dvm.kernel(ktype={self.ktype!r}, dyn_shape={self.is_dynamic})"
        )
        self.code.splice(f"def {self.KERNEL_NAME_PLACEHOLDER}(k):")
        self.code.do_indent()

    def run_node(self, n: torch.fx.Node) -> Argument:
        self.current_node = n
        expr = super().run_node(n)
        if n.op == "output":
            for _expr in pytree.tree_leaves(expr):
                self.code.splice(f"{_expr}")
        else:
            self.code.splice(f"{n} = {expr}")
            if n in self.spec_nodes:
                self.code.splice("k.spec_next()")
        return f"{n}"

    def placeholder(
        self, target: "Target", args: tuple[Argument], kwargs: dict[str, Argument]
    ) -> Argument:
        meta = self.current_node.meta
        val = meta["val"]
        if isinstance(val, torch.SymInt):
            self.cont_flag_input.append(True)
            return "k.scalar(dvm.int64)"
        if isinstance(val, torch.SymFloat):
            self.cont_flag_input.append(True)
            return "k.scalar(dvm.float32)"

        shape, stride, dtype = val.shape, val.stride(), val.dtype
        is_contiguous = val.is_contiguous()

        if self.is_mix_kernel:
            trans = meta.get("trans", False)
            self.need_trans_input.append(trans)
            if trans:
                self.cont_flag_input.append(True)
                shape = [-1 if isinstance(s, torch.SymInt) else s for s in val.mT.shape]
                return load(shape, dtype)

        shape = [-1 if isinstance(s, torch.SymInt) else s for s in shape]
        stride = [-1 if isinstance(s, torch.SymInt) else s for s in stride]
        if is_contiguous:
            expr, skip_cont = load(shape, dtype), True
        else:
            expr, skip_cont = codegen_maybe_view_load(
                shape,
                stride,
                dtype,
                view_fusion_level=self.view_fusion_level,
            )
        self.cont_flag_input.append(skip_cont)
        return expr

    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Argument]
    ) -> Argument:
        if target not in DVM_OP_REGISTRY:
            raise NotImplementedError(f"{target} not implemented in DVM")
        func, _ = DVM_OP_REGISTRY.get(target)
        meta = self.current_node.meta

        if target in (aten.mm.default, aten.bmm.default):
            args = (*args, meta.get("trans_a", False), meta.get("trans_b", False))

        elif target in (aten.addmm.default, aten.baddbmm.default):
            args = (
                *args,
                meta.get("trans_a", False),
                meta.get("trans_b", False),
                meta.get("use_bias", False),
            )

        return func(*args, **kwargs)

    def output(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Argument]
    ) -> Argument:
        outs = super().output(target, args, kwargs)

        def codegen(out, node):
            if isinstance(node, torch.fx.Node):
                return store(out, node.meta["val"].dtype)
            return ""

        return pytree.tree_map(codegen, outs, self.current_node.args[0])

    def append_mfusion_kernel_profiling_metadata(
        self, kernel_name: str, num_outputs: int
    ) -> None:
        """Emit ``k.set_kernel_info`` for stable Ascend profiler names."""
        self.code.splice(f"k.set_kernel_info({kernel_name!r}, {kernel_name!r})")
