import torch
import torch.utils._pytree as pytree
from torch.fx.node import Argument, Target
from torch._inductor.utils import IndentedBuffer
from .op_emitter import DVM_OP_REGISTRY, load, store, view_load
from .fx_pass import annotate_mm_transpose_flags

aten = torch.ops.aten


def is_fx_dynamic(graph):
    for node in graph.graph.nodes:
        if node.op == "placeholder" or node.op == "call_function":
            if isinstance(node.meta["val"], torch.Tensor):
                if any(isinstance(dim, torch.SymInt) for dim in node.meta["val"].shape):
                    return True
            elif isinstance(node.meta["val"], (torch.SymInt, torch.SymFloat)):
                return True
    return False


class DvmCodegenInterpreter(torch.fx.Interpreter):
    KERNEL_NAME_PLACEHOLDER = "__DVM_KERNEL_NAME__"

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        ktype: str,
        uncont_policy="fuse",
    ):
        super().__init__(gm)
        self.gm = gm
        self.ktype = ktype
        self.is_mix_kernel = annotate_mm_transpose_flags(gm)
        self.is_dynamic = is_fx_dynamic(gm)
        self.current_node = None
        self.cont_flag_input = []
        self.need_trans_input = []
        self.use_view = uncont_policy == "fuse"
        self.code = IndentedBuffer()

        self.spec_nodes = set()
        if self.ktype == "vector" and self.need_spec():
            self.ktype = "spec"
        self.code.splice(
            f'\n"""\n{self.gm.print_readable(print_output=False)}\n"""')
        decorator = (
            f"{chr(64)}dvm.kernel(ktype={self.ktype!r}, dyn_shape={self.is_dynamic})"
        )
        self.code.splice(decorator)
        self.code.splice(f"def {self.KERNEL_NAME_PLACEHOLDER}(k):")
        self.code.do_indent()

    def need_spec(self) -> bool:
        self.spec_nodes.clear()
        for node in self.gm.graph.nodes:
            if node.op != "call_function":
                continue
            for input_node in node.all_input_nodes:
                if input_node.op == "call_function" and input_node.target in [
                    aten.sum.default,
                    aten.sum.dim_IntList,
                    aten.amax.default,
                    aten.amin.default,
                ]:
                    self.spec_nodes.add(input_node)
        return len(self.spec_nodes) > 0

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

        is_contiguous = val.is_contiguous()
        shape, stride, dtype = val.shape, val.stride(), val.dtype
        is_symbolic = any(
            isinstance(s, torch.SymInt) and s.node.is_symbolic() for s in shape
        )
        shape = [-1 if isinstance(s, torch.SymInt) else s for s in shape]
        stride = [-1 if isinstance(s, torch.SymInt) else s for s in stride]
        self.need_trans_input.append(meta.get("trans", False))
        if self.is_mix_kernel:
            if meta.get("trans", False):
                self.cont_flag_input.append(True)
                shape = val.mT.shape
                return load(shape, dtype)
            else:
                self.cont_flag_input.append(is_contiguous)
                return load(shape, dtype)
        else:
            if is_contiguous:
                self.cont_flag_input.append(True)
                return load(shape, dtype)
            else:
                if is_symbolic or not self.use_view:
                    self.cont_flag_input.append(False)
                    return load(shape, dtype)
                else:
                    if stride[-1] == 1 and shape[-1] != 1:
                        self.cont_flag_input.append(True)
                        return view_load(shape, stride, 0, dtype)
                    else:
                        self.cont_flag_input.append(False)
                        return load(shape, dtype)

    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Argument]
    ) -> Argument:
        if target not in DVM_OP_REGISTRY:
            raise NotImplementedError(f"{target} not implemented in DVM")
        func, _ = DVM_OP_REGISTRY.get(target)
        meta = self.current_node.meta

        if target in (aten.mm.default, aten.bmm.default):
            args = (*args, meta.get("trans_a", False),
                    meta.get("trans_b", False))

        elif target is aten.addmm.default:
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
