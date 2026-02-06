import copy
from collections import defaultdict

import torch
from torch.fx import Graph, GraphModule, Node
from torch.library import custom_op
from torch.utils._pytree import tree_map
from torch._inductor import config as inductor_config
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._subclasses import FakeTensor

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.fx.passes.tools_common import legalize_graph
from torch.fx.passes.utils.fuser_utils import (
    topo_sort,
    fuse_as_graphmodule,
    erase_nodes,
)

from .graph_build import DvmCodegenInterpreter
from .fx_test import generate_dvm_fx_case
from .op_emitter import DVM_OP_REGISTRY
from .fx_pass import (
    insert_promote_cast_by_pos_prims,
    insert_sum_fp32_prepost_cast_prims,
    expand_to_reshape,
    need_fallback_gm,
)

dump_fx_test = False

uncont_policy = "fuse"

aten = torch.ops.aten
prims = torch.ops.prims

GRAPH_FUSION_SUPPORT_OP = [
    aten.add.Tensor,
    aten.add.Scalar,
    aten.sub.Tensor,
    aten.sub.Scalar,
    aten.mul.Tensor,
    aten.mul.Scalar,
    aten.div.Tensor,
    aten.div.Scalar,
    aten.pow.Tensor_Tensor,
    aten.pow.Tensor_Scalar,
    aten.pow.Scalar,
    aten.lt.Tensor,
    aten.lt.Scalar,
    aten.le.Tensor,
    aten.le.Scalar,
    aten.gt.Tensor,
    aten.gt.Scalar,
    aten.ge.Tensor,
    aten.ge.Scalar,
    aten.eq.Tensor,
    aten.eq.Scalar,
    aten.ne.Tensor,
    aten.ne.Scalar,
    aten.maximum.default,
    aten.minimum.default,
    aten.sqrt.default,
    aten.rsqrt.default,
    aten.abs.default,
    aten.log.default,
    aten.exp.default,
    aten.reciprocal.default,
    aten.isfinite.default,
    prims.convert_element_type.default,
    torch.ops.npu.npu_dtype_cast.default,
    torch.ops.npu.npu_dtype_cast_backward.default,
    torch.ops.npu._npu_dtype_cast.default,
    torch.ops.npu._npu_dtype_cast_backward.default,
    aten.sum.dim_IntList,
    aten.sum.default,
    aten.neg.default,
    aten.relu.default,
    aten.mm.default,
    aten.bmm.default,
    aten.addmm.default,
    aten.where.default,
    aten.where.self,
    # aten.expand.default,
    # aten.full.default,
    # aten.reshape.default,
]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Node, Node] = {}
        self.rank: dict[Node, int] = {}

    def find(self, x: Node) -> Node:
        p = self.parent.get(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        else:
            self.parent[x] = x
        return self.parent[x]

    def union(self, a: Node, b: Node) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return

        rka = self.rank.get(ra, 0)
        rkb = self.rank.get(rb, 0)
        if rka < rkb:
            ra, rb = rb, ra
            rka, rkb = rkb, rka

        self.parent[rb] = ra
        if rka == rkb:
            self.rank[ra] = rka + 1


class DvmOpSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node):
        if node.op == "call_function" and node.target in GRAPH_FUSION_SUPPORT_OP:
            _, rule = DVM_OP_REGISTRY.get(node.target)
            return rule(node)
        return False


def split_partition_with_union_find(
    partition_nodes: dict[Node, None],
) -> list[dict[Node, None]]:
    """
    Split a partition into connected components based on data-dependency edges
    within the partition: if u is an input of v and both are in partition,
    they belong to the same component.
    """
    nodes = list(partition_nodes.keys())
    node_set = set(nodes)

    uf = UnionFind()
    for n in nodes:
        uf.find(n)

    for v in nodes:
        for u in v.all_input_nodes:
            if u in node_set:
                uf.union(u, v)

    groups: dict[Node, dict[Node, None]] = defaultdict(dict)
    for n in nodes:
        root = uf.find(n)
        groups[root][n] = None

    return list(groups.values())


class _FusedMeta:
    def __init__(self, name: str, gm: GraphModule, input_nodes: list[Node]):
        self.name = name
        self.gm = gm

    def codegen(self):
        """
        Return (codegen_interpreter, python_source_string).
        """
        if dump_fx_test:
            generate_dvm_fx_case(self.gm, fusion_type="graph")
        cg = DvmCodegenInterpreter(self.gm, ktype="split")
        cg.run()
        code = cg.code.getvalue().replace(cg.KERNEL_NAME_PLACEHOLDER, self.name)
        return cg, code


# fused_id -> meta
_fused_metas: dict[int, _FusedMeta] = {}


def _fused_run_stub(*args, **kwargs):
    raise AssertionError("This op should never run at eager runtime.")


def _fused_run_fake(*args, **kwargs):
    """
    Fake implementation for custom_op. The last arg is fused_id.
    We execute sub_gm.forward on meta tensors and return contiguous meta outputs.
    """
    fused_id = int(args[-1])
    meta = _fused_metas[fused_id]
    out = meta.gm.forward(*args[:-1])
    return tree_map(lambda t: t if t.is_contiguous() else t.contiguous(), out)


class GraphFusionPartitioner(CapabilityBasedPartitioner):
    fused_op_map_: dict[str, object] = {}
    fused_id_: int = 0

    def _input_schema_types(self, input_nodes: list[Node]) -> list[str]:
        input_types: list[str] = []
        for node in input_nodes:
            val = node.meta.get("val", None)
            if isinstance(val, torch.SymInt):
                input_types.append("SymInt")
            elif isinstance(val, torch.SymFloat):
                input_types.append("SymFloat")
            else:
                input_types.append("Tensor")
        return input_types

    def _build_schema(self, input_types: list[str], output_len: int) -> str:
        extra_inputs = "int fused_id"
        input_schema = ", ".join(
            f"{input_type} x{i}" for i, input_type in enumerate(input_types)
        )
        input_schema = (
            f"{input_schema}, {extra_inputs}" if input_schema else extra_inputs
        )

        if output_len == 1:
            output_schema = "Tensor"
        else:
            output_schema = f'({", ".join(["Tensor"] * output_len)})'
        return f"({input_schema}) -> {output_schema}"

    def _get_or_create_custom_op(self, input_nodes: list[Node], output_len: int):
        input_types = self._input_schema_types(input_nodes)
        type_suffix = "_".join(
            "t" if t == "Tensor" else "si" if t == "SymInt" else "sf"
            for t in input_types
        )
        input_len = len(input_types)
        op_def = f"{input_len}_{output_len}"
        if any(t != "Tensor" for t in input_types):
            op_def = f"{op_def}_{type_suffix}"
        custom = self.fused_op_map_.get(op_def, None)
        if custom is not None:
            return custom
        schema = self._build_schema(input_types, output_len)

        custom = custom_op(
            "dvm::fused_graph_" + op_def,
            _fused_run_stub,
            mutates_args=(),
            device_types="npu",
            schema=schema,
        )
        custom.register_fake(_fused_run_fake)
        self.fused_op_map_[op_def] = custom
        return custom

    def _should_fuse(
        self,
        sub_gm: GraphModule,
        orig_outputs: list[Node],
        orig_inputs: list[Node],
    ) -> bool:
        if len(orig_outputs) == 0:
            return False
        if need_fallback_gm(sub_gm):
            return False
        return any(
            isinstance(node.meta.get("val", None), FakeTensor) for node in orig_inputs
        )

    def partition_and_fuse(self) -> GraphModule:
        partitions = self.propose_partitions()

        # further split each proposed partition by union-find connectivity
        partition_nodes_list: list[dict[Node, None]] = [
            sp_nodes
            for partition in partitions
            for sp_nodes in split_partition_with_union_find(partition.nodes)
        ]

        for partition_nodes in partition_nodes_list:
            fused_id = self.fused_id_
            self.fused_id_ += 1

            fused_name = f"dvm_graph_fused_{fused_id}"
            sorted_nodes = topo_sort(list(partition_nodes))

            sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
                self.graph_module,
                sorted_nodes,
                fused_name,
                partition_nodes,
            )

            # post-processing inside sub graph
            insert_promote_cast_by_pos_prims(sub_gm)
            insert_sum_fp32_prepost_cast_prims(sub_gm)
            expand_to_reshape(sub_gm)

            if not self._should_fuse(sub_gm, orig_outputs, orig_inputs):
                continue

            _fused_metas[fused_id] = _FusedMeta(fused_name, sub_gm, orig_inputs)

            output_len = len(orig_outputs)
            custom = self._get_or_create_custom_op(orig_inputs, output_len)

            # create fused call node in original graph
            args = (*orig_inputs, fused_id)
            new_node = self.graph_module.graph.call_function(
                custom._opoverload, tuple(args), None
            )

            if output_len == 1:
                orig_outputs[0].replace_all_uses_with(new_node)
                new_node.meta["val"] = orig_outputs[0].meta.get("val", None)
            else:
                for i, orig_output in enumerate(orig_outputs):
                    proxy_out = torch.fx.Proxy(new_node)[i].node
                    proxy_out.meta["val"] = copy.copy(orig_output.meta["val"])
                    orig_output.replace_all_uses_with(proxy_out)
                new_node.meta["val"] = tuple(
                    copy.copy(out.meta["val"]) for out in orig_outputs
                )

            # erase old nodes
            erase_nodes(self.graph_module, sorted_nodes)

        legalize_graph(self.graph_module)
        return self.graph_module


def dvm_graph_fusion(graph: Graph):
    gm: GraphModule = graph.owning_module

    dvm_support = DvmOpSupport()
    fusion_part = GraphFusionPartitioner(
        gm,
        dvm_support,
        allows_single_node_partition=True,
    )
    fusion_part.partition_and_fuse()


def _dvm_generate_fallback_kernel(self, fallback_kernel, args):
    """
    Patch point: PythonWrapperCodegen.generate_fallback_kernel
    If it's our custom op, pop meta and emit dvm kernel code.
    """
    if not fallback_kernel.op_overload._name.startswith("dvm::fused_graph_"):
        self.generate_extern_kernel_alloc(fallback_kernel, args)
        return

    fused_id = int(args[-1])
    meta = _fused_metas.pop(fused_id)

    cg, code = meta.codegen()
    self.header.splice(code)

    buf_name = fallback_kernel.get_name()

    args_list = list(args[:-1])
    # cont/trans handling based on codegen interpreter
    for i, no_trans in enumerate(cg.cont_flag_input):
        if not no_trans:
            args_list[i] += ".contiguous()"
    for i, trans in enumerate(cg.need_trans_input):
        if trans:
            args_list[i] += ".mT"

    self.writeline(f"{buf_name} = {meta.name}({', '.join(args_list)})")
    self.add_import_once("from torch_npu._inductor import dvm")


class DvmGraphFusionPatch:
    _enabled = False
    _orig_generate_fallback_kernel = None
    _orig_post_grad_custom_post_pass = None

    @staticmethod
    def enable() -> None:
        if not DvmGraphFusionPatch._enabled:
            DvmGraphFusionPatch._orig_generate_fallback_kernel = (
                PythonWrapperCodegen.generate_fallback_kernel
            )
            DvmGraphFusionPatch._orig_post_grad_custom_post_pass = (
                inductor_config.post_grad_custom_post_pass
            )
            PythonWrapperCodegen.generate_fallback_kernel = (
                _dvm_generate_fallback_kernel
            )
            inductor_config.post_grad_custom_post_pass = dvm_graph_fusion
            DvmGraphFusionPatch._enabled = True

    @staticmethod
    def disable() -> None:
        if not DvmGraphFusionPatch._enabled:
            return
        PythonWrapperCodegen.generate_fallback_kernel = (
            DvmGraphFusionPatch._orig_generate_fallback_kernel
        )
        inductor_config.post_grad_custom_post_pass = (
            DvmGraphFusionPatch._orig_post_grad_custom_post_pass
        )
        DvmGraphFusionPatch._enabled = False

    def __enter__(self) -> "DvmGraphFusionPatch":
        DvmGraphFusionPatch.enable()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        DvmGraphFusionPatch.disable()
        return False
