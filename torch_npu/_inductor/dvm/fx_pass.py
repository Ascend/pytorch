from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node
from torch._subclasses import FakeTensor
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from .op_emitter import _is_last2_transpose_tensor


aten = torch.ops.aten
prims = torch.ops.prims


def need_fallback_gm(gm: torch.fx.GraphModule) -> bool:
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target not in (
            aten.reshape.default,
            aten.expand.default,
        ):
            return False
    return True


def annotate_mm_transpose_flags(gm: torch.fx.GraphModule):
    flag = False
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in [aten.mm.default, aten.bmm.default]:
            lhs = node.args[0]
            rhs = node.args[1]
            flag = True
        elif node.target is aten.addmm.default:
            add = node.args[0]
            lhs = node.args[1]
            rhs = node.args[2]
            if (
                add.meta["val"].dim() == 1
                and node.kwargs.get("beta", 1) == 1
                and node.kwargs.get("alpha", 1) == 1
            ):
                node.meta["use_bias"] = True
            flag = True
        else:
            continue
        node.meta["trans_a"] = False
        node.meta["trans_b"] = False
        if lhs.op == "placeholder" and _is_last2_transpose_tensor(lhs.meta["val"]):
            node.meta["trans_a"] = True
            lhs.meta["trans"] = True
        if rhs.op == "placeholder" and _is_last2_transpose_tensor(rhs.meta["val"]):
            node.meta["trans_b"] = True
            rhs.meta["trans"] = True

    return flag


def make_cast_node(g, src: Node, target_dtype: torch.dtype) -> Node:
    cast = g.call_function(
        prims.convert_element_type.default,
        args=(src, target_dtype),
    )

    cast.meta["val"] = src.meta["val"].to(dtype=target_dtype)
    return cast


def insert_sum_fp32_prepost_cast_prims(gm: GraphModule):
    g = gm.graph
    for node in g.nodes:
        if node.op != "call_function":
            continue
        if node.target not in [aten.sum.default, aten.sum.dim_IntList]:
            continue

        out_val = node.meta.get("val", None)
        if not isinstance(out_val, FakeTensor):
            continue
        orig_out_dtype = out_val.dtype

        if not node.args:
            continue
        x = node.args[0]
        if not isinstance(x, Node):
            continue

        in_val = x.meta.get("val", None)
        if not isinstance(in_val, FakeTensor):
            continue
        in_dtype = in_val.dtype

        if in_dtype == torch.float32:
            continue

        with g.inserting_before(node):
            x_fp32 = make_cast_node(g, x, torch.float32)

        new_args = list(node.args)
        new_args[0] = x_fp32
        node.args = tuple(new_args)

        if orig_out_dtype != torch.float32:
            with g.inserting_after(node):
                y = make_cast_node(g, node, orig_out_dtype)
            node.replace_all_uses_with(y)
            y.args = (node, orig_out_dtype)

    g.lint()
    gm.recompile()
    return gm


@dataclass(frozen=True)
class PromoteRule:
    pos: tuple[int, ...]
    kind: object = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT


PROMOTE_TYPE_OP = {
    # ========= Elementwise =========
    aten.add.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT),
    aten.sub.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT),
    aten.mul.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT),
    aten.div.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT),
    aten.pow.Tensor_Tensor: PromoteRule(
        (0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    aten.lt.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL),
    aten.le.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL),
    aten.gt.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL),
    aten.ge.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL),
    aten.eq.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL),
    aten.ne.Tensor: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL),
    aten.maximum.default: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT),
    aten.minimum.default: PromoteRule((0, 1), ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT),
    # ========= Where =========
    # where(cond, x, y) → promote x/y
    # aten.where.default: PromoteRule((1, 2), ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT),
}


def insert_promote_cast_by_pos_prims(gm: GraphModule) -> GraphModule:
    g = gm.graph

    for node in g.nodes:
        if node.op != "call_function":
            continue

        rule = PROMOTE_TYPE_OP.get(node.target, None)
        if rule is None:
            continue

        arg_vals = []
        arg_nodes = {}

        for idx in rule.pos:
            if idx >= len(node.args):
                continue
            arg = node.args[idx]
            if not isinstance(arg, Node):
                continue
            val = arg.meta.get("val", None)
            if not isinstance(val, FakeTensor):
                continue
            arg_vals.append(val)
            arg_nodes[idx] = arg

        if len(arg_vals) <= 1:
            continue

        dtypes = [v.dtype for v in arg_vals]
        if all(dt == dtypes[0] for dt in dtypes[1:]):
            continue

        compute_dtype, _ = elementwise_dtypes(
            *arg_vals,
            type_promotion_kind=rule.kind,
        )

        new_args = list(node.args)
        for idx, arg in arg_nodes.items():
            in_val = arg.meta.get("val", None)
            if in_val.dtype == compute_dtype:
                continue
            with g.inserting_before(node):
                cast = make_cast_node(g, arg, compute_dtype)
            new_args[idx] = cast

        node.args = tuple(new_args)

    g.lint()
    gm.recompile()
    return gm


def expand_to_reshape(gm: GraphModule) -> GraphModule:
    for node in gm.graph.find_nodes(op="call_function", target=aten.expand.default):
        x = node.args[0]
        in_val = x.meta.get("val")
        out_val = node.meta.get("val")
        if tuple(in_val.shape) == tuple(out_val.shape):
            node.target = aten.reshape.default
