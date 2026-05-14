from dataclasses import dataclass

import torch
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._subclasses import FakeTensor
from torch.fx import GraphModule, Node

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


def expand_dvm_mm_to_explicit_transpose_for_inductor(gm: torch.fx.GraphModule) -> bool:
    """Rewrite ``aten.mm`` + ``dvm_trans_*`` into explicit 2D ``permute(1,0)`` + ``mm``.

    MFusion IR roundtrip can emit a single ``torch.aten.mm`` with ``dvm_trans_a`` / ``dvm_trans_b``
    (fused ``aclnn.mm`` transpose flags). FakeTensor propagation uses a stub so ``meta`` is
    consistent, but Inductor lowering calls ``mm_args`` on **storage** shapes and fails
    ``guard_equals`` on the inner dimension. Inserting ``aten.permute`` (last-two dims swap for 2D)
    restores layouts Inductor expects. We use ``permute`` rather than ``transpose.int`` because
    some NPU Inductor builds register both a fallback and a decomposition for ``transpose.int``,
    which triggers ``AssertionError: both a fallback and a decomp for same op``.
    """
    g = gm.graph
    changed = False
    swap_2d = (1, 0)
    for n in list(g.nodes):
        if n.op != "call_function" or n.target != aten.mm.default:
            continue
        ta = bool(n.meta.get("dvm_trans_a", False))
        tb = bool(n.meta.get("dvm_trans_b", False))
        if not (ta or tb):
            continue
        if len(n.args) < 2:
            continue
        lhs, rhs = n.args[0], n.args[1]
        if not isinstance(lhs, Node) or not isinstance(rhs, Node):
            continue
        lv = lhs.meta.get("val")
        rv = rhs.meta.get("val")
        if not isinstance(lv, torch.Tensor) or not isinstance(rv, torch.Tensor):
            continue
        if lv.dim() != 2 or rv.dim() != 2:
            continue

        new_lhs, new_rhs = lhs, rhs
        with g.inserting_before(n):
            if ta:
                new_lhs = g.call_function(aten.permute.default, (lhs, list(swap_2d)))
            if tb:
                new_rhs = g.call_function(aten.permute.default, (rhs, list(swap_2d)))
        n.args = (new_lhs, new_rhs)
        n.meta.pop("dvm_trans_a", None)
        n.meta.pop("dvm_trans_b", None)
        changed = True

    if changed:
        g.lint()
        gm.recompile()
    return changed


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
        elif getattr(lhs, "meta", None) and lhs.meta.get("trans"):
            node.meta["trans_a"] = True
        if rhs.op == "placeholder" and _is_last2_transpose_tensor(rhs.meta["val"]):
            node.meta["trans_b"] = True
            rhs.meta["trans"] = True
        elif getattr(rhs, "meta", None) and rhs.meta.get("trans"):
            node.meta["trans_b"] = True

        # Flags from MLIR (mfuse.aclnn.mm trans_x1/trans_x2), exported as dvm_trans_* on torch.aten.mm.
        # Only set trans_a/trans_b on the mm node for k.matmul(...). Do NOT set lhs/rhs placeholder
        # meta["trans"]: that path means "stride is a transposed view" and mix-kernel load() uses
        # val.mT.shape — but dvm_trans_* means storage layout is unchanged; transpose is applied
        # inside matmul only. Marking placeholder trans would double-apply (wrong load + .mT) and
        # can crash the CANN/DVM backend.
        if node.meta.get("dvm_trans_a"):
            node.meta["trans_a"] = True
        if node.meta.get("dvm_trans_b"):
            node.meta["trans_b"] = True

    return flag


def make_cast_node(g, src: Node, target_dtype: torch.dtype) -> Node:
    cast = g.call_function(
        prims.convert_element_type.default,
        args=(src, target_dtype),
    )

    cast.meta["val"] = src.meta["val"].to(dtype=target_dtype)
    return cast


def decompose_k1_matmul_to_mul(gm: GraphModule) -> GraphModule:
    g = gm.graph
    changed = False

    for node in list(g.nodes):
        if node.op != "call_function":
            continue
        if node.target not in (aten.mm.default, aten.bmm.default):
            continue

        lhs, rhs = node.args[:2]

        if not isinstance(lhs, Node) or not isinstance(rhs, Node):
            continue

        lhs_val = lhs.meta.get("val", None)
        rhs_val = rhs.meta.get("val", None)
        if not isinstance(lhs_val, FakeTensor) or not isinstance(rhs_val, FakeTensor):
            continue

        if _is_last2_transpose_tensor(lhs_val):
            continue
        if _is_last2_transpose_tensor(rhs_val):
            continue

        lhs_k = lhs_val.shape[-1]
        rhs_k = rhs_val.shape[-2]

        if isinstance(lhs_k, torch.SymInt) or isinstance(rhs_k, torch.SymInt):
            continue
        if lhs_k != 1 or rhs_k != 1:
            continue

        with g.inserting_before(node):
            mul_node = g.call_function(aten.mul.Tensor, args=(lhs, rhs))
            mul_node.meta["val"] = node.meta["val"]

        node.replace_all_uses_with(mul_node)
        g.erase_node(node)
        changed = True

    if changed:
        g.lint()
        gm.recompile()
    return gm


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
