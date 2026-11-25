"""Central registry of Python overlay rows for the converter op catalog.

Internal bring-up tool, not a user-facing API. Each dict in ``OVERLAY_SPECS`` describes one
catalog row -- how a logical op (an ``ops.list`` name) lowers onto a GE opp type -- and
``compile_executor`` registers the whole table before every convert. Trying out a new op is
therefore: append a row here, rerun the model. No rebuild, no manual registration call.

An overlay must name an existing dialect op and an existing GE opp type; it cannot invent
either. Registration is process-wide, a row replaces the same-named row, and rows live until
process exit. Permanent support belongs in a builtin row (``FX_CATALOG_OP`` in
``src/register/op_catalog_table.cc``) plus an ST, not here.

Spec keys: ``logical_name``* / ``opp_type``* (required), ``arity`` [2], ``lowering``
["aicore"|"dual_stage"], ``cni_io`` ["indexed0"|"packed_x"], ``recipe``
["tensor_io"|"as_strided"], ``attrs`` [{"name", "src", "literal", "text", "required",
"derived"}, ...] where ``src`` is the GE attr type (``bool``/``int``/``float``/``str``/
``list_int``) and ``derived`` marks an attr the lowering fills (``"input_count"`` =
collected tensor input count, ``"output_dtype"`` = GE dtype of the IR output).
``capture_axis`` pairs with ``axis_attr`` naming the GE attr the axis is published to
(the name must also appear in ``attrs``). ``capture_skip_bool`` drops an IR bool input
(dual-stage rejects undeclared bools). ``opp_output_dtypes`` is a CSV forcing a GE dtype
per output (empty cell = infer, e.g. ``",int32"``). Plus the remaining shape/capture/
tuning knobs from ``CatalogOpSpec``
(``fxrt/gert_backend/include/register/op_catalog_register.h``); the builtin rows in
``src/register/op_catalog_table.cc`` show each one in use.
"""

from __future__ import annotations

# The overlay table. Append one dict per op; compile_executor applies the whole table on
# every convert.
#
# Row-writing hints:
# - "aicore" reuses a CANN prebuilt kernel binary and wants an exact input count;
#   "dual_stage" goes through aclnn InferShape/Prepare/Launch.
# - aten schemas with a trailing default scalar (add/sub alpha) reach the converter with an
#   extra input; pick "dual_stage" there (like the builtin add row).
OVERLAY_SPECS = [
    # Example row: dialect op "relu" (fed by aten.relu) onto GE opp "Relu". Unary with no
    # extra aten args, so the exact-count aicore path fits -- mirrors the builtin "neg" row.
    {
        "logical_name": "relu",
        "opp_type": "Relu",
        "arity": 1,
        "lowering": "aicore",
    },
]


def register_overlay_ops() -> None:
    """Register every row in ``OVERLAY_SPECS``; called by compile_executor before each convert.

    A row replaces the same-named row, so re-applying on later converts is a no-op in effect.
    Raises RuntimeError naming the first invalid row (missing required key, unknown
    lowering/cni_io/recipe/attr src).
    """
    from fxrt.gert_backend import _fxrt_fx_converter  # pylint: disable=import-outside-toplevel

    for spec in OVERLAY_SPECS:
        _fxrt_fx_converter.register_catalog_op(spec)
