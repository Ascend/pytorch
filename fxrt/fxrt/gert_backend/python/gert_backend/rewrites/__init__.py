"""Converter-only overlays on top of fx_backend, organized by what they extend.

- ``op_map``: extra ``torch target -> Op`` rows the converter lowers but fx_backend
  does not register;
- ``arg``:    arg-mapping hooks (shape-list concretize for expand/view/repeat/
  as_strided; offsets/size vectors for slice/getitem; GE-order pad);
- ``ops``:    ops-mapping hooks that pick the IR op by input form (slice/getitem to
              the slice ops; mul/sub/where forms Path C cannot express demoted to
              custom_call);
- ``output``: output-mapping hooks that build the IR form when the default lowering
              is not enough (strided slice as a cat of length-1 slices, select as
              slice + view, reduce_sum keepdim as reduce + view, mutation-only
              custom_call synthesized from the mutated input);
- ``as_strided``: a graph rewrite turning known as_strided layouts into permute/expand,
              self-registered into the framework list at import time;
- ``argmax``: a graph rewrite turning dim=None argmax into flatten + argmax(-1),
              self-registered into the framework list at import time;
- ``op_support``: keeps a call only when its logical name is in the converter's
              supported set (catalog rows plus dedicated lowerings) and demotes the
              rest to custom_call;
- ``npu_fusion_attention``: FlashAttentionScore overlay. aclnn writes the softmax
              tensors before attention_out, so the output hook rebuilds the IR tuple
              and the getitem hook remaps torch slots. Mask, varlen, dropout and the
              host int outputs stay on custom_call.

Each module owns a ``register()`` wiring its part into fx_backend's tables (the graph
rewrite registers at import instead); the converter backend context calls the four
table registers before each conversion, and the overlays stay for the process.

Shared analysis lives in ``_common`` (example values, shapes, concretize helpers,
offsets/size vectors) and ``_indexing`` (slice / getitem / select parsing and
emission). Graph rewrites register through ``register_converter_graph_rewrite`` and
run via ``apply_converter_graph_rewrites``.

To add hooks for a new op: put each hook in its kind module and add one line in that
module's register(). To add an op mapping: one row in ``op_map.OP_MAP_EXTRA``.
"""

# Re-exported so callers use ``rewrites.arg.register()`` etc.; importing as_strided and
# argmax also self-registers their graph rewrites.
# pylint: disable=unused-import
from fxrt.gert_backend.rewrites import arg, argmax, as_strided, op_map, op_support, ops, output, tuple_views
from fxrt.gert_backend.rewrites import newaxis, npu_fusion_attention, safe_softmax, topk
from fxrt.gert_backend.rewrites import empty_strided
# pylint: enable=unused-import
from fxrt.gert_backend.rewrites._framework import (  # pylint: disable=unused-import
    apply_converter_graph_rewrites,
    register_converter_graph_rewrite,
)
