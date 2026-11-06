"""Lower index -> flattening reshape -> index using composed flattened indices."""

import sympy
import torch
from torch._inductor import ir, lowering
from torch._inductor.lowering import register_lowering
from torch._inductor.virtualized import V


def _register_npu_inductor_index_reshape_index():
    from ..fx_passes.ascend_custom_passes.index_reshape_index import TARGET

    @register_lowering(TARGET, type_promotion_kind=None)
    def index_reshape_index(input_tensor, first_indices, second_indices):
        input_shape = input_tensor.get_size()
        source_count = input_shape[0]
        block_size = sympy.prod(input_shape[1:-1])
        feature_size = input_shape[-1]
        count = first_indices.get_size()[0] * block_size
        input_tensor = ir.ExternKernel.realize_input(input_tensor)

        def normalize(indices, size):
            # The identity range stays virtual. index emits the usual wrapping
            # and bounds checks; unlike remainder it cannot hide invalid indices.
            identity = lowering.iota(
                size,
                start=0,
                step=1,
                dtype=torch.int64,
                device=indices.get_device(),
                requires_grad=False,
            )
            return lowering.index(identity, [indices])

        second_indices = normalize(second_indices, count)
        first_indices = normalize(first_indices, source_count)
        # Validate every first-stage index, even if the second stage never reads it.
        # Materialize this validation independently of second-stage index use.
        first_indices = ir.ExternKernel.realize_input(first_indices)
        V.graph.no_fuse_buffer_names.add(first_indices.get_name())
        quotient = lowering.div_mode(second_indices, block_size, rounding_mode="floor")
        selected = lowering.index(first_indices, [quotient])
        # Second-stage indices are normalized. Reuse the integer quotient to avoid
        # an external FloorMod kernel and its input materialization.
        whole_rows = lowering.lowerings[torch.ops.aten.mul.Scalar](quotient, block_size)
        offset = lowering.lowerings[torch.ops.aten.sub.Tensor](
            second_indices, whole_rows
        )
        multiplied = lowering.lowerings[torch.ops.aten.mul.Scalar](selected, block_size)
        combined = lowering.lowerings[torch.ops.aten.add.Tensor](multiplied, offset)
        # Cast only after negative wrapping and bounds validation in INT64.
        # The normalized flattened-index range fits in INT32 under this proof.
        flattened_rows = source_count * block_size
        # Static proof only, including for direct internal-op callers. Do not
        # specialize symbolic sizes. The largest valid index is flattened_rows - 1.
        if isinstance(flattened_rows, (int, sympy.Integer)) and flattened_rows <= 2**31:
            combined = lowering.to_dtype(combined, torch.int32)
        combined = ir.ExternKernel.realize_input(combined)
        flattened = lowering.view(
            input_tensor, [source_count * block_size, feature_size]
        )
        return lowering.fallback_handler(
            torch.ops.aten.index_select.default, add_to_fallback_set=False
        )(flattened, 0, combined)
