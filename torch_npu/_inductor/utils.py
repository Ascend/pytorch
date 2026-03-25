import functools
from typing import Optional

import torch
import torch_npu


# Not good implementation, but no other way
def get_current_raw_stream(device):
    return torch.npu.current_stream(device).npu_stream


def patch_is_same_tensor():
    from torch._subclasses.fake_tensor import FakeTensor

    def is_same_tensor(data: torch.Tensor, value: torch.Tensor):
        if isinstance(data, FakeTensor) or isinstance(value, FakeTensor):
            return False
        return (
            not data.is_mkldnn
            and data.size() == value.size()
            and data.stride() == value.stride()
            and data.dtype == value.dtype
            and data.device == value.device
            and data.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            and data.storage_offset() == value.storage_offset()
        )
    
    from torch._inductor import utils, graph
    utils.is_same_tensor = is_same_tensor
    # We need to do extra-patch because of code like `from xxx import is_same_tensor`
    graph.is_same_tensor = is_same_tensor


def patch_is_gpu():
    from torch._inductor.utils import GPU_TYPES

    GPU_TYPES.append('npu')

    def _return_false(device_interface):
        return False

    torch._inductor.scheduler.device_need_guard = _return_false


def patch_has_triton():
    from torch.utils._triton import has_triton_package

    @functools.lru_cache(None)
    def has_triton() -> bool:
        if not has_triton_package():
            return False

        from torch._dynamo.device_interface import get_interface_for_device

        def cuda_extra_check(device_interface):
            return True

        def cpu_extra_check(device_interface):
            import triton.backends

            return "cpu" in triton.backends.backends

        def _return_true(device_interface):
            return True

        triton_supported_devices = {
            "cuda": cuda_extra_check,
            "xpu": _return_true,
            "cpu": cpu_extra_check,
            "npu": _return_true
        }

        def is_device_compatible_with_triton():
            for device, extra_check in triton_supported_devices.items():
                device_interface = get_interface_for_device(device)
                if device_interface.is_available() and extra_check(device_interface):
                    return True
            return False

        return is_device_compatible_with_triton()

    torch.utils._triton.has_triton = has_triton
    torch._inductor.scheduler.has_triton = has_triton


def _fx_node_is_input_dependent_cudagraph_unsafe(fx_node: torch.fx.Node) -> bool:
    """
    Check if an FX node is cudagraph-unsafe based on its input arguments.

    Some ops are only cudagraph-unsafe depending on their inputs (e.g., index_put
    with boolean indices triggers .nonzero() during capture, but integer indices
    are safe).
    """
    from torch.fx.operator_schemas import normalize_function

    target = fx_node.target
    if not isinstance(target, torch._ops.OpOverload):
        return False

    # index_put with boolean indices triggers .nonzero() during capture
    if target in (
        torch.ops.aten.index_put.default,
        torch.ops.aten.index_put_.default,
        torch.ops.aten._unsafe_index_put.default,
    ):
        normalized = normalize_function(
            target, fx_node.args, fx_node.kwargs, normalize_to_only_use_kwargs=True
        )
        if normalized is not None:
            _, kwargs = normalized
            indices = kwargs["indices"]
            for idx in indices:
                if idx is not None and idx.meta["val"].dtype in (
                    torch.bool,
                    torch.uint8,
                ):
                    return True

    return False


def patch_get_first_incompatible_cudagraph_node():
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
    from torch.utils._ordered_set import OrderedSet

    def get_first_incompatible_cudagraph_node(
        gm: torch.fx.GraphModule,
    ) -> Optional[torch.fx.Node]:
        forbidden_set = OrderedSet(
            [
                "aten._fused_moving_avg_obs_fq_helper.default",
                "aten._fused_moving_avg_obs_fq_helper_functional.default",
                "fbgemm.dense_to_jagged.default",
                "fbgemm.jagged_to_padded_dense.default",
                "run_and_save_rng_state",
                "run_with_rng_state",
                "aten._local_scalar_dense",
                # Technically, it's not necessary to ban this, because an
                # assert_scalar with constant arguments can be validly run
                # with CUDA graphs, but the operator is also pointless with
                # constant arguments, so might as well ban
                "aten._assert_scalar",
            ]
        )
        if torch.are_deterministic_algorithms_enabled():
            forbidden_set.update(
                (
                    "aten._unsafe_index_put.default",
                    "aten._unsafe_masked_index_put_accumulate.default",
                    "aten.index_put.default",
                    "aten.index_put_.default",
                    "aten.scatter.src",
                    "aten.scatter.reduce",
                    "aten.scatter.value_reduce",
                    "aten.scatter_add_",
                    "aten.scatter_add.default",
                    "aten.scatter_reduce.two",
                    "aten.scatter_reduce_.two",
                    "aten.scatter_reduce.two_out",
                )
            )

        for node in gm.graph.nodes:
            if _fx_node_is_input_dependent_cudagraph_unsafe(node):
                return node
            if str(node.target) in forbidden_set:
                return node

            if (
                not torch._inductor.config.graph_partition
                and isinstance(node.target, torch._ops.OpOverload)
                and torch._C.Tag.cudagraph_unsafe in node.target.tags  # type: ignore[attr-defined]
            ):
                # skip cudagraph if a cudagraph_unsafe op is detected.
                # graph_partition helps by splitting on this cudagraph_unsafe
                # op and cudagraphifying the subgraphs.
                return node

            if (val := node.meta.get("val")) is not None and free_unbacked_symbols(val):
                return node

        return None

    from torch._inductor import utils as inductor_utils
    inductor_utils.get_first_incompatible_cudagraph_node = get_first_incompatible_cudagraph_node

    from torch._inductor import compile_fx
    compile_fx.get_first_incompatible_cudagraph_node = get_first_incompatible_cudagraph_node

    from torch._dynamo.backends import cudagraphs
    cudagraphs.get_first_incompatible_cudagraph_node = get_first_incompatible_cudagraph_node

    from torch_npu.utils import _graph_tree
    _graph_tree.get_first_incompatible_cudagraph_node = get_first_incompatible_cudagraph_node


def disable_foreach():
    from torch._inductor.scheduler import Scheduler

    def create_foreach_nodes(self):
        return

    Scheduler.create_foreach_nodes = create_foreach_nodes
