import functools
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

    # ---------------------------------------------------------------------
    # FA v3 specific check: TND + keep_prob fallback / not supported
    # When keep_prob is in the range [0, 1) (i.e., dropout is enabled),
    # the operator should not be captured by ACLGraph. According to the spec,
    # we raise an explicit error if the user invokes the atomic interface
    # directly, otherwise we treat it as unsafe for the graph partition.
    # ---------------------------------------------------------------------
    if target in (
            torch.ops.npu.npu_fusion_attention_v3.default,
            torch.ops.npu.npu_fusion_attention_grad_v3.default,
    ):
        normalized = normalize_function(
            target, fx_node.args, fx_node.kwargs, normalize_to_only_use_kwargs=True
        )
        if normalized is not None:
            _, kwargs = normalized
            keep_prob = kwargs.get("keep_prob")
            input_layout = kwargs.get("input_layout")
            if (
                keep_prob is not None
                and float(keep_prob) < 1
                and input_layout is not None
                and str(input_layout).upper() == "TND"
            ):
                return True

    return False


def patch_fx_node_is_input_dependent_cudagraph_unsafe():

    from torch._inductor import utils as inductor_utils
    inductor_utils._fx_node_is_input_dependent_cudagraph_unsafe = _fx_node_is_input_dependent_cudagraph_unsafe
    from torch._inductor import lowering as inductor_lowering
    inductor_lowering._fx_node_is_input_dependent_cudagraph_unsafe = _fx_node_is_input_dependent_cudagraph_unsafe


def disable_foreach():
    from torch._inductor.scheduler import Scheduler

    def create_foreach_nodes(self):
        return

    Scheduler.create_foreach_nodes = create_foreach_nodes
