import os

import torch
from torch._inductor import config
from torch._inductor.fx_passes.control_dependencies import control_deps
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.simd import code_hash, SIMDKernel
from torch._inductor.scheduler import WhyNoFuse
from torch._inductor.utils import get_fused_kernel_name
from torch._inductor.virtualized import V
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as anir_config
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.mlir import (
    NpuMlirKernel,
    NpuMlirScheduling,
)
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch import (
    lowering as npu_lowering,
)
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.utils import (
    get_num_call_functions,
    to_folder,
)

from .decomp import patch_decomp
from .fx_test import generate_dvm_fx_case
from .graph_build import DvmCodegenInterpreter
from .op_emitter import common_rule, DVM_OP_REGISTRY, DVM_SUPPORT_TYPE, _extra_int_types


dump_fx_test = os.environ.get("INDUCTOR_DVM_DUMP_FX_TEST", "0") == "1"
view_fusion_level = int(os.environ.get("INDUCTOR_DVM_VIEW_FUSION_LEVEL", "1"))
disable_post_reduce_fusion = (
    os.environ.get("INDUCTOR_DVM_DISABLE_POST_REDUCE_FUSION", "0") == "1"
)
aten = torch.ops.aten
prims = torch.ops.prims
quantized = torch.ops.quantized
_quantized = torch.ops._quantized


anir_config.GENERATE_LIST = [
    control_deps,
    aten._assert_scalar,
    aten.mul,
    aten.add,
    aten.sub,
    aten.div,
    aten.clamp_min,
    aten.clamp_max,
    aten.maximum,
    aten.minimum,
    aten.abs,
    aten.reciprocal,
    aten.log,
    aten.exp,
    aten.pow,
    aten.sqrt,
    aten.rsqrt,
    aten.neg,
    aten.lt,
    aten.le,
    aten.gt,
    aten.ge,
    aten.eq,
    aten.ne,
    aten.bitwise_and,
    aten.bitwise_or,
    aten.bitwise_not,
    aten.where,
    prims.convert_element_type,
    torch.ops.npu.npu_dtype_cast,
    torch.ops.npu.npu_dtype_cast_backward,
    torch.ops.npu._npu_dtype_cast,
    torch.ops.npu._npu_dtype_cast_backward,
    aten.expand,
    aten.var_mean,
    aten.sum,
    aten.mean,
    aten.amax,
    aten.amin,
    aten.full,
    aten.relu,
    aten.where,
    aten.scalar_tensor,
    aten.unsqueeze,
    aten.squeeze,
    aten.reshape,
    # aten.copy,
    # aten.copy_,
    # aten.clone,
]

def _is_node_supported_by_dvm_rule(node, allow_common_rule=False):
    if node.target in DVM_OP_REGISTRY:
        _, rule = DVM_OP_REGISTRY.get(node.target)
        return rule(node)
    return allow_common_rule and common_rule(node)


def _codegen_dvm_kernel(self, Name=None):
    def is_node_dvm_supported(node):
        if node.op == "placeholder":
            meta = node.meta["val"]
            if isinstance(meta, torch._subclasses.FakeTensor):
                return meta.dtype in [*DVM_SUPPORT_TYPE, *_extra_int_types]
        if node.op == "call_function":
            return _is_node_supported_by_dvm_rule(node)
        return True

    if all(is_node_dvm_supported(node) for node in self._gm.graph.nodes):
        self.dvm_codegen = DvmCodegenInterpreter(
            self._gm, ktype="vector", view_fusion_level=view_fusion_level
        )
        self.dvm_codegen.run()
        return self.dvm_codegen.code.getvalue()
    else:
        self.dvm_codegen = None
        return self._gm.print_readable(print_output=False)


def _kernel_layout_key(mlir_kernel):
    non_contiguous_key = tuple(
        (name, tuple(indices))
        for name, indices in sorted(mlir_kernel.non_contiguous_indices.items())
    )
    dvm_codegen = getattr(mlir_kernel, "dvm_codegen", None)
    cont_flag_input = tuple(getattr(dvm_codegen, "cont_flag_input", ()))
    return non_contiguous_key, cont_flag_input


def _define_dvm_kernel(self, src_code, mlir_kernel, traced_graph, mode=None):
    kernel_key = (src_code, _kernel_layout_key(mlir_kernel))
    wrapper = V.graph.wrapper_code

    if kernel_key in wrapper.src_to_kernel:
        kernel_name = wrapper.src_to_kernel[kernel_key]
    else:
        fused_kernel_name = "dvm_" + get_fused_kernel_name(
            mlir_kernel._snodes, config.triton.descriptive_names
        )
        kernel_suffix = V.graph.wrapper_code.next_kernel_suffix()
        kernel_name = "_".join([fused_kernel_name, kernel_suffix])

        traced_graph_hash = code_hash(
            traced_graph.print_readable(print_output=False) + kernel_name
        )

        kernel_info = {}

        wrapper.src_to_kernel[kernel_key] = kernel_name
        current_device = V.graph.get_current_device_or_throw()

        compile_wrapper = IndentedBuffer()
        if (
            mlir_kernel.dvm_codegen is None
            or kernel_name in anir_config.force_fallback_kernel_names
        ):
            num_call_functions = get_num_call_functions(mlir_kernel._gm)
            kernel_meta = {
                "device_str": current_device.type,
                "device_index": current_device.index,
                "num_outputs": mlir_kernel.num_outputs,
                "non_contiguous_indices": mlir_kernel.non_contiguous_indices,
                "dynamic": mlir_kernel._is_dynamic,
                "mutated_indices": mlir_kernel.mutated_indices,
                "traced_graph_cache": anir_config.traced_graph_cache,
                "traced_graph_hash": traced_graph_hash,
                "num_call_functions": num_call_functions,
                **kernel_info,
            }
            compile_wrapper.writeline(
                f"async_compile.import_fx({kernel_name!r}, kernel_meta={kernel_meta})"
            )
            metadata_comment = (
                f'"""\n{mlir_kernel._gm.print_readable(print_output=False)}\n"""'
            )
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
            dump_path = os.path.join(
                os.getenv("TORCHINDUCTOR_CACHE_DIR"),
                anir_config.traced_graph_cache,
                str(current_device.index),
                traced_graph_hash,
            )
            if not os.path.exists(dump_path):
                os.makedirs(dump_path, exist_ok=True)
                to_folder(
                    mlir_kernel._gm,
                    dump_path,
                    graph_hash=traced_graph_hash,
                    module_name=traced_graph_hash,
                )
        else:
            wrapper.add_import_once("from torch_npu._inductor import dvm")
            if dump_fx_test:
                generate_dvm_fx_case(mlir_kernel._gm, fusion_type="mlir")
            kernel_meta = {
                "kernel_name": fused_kernel_name,
                "kernel_fullname": kernel_name,
            }
            code = mlir_kernel.dvm_codegen.code
            code.splice(
                f"""
                k.set_kernel_info(
                    {kernel_meta.get("kernel_name")!r},  # kernel_name
                    {kernel_meta.get("kernel_fullname")!r},  # kernel_fullname
                )
                """,
                strip=True,
            )
            func_name = kernel_name + "_build"
            func_code = code.getvalue().replace(
                mlir_kernel.dvm_codegen.KERNEL_NAME_PLACEHOLDER, func_name
            )
            compile_wrapper.writeline(func_name)

            if anir_config.online_acc_comp:
                dump_path = os.path.join(
                    os.getenv("TORCHINDUCTOR_CACHE_DIR"),
                    anir_config.traced_graph_cache,
                    str(current_device.index),
                    traced_graph_hash,
                )
                if not os.path.exists(dump_path):
                    os.makedirs(dump_path, exist_ok=True)
                    to_folder(
                        mlir_kernel._gm,
                        dump_path,
                        graph_hash=traced_graph_hash,
                        module_name=traced_graph_hash,
                    )
                compile_wrapper.writeline(
                    f"{kernel_name}._acc_meta = {{"
                    f"'traced_graph_hash': {traced_graph_hash!r}, "
                    f"'traced_graph_cache': {anir_config.traced_graph_cache!r}, "
                    f"'device_index': {current_device.index}, "
                    f"'num_outputs': {mlir_kernel.num_outputs}}}"
                )

            wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), func_code)

    return kernel_name


def _dvm_can_fuse_vertical(self, node1, node2):
    _, (numel1, rnumel1) = node1.group
    _, (numel2, rnumel2) = node2.group
    why = WhyNoFuse(node1, node2)

    if node1.is_reduction():
        return False

    if not node2.is_reduction():
        return numel1 == numel2 and rnumel1 == rnumel2
    else:
        if numel1 == numel2 * rnumel2:
            if not all(
                SIMDKernel.is_compatible((numel2, rnumel2), n.get_ranges())
                for n in node1.get_nodes()
            ):
                why("nodes numel/rnumel incompatibility")
                return False

            return True
        if numel1 != numel2:
            why("nodes numel incompatibility")
        return numel1 == numel2


def _dvm_can_fuse_horizontal(self, node1, node2):
    return False


def _patch_lowering_type_checks():
    import torch._inductor.graph as inductor_graph
    import torch._inductor.lowering as inductor_lowering
    import torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.lowering as npu_lowering_mod
    import torch_npu._inductor.graph as npu_graph_mod

    fallback_node_due_to_unsupported_type = (
        inductor_lowering.fallback_node_due_to_unsupported_type
    )

    def _fallback_node_due_to_unsupported_type(
        node: torch.fx.Node, allow_cpu_inputs=True
    ):
        if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs):
            return True

        if node.target is torch.ops.higher_order.triton_kernel_wrapper_functional:
            return False
        if node.target is torch.ops.higher_order.triton_kernel_wrapper_mutation:
            return False
        if node.target is aten.lift_fresh_copy.default:
            return False

        return not _is_node_supported_by_dvm_rule(node, allow_common_rule=True)

    inductor_lowering.fallback_node_due_to_unsupported_type = (
        _fallback_node_due_to_unsupported_type
    )
    npu_lowering_mod.fallback_node_due_to_unsupported_type = (
        _fallback_node_due_to_unsupported_type
    )
    inductor_graph.fallback_node_due_to_unsupported_type = (
        _fallback_node_due_to_unsupported_type
    )
    npu_graph_mod.fallback_node_due_to_unsupported_type = (
        _fallback_node_due_to_unsupported_type
    )


def _patch_lowering():
    from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.lowering import (
        is_boolean_dtype,
        is_integer_dtype,
        make_reduction,
        to_dtype,
    )

    def get_overloads(aten_fn):
        if not isinstance(aten_fn, (list, tuple)):
            aten_fn = [aten_fn]
        else:
            aten_fn = list(aten_fn)

        for fn in list(aten_fn):
            if isinstance(fn, torch._ops.OpOverloadPacket):
                for overload in fn.overloads():
                    other_fn = getattr(fn, overload)
                    aten_fn.append(other_fn)

        return aten_fn

    def sum_(x, axis=None, keepdims=False, *, dtype=None):
        if axis and any(ax < 0 for ax in axis):
            offset = len(x.get_size())
            axis = [ax + offset if ax < 0 else ax for ax in axis]
        if (
            is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
        ) and dtype is None:
            dtype = torch.int64

        out_dtype = x.get_dtype() if dtype is None else dtype

        fn = make_reduction("sum", override_return_dtype=torch.float32)
        r = fn(x, axis, keepdims, dtype=torch.float32)

        if out_dtype != torch.float32:
            r = to_dtype(r, out_dtype)

        return r

    anir_config.disable_any_pbr = False
    ops = get_overloads([aten.sum, prims.sum])
    npu_lowering.register_lowering(ops)(sum_)
    npu_lowering.make_fallback(
        aten.matmul_backward.default,
        layout_constraint=None,
        warn=False,
        override_decomp=True,
    )
    npu_lowering.add_layout_constraint(aten.matmul_backward.default, None)


class DvmMlirFusionPatch:
    _enabled = False

    @staticmethod
    def enable() -> None:
        if DvmMlirFusionPatch._enabled:
            return
        from torch._dynamo import config as dynamo_config
        from torch._inductor import config as inductor_config

        dynamo_config.specialize_float = True  # enable float specialization until launch with scalar supported
        inductor_config.unroll_reductions_threshold = 1  # disable unroll reductions
        inductor_config.size_asserts = (
            False  # npu ops always return contiguous tensors which maybe different from meta outputs
        )
        inductor_config.allow_buffer_reuse = False
        inductor_config.comprehensive_padding = False
        patch_decomp()
        _patch_lowering_type_checks()
        _patch_lowering()
        NpuMlirKernel.codegen_kernel = _codegen_dvm_kernel
        NpuMlirScheduling.define_kernel = _define_dvm_kernel
        if disable_post_reduce_fusion:
            NpuMlirScheduling.can_fuse_horizontal = _dvm_can_fuse_horizontal
            NpuMlirScheduling.can_fuse_vertical = _dvm_can_fuse_vertical
        DvmMlirFusionPatch._enabled = True


DvmMlirFusionPatch.enable()
