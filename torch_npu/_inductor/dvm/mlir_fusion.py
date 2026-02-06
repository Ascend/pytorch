import os
import torch

import torch.utils._pytree as pytree
from torch._inductor.codegen.simd import code_hash, SIMDKernel
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.utils import get_fused_kernel_name
from torch._inductor import config
from torch._inductor.virtualized import V
from torch._inductor.scheduler import WhyNoFuse
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as anir_config
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.utils import (
    to_folder,
    get_num_call_functions,
)
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.mlir import (
    NpuMlirKernel,
    NpuMlirScheduling,
)
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch import (
    lowering as npu_lowering,
)
from .graph_build import DvmCodegenInterpreter
from .op_emitter import DVM_OP_REGISTRY, common_rule, DVM_SUPPORT_TYPE
from .decomp import patch_decomp
from .fx_test import generate_dvm_fx_case


dump_fx_test = False
uncont_policy = "fuse"
aten = torch.ops.aten
prims = torch.ops.prims
quantized = torch.ops.quantized
_quantized = torch.ops._quantized


anir_config.GENERATE_LIST = [
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
    aten.full,
    aten.relu,
    aten.where,
    aten.scalar_tensor,
    # aten.clone,
    # aten.reshape,
    # aten.copy_,
    # aten.copy,
]


def _codegen_dvm_kernel(self, Name=None):
    def is_node_dvm_supported(node):
        if node.op in ("call_function", "placeholder"):
            meta = node.meta["val"]
            if isinstance(meta, torch._subclasses.FakeTensor):
                return meta.dtype in DVM_SUPPORT_TYPE
        return True

    if all(is_node_dvm_supported(node) for node in self._gm.graph.nodes):
        self.dvm_codegen = DvmCodegenInterpreter(self._gm, ktype="vector")
        self.dvm_codegen.run()
        return self.dvm_codegen.code.getvalue()
    else:
        self.dvm_codegen = None
        return self._gm.print_readable(print_output=False)


def _define_dvm_kernel(self, src_code, mlir_kernel, traced_graph, mode=None):
    kernel_key = (src_code, tuple(mlir_kernel.non_contiguous_indices))
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
            out_indices = mlir_kernel.non_contiguous_indices.get("outputs")
            num_inputs = len(mlir_kernel.dvm_codegen.cont_flag_input)
            contiguity_flags = mlir_kernel.dvm_codegen.cont_flag_input + [
                i not in out_indices
                for i in range(num_inputs, num_inputs + mlir_kernel.num_outputs)
            ]
            kernel_meta = {
                "kernel_name": fused_kernel_name,
                "kernel_fullname": kernel_name,
                "contiguity_flags": contiguity_flags,
            }
            code = mlir_kernel.dvm_codegen.code
            code.splice(
                f"""
                k.set_kernel_info(
                    {kernel_meta.get('kernel_name')!r},  # kernel_name
                    {kernel_meta.get('kernel_fullname')!r},  # kernel_fullname
                    {kernel_meta.get('contiguity_flags')},  # contiguity_flags
                )
                """,
                strip=True,
            )
            func_name = kernel_name + "_build"
            func_code = code.getvalue().replace(
                mlir_kernel.dvm_codegen.KERNEL_NAME_PLACEHOLDER, func_name
            )
            compile_wrapper.writeline(func_name)
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

    def _fallback_node_due_to_unsupported_type(
        node: torch.fx.Node, allow_cpu_inputs=True
    ):
        if "val" in node.meta:
            for meta in pytree.tree_leaves(node.meta["val"]):
                if not isinstance(meta, torch._subclasses.FakeTensor):
                    continue

                if meta.is_cpu:
                    return True

        if node.target in DVM_OP_REGISTRY:
            _, rule = DVM_OP_REGISTRY.get(node.target)
            return not rule(node)
        return not common_rule(node)

    import torch._inductor.graph as inductor_graph
    import torch._inductor.lowering as inductor_lowering
    import torch._inductor.pattern_matcher as pattern_matcher
    import torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.lowering as npu_lowering_mod

    inductor_lowering.fallback_node_due_to_unsupported_type = (
        _fallback_node_due_to_unsupported_type
    )
    pattern_matcher.fallback_node_due_to_unsupported_type = (
        _fallback_node_due_to_unsupported_type
    )
    npu_lowering_mod.fallback_node_due_to_unsupported_type = (
        _fallback_node_due_to_unsupported_type
    )
    inductor_graph.fallback_node_due_to_unsupported_type = (
        _fallback_node_due_to_unsupported_type
    )


def _patch_sum_lowering():
    from torch._inductor import lowering as inductor_lowering_local
    from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.lowering import (
        is_integer_dtype,
        is_boolean_dtype,
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
        if axis and axis[-1] < 0:
            offset = len(x.get_size())
            axis = [ax + offset for ax in axis]
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


class DvmMlirFusionPatch:
    _enabled = False

    @staticmethod
    def enable() -> None:
        if DvmMlirFusionPatch._enabled:
            return
        config.allow_buffer_reuse = False
        patch_decomp()
        _patch_lowering_type_checks()
        _patch_sum_lowering()
        NpuMlirKernel.codegen_kernel = _codegen_dvm_kernel
        NpuMlirScheduling.can_fuse_horizontal = _dvm_can_fuse_horizontal
        NpuMlirScheduling.can_fuse_vertical = _dvm_can_fuse_vertical
        NpuMlirScheduling.define_kernel = _define_dvm_kernel
        DvmMlirFusionPatch._enabled = True


DvmMlirFusionPatch.enable()
