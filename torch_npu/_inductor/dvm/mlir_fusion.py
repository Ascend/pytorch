import os
from typing import List, Optional, Union

import sympy
import torch
from torch._dynamo.utils import counters
from torch._inductor import config, ir, scheduler
from torch._inductor.codegen.common import IndentedBuffer, Kernel
from torch._inductor.codegen.simd import (
    code_hash, log, MultiKernel, SIMDKernel, SIMDKernelFeatures,
)
from torch._inductor.codegen.triton import SIMDScheduling, TritonKernel
from torch._inductor.scheduler import SchedulerNode, WhyNoFuse
from torch._inductor.utils import aggregate_origins, get_fused_kernel_name
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch_npu._inductor.lowering_common import (
    MLIR_OPERATOR_MAPPING,
    map_strings_to_operators as _map_strings_to_operators,
    merge_fx_graphs,
)
from .wrapper import NpuDvmWrapperCodeGen
from torch_npu._inductor.dvm.inductor_patch import lowering as npu_lowering

from . import config as dvm_config, is_ascend950
from .decomp import patch_decomp
from .fx_test import generate_dvm_fx_case
from .graph_build import DvmCodegenInterpreter
from .inductor_patch.fx_graph import (
    get_num_call_functions, modify_gm_for_acc_comp, to_folder, view_to_reshape,
)
from .op_emitter import (
    common_rule,
    DVM_OP_REGISTRY,
    DVM_SUPPORT_TYPE,
    _extra_int_types,
)
from .template import (
    can_fuse_dvm_epilogue,
    DvmTemplateBuffer,
    _disable_addmm_fusion_for_dvm,
    patch_dvm_matmul_template_fusion,
)
from .util import can_use_view_load

aten = torch.ops.aten
prims = torch.ops.prims



def _is_clone_supported_by_view_level(node):
    inp = node.args[0].meta["val"]
    return inp.is_contiguous() or can_use_view_load(
        inp.shape,
        inp.stride(),
        inp.dtype,
        view_fusion_level=dvm_config.view_fusion_level,
    )


def _is_node_supported_by_dvm_rule(node, allow_common_rule=False):
    if node.target == aten.cat.default and not dvm_config.enable_cat:
        return False
    if node.target == aten.permute.default and not dvm_config.enable_fusion_permute:
        return False
    if node.target == aten.clone.default and not _is_clone_supported_by_view_level(node):
        return False
    if node.target in (
        aten.mm.default,
        aten.bmm.default,
        aten.addmm.default,
        aten.baddbmm.default,
    ) and not dvm_config.enable_matmul_fusion:
        return False
    if node.target in DVM_OP_REGISTRY:
        _, rule = DVM_OP_REGISTRY.get(node.target)
        return rule(node)
    return allow_common_rule and common_rule(node)


def map_strings_to_operators(expr_str: str):
    return _map_strings_to_operators(expr_str, MLIR_OPERATOR_MAPPING)


class NpuTritonKernel(TritonKernel):
    @staticmethod
    def inductor_meta_common():
        return {}

    def call_kernel(self, call_args, name: str):
        wrapper = V.graph.wrapper_code
        for call_arg in call_args:
            if call_arg.startswith('_uwu_'):
                expression = map_strings_to_operators(call_arg)
                wrapper.writeline(f'{call_arg} = {expression}')
        if len(call_args) > 0:
            wrapper.generate_kernel_call(
                name,
                call_args,
            )


def refresh_input_meta_with_buffer_layout(node):
    val = node.meta.get('val')
    try:
        layout = getattr(V.graph.try_get_buffer(node.target), "layout", None)
    except Exception:
        layout = None
    if not torch.is_tensor(val) or layout is None:
        return val

    size, stride = tuple(layout.size), tuple(layout.stride)
    if any(not isinstance(x, (int, sympy.Integer)) for x in (*size, *stride)):
        return val

    if tuple(val.size()) != size or tuple(val.stride()) != stride:
        with V.graph.fake_mode:
            val = torch.empty_strided(
                size,
                stride,
                dtype=val.dtype,
                device=val.device,
                requires_grad=val.requires_grad,
            )
        node.meta['val'] = val
    return val


def create_fx_from_snodes_by_traced_graph(snodes: List[scheduler.SchedulerNode]):
    call_inputs = []
    for snode in snodes:
        snode.node.data.traced_graph.last_node.name = snode.node.get_name()
    if len(snodes) == 1:
        traced_graph = snodes[0].node.data.traced_graph
    else:
        traced_graph = merge_fx_graphs([snode.node.data.traced_graph for snode in snodes])
    inputs = []
    for node in traced_graph.graph.nodes:
        if node.op == 'placeholder':
            call_inputs.append(node.target)
            inputs.append(refresh_input_meta_with_buffer_layout(node))
    non_contiguous_indices = {}
    non_contiguous_indices["inputs"] = [i for i, inp in enumerate(inputs) if torch.is_tensor(inp) and not inp.is_contiguous()]
    num_inputs = len(call_inputs)
    call_outputs = []
    for snode in snodes:
        if snode.has_aliasing_or_mutation():
            for buf in snode.get_outputs():
                if len(buf.get_mutations()):
                    call_outputs.extend(buf.get_mutations())
                elif len(buf.get_aliases()):
                    call_outputs.append(buf.get_name())
        elif snode.node.get_name() not in (V.graph.removed_buffers | V.graph.inplaced_to_remove):
            call_outputs.append(snode.node.get_name())
    num_outputs = len(call_outputs)
    call_args = call_inputs + call_outputs
    outputs = traced_graph.last_node if isinstance(traced_graph.last_node, List) \
        else [traced_graph.last_node]
    outputs = [output for output in outputs if output.name not in (V.graph.removed_buffers | V.graph.inplaced_to_remove)]
    traced_graph.graph.output(tuple(outputs))
    traced_graph.graph.lint()
    orig_module = torch.nn.Module()
    gm = torch.fx.GraphModule(orig_module, traced_graph.graph)
    gm.recompile()

    def runnable_gm(*args):
        return torch.fx.Interpreter(gm).run(*args)
    with V.graph.fake_mode:
        gm = make_fx(runnable_gm)(*inputs)
    view_to_reshape(gm)
    non_contiguous_indices["outputs"] = [i + num_inputs
                                         for i, call_output in enumerate(call_outputs)
                                         if not V.graph.try_get_buffer(call_output).layout.is_contiguous()]
    return (gm, call_args, {"num_outputs": num_outputs,
                            "non_contiguous_indices": non_contiguous_indices, })


class NpuDvmKernel(Kernel):
    def __init__(
            self,
            gm: torch.fx.GraphModule,
            snodes: list[scheduler.SchedulerNode],
            call_args: list[str],
            non_contiguous_indices: list[int],
            num_outputs: list[int] = None,
    ):
        super().__init__()
        self._gm = gm

        if dvm_config.online_acc_comp:
            modify_gm_for_acc_comp(self._gm)

        self._snodes = snodes
        self._call_args = call_args
        self.non_contiguous_indices = non_contiguous_indices
        self.num_outputs = num_outputs

    def get_call_args(self):
        return self._call_args

    def call_kernel(self, name: str, node: Optional[torch.fx.Node] = None):
        wrapper = V.graph.wrapper_code
        call_args = self.get_call_args()
        for call_arg in call_args:
            if call_arg.startswith('_uwu_'):
                expression = map_strings_to_operators(call_arg)
                wrapper.writeline(f'{call_arg} = {expression}')
        if len(call_args) > 0:
            wrapper.generate_kernel_call(name, call_args)

    def codegen_kernel(self, name=None):
        def is_node_dvm_supported(node):
            if node.op == "placeholder":
                meta = node.meta["val"]
                if isinstance(meta, torch._subclasses.FakeTensor):
                    return meta.dtype in [*DVM_SUPPORT_TYPE, *_extra_int_types]
            if node.op == "call_function":
                return _is_node_supported_by_dvm_rule(node)
            return True

        if all(is_node_dvm_supported(node) for node in self._gm.graph.nodes):
            self.dvm_codegen = DvmCodegenInterpreter(self._gm, ktype="vector")
            self.dvm_codegen.run()
            return self.dvm_codegen.code.getvalue()

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


class NpuDvmSchedulingKernel(NpuTritonKernel):
    def remove_kernel_local_buffers(self):
        self.must_keep_buffers.clear()
        super().remove_kernel_local_buffers()


class NpuDvmScheduling(SIMDScheduling):
    kernel_type = NpuDvmSchedulingKernel
    meta_kernel_type = NpuDvmKernel

    def _handle_complete_fallback_mode(
        self, compile_wrapper, name, meta, wrapper, metadata_comment, mlir_kernel,
        *, gpu=True,
    ):
        compile_wrapper.writeline(f"async_compile.import_fx({name!r}, kernel_meta={meta})")
        metadata_comment = f'"""\n{mlir_kernel._gm.print_readable(print_output=False)}\n"""'
        wrapper.define_kernel(name, compile_wrapper.getvalue(), metadata_comment, gpu=gpu)

    def _dump_fx_graph_for_fallback(self, mlir_kernel, device, graph_hash):
        """Cache the FX model for fallback execution and online accuracy checks."""
        cache_root = os.getenv("TORCHINDUCTOR_CACHE_DIR")
        dump_path = os.path.join(
            cache_root,
            "traced_graph_cache",
            str(device.index),
            graph_hash
        )

        if not os.path.exists(dump_path):
            os.makedirs(dump_path, exist_ok=True)
            to_folder(mlir_kernel._gm, dump_path, graph_hash=graph_hash, module_name=graph_hash)

    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures, nodes):
        node_schedule = kernel_features.node_schedule

        tiling = self.select_tiling(
            node_schedule, kernel_features.numel, kernel_features.reduction_numel
        )

        kernels = self.create_kernel_choices(
            kernel_features, [tiling], {"features": kernel_features}
        )
        for kernel in kernels:
            super().codegen_node_schedule_with_kernel(node_schedule, kernel)
        MultiKernel.merge_workspaces_inplace(kernels)
        for kernel in kernels:
            V.graph.removed_buffers |= kernel.removed_buffers
            V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
            traced_graph, call_args, compile_kwargs = create_fx_from_snodes_by_traced_graph(
                nodes
            )
            mlir_kernel = self.meta_kernel_type(traced_graph, nodes, call_args, **compile_kwargs)
            with V.set_kernel_handler(mlir_kernel):
                src_code = mlir_kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, mlir_kernel, traced_graph)
            log.debug("Generating kernel code with kernel_name: %s", kernel_name)
            kernel.kernel_name = kernel_name
            kernel.code_hash = code_hash(src_code)
        del kernel

        final_kernel: Union[SIMDKernel, MultiKernel]
        if len(kernels) > 1:
            raise RuntimeError("MultiKernel not Implemented for this backend!")
        else:
            (final_kernel,) = kernels

        with V.set_kernel_handler(final_kernel):
            for node in kernel_features.scheduler_nodes():
                node.mark_run()

        self.codegen_comment(node_schedule)
        final_kernel.call_kernel(call_args, final_kernel.kernel_name)

        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernels[0].kernel_name)

        if (
            V.graph.wrapper_code.supports_intermediate_hooks
            and config.generate_intermediate_hooks
        ):
            live_outs = kernels[0].args.live_output_buffers()
            for node in kernel_features.scheduler_nodes():
                name = node.get_name()
                if name not in live_outs:
                    continue
                if node.node is None:
                    raise RuntimeError("assert node.node is not None")

                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters["inductor"]["intermediate_hooks"] += 1
                    V.graph.wrapper_code.writeline(
                        f"run_intermediate_hooks({origin_node.name!r}, {name})"
                    )

        self.scheduler.free_buffers()

    def codegen_node(self, node: Union[scheduler.SchedulerNode, object]):
        nodes: List[scheduler.SchedulerNode] = node.get_nodes()

        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group

        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        kernel_features = SIMDKernelFeatures(node_schedule, numel, rnumel)

        return self.codegen_node_schedule(kernel_features, nodes)

    def _get_npu_triton_scheduling(self):
        scheduling = getattr(self, "_npu_triton_scheduling", None)
        if scheduling is None:
            from torch_npu._inductor.codegen.scheduling import NPUTritonScheduling

            scheduling = NPUTritonScheduling(self.scheduler)
            self._npu_triton_scheduling = scheduling
        return scheduling

    def define_kernel(self, src_code, mlir_kernel, traced_graph):
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

            wrapper.src_to_kernel[kernel_key] = kernel_name
            current_device = V.graph.get_current_device_or_throw()

            compile_wrapper = IndentedBuffer()
            if (
                mlir_kernel.dvm_codegen is None
                or kernel_name in dvm_config.force_fallback_kernel_names
            ):
                kernel_meta = {
                    "device_str": current_device.type,
                    "device_index": current_device.index,
                    "num_outputs": mlir_kernel.num_outputs,
                    "traced_graph_cache": "traced_graph_cache",
                    "traced_graph_hash": traced_graph_hash,
                    "num_call_functions": get_num_call_functions(mlir_kernel._gm),
                }
                # FX fallback does not need autotuning. gpu=False keeps its definition
                # out of Inductor's compile-time Triton autotune block.
                self._handle_complete_fallback_mode(
                    compile_wrapper,
                    kernel_name,
                    kernel_meta,
                    wrapper,
                    "",
                    mlir_kernel,
                    gpu=False,
                )
                self._dump_fx_graph_for_fallback(
                    mlir_kernel,
                    current_device,
                    traced_graph_hash,
                )
            else:
                wrapper.add_import_once("from torch_npu._inductor import dvm")
                if dvm_config.dump_fx_test:
                    generate_dvm_fx_case(mlir_kernel._gm, fusion_type="mlir")
                code = mlir_kernel.dvm_codegen.code
                code.splice(
                    f"""
                    k.set_kernel_info(
                        {fused_kernel_name!r},  # kernel_name
                        {kernel_name!r},  # kernel_fullname
                    )
                    """,
                    strip=True,
                )
                func_name = kernel_name + "_build"
                func_code = code.getvalue().replace(
                    mlir_kernel.dvm_codegen.KERNEL_NAME_PLACEHOLDER, func_name
                )
                compile_wrapper.writeline(func_name)

                if dvm_config.online_acc_comp:
                    compile_wrapper.writeline(
                        f"{kernel_name}._acc_meta = {{"
                        f"'traced_graph_hash': {traced_graph_hash!r}, "
                        "'traced_graph_cache': 'traced_graph_cache', "
                        f"'device_index': {current_device.index}, "
                        f"'num_outputs': {mlir_kernel.num_outputs}}}"
                    )
                    self._dump_fx_graph_for_fallback(
                        mlir_kernel,
                        current_device,
                        traced_graph_hash,
                    )

                # DVM kernels do not need autotuning. gpu=False keeps their definitions
                # out of Inductor's compile-time Triton autotune block.
                wrapper.define_kernel(
                    kernel_name, compile_wrapper.getvalue(), func_code, gpu=False
                )

        return kernel_name

    def can_fuse_vertical(self, node1, node2):
        if node2.is_reduction() and any(
            origin.target == aten.cat.default
            for origin in aggregate_origins(node1.get_nodes())
        ):
            return False

        template1 = node1.get_template_node()
        template2 = node2.get_template_node()
        if isinstance(template1, ir.TritonTemplateBuffer) or isinstance(
            template2, ir.TritonTemplateBuffer
        ):
            # Keep the DVM/Triton boundary explicit.  The Triton template is
            # emitted as one fused kernel by NPUTritonScheduling below.
            return False
        if isinstance(template1, DvmTemplateBuffer):
            return can_fuse_dvm_epilogue(node1, node2)
        if isinstance(template2, DvmTemplateBuffer):
            return False
        if not dvm_config.disable_post_reduce_fusion:
            return super().can_fuse_vertical(node1, node2)

        _, (numel1, rnumel1) = node1.group
        _, (numel2, rnumel2) = node2.group
        why = WhyNoFuse(node1, node2)

        if node1.is_reduction():
            return False

        if not node2.is_reduction():
            return numel1 == numel2 and rnumel1 == rnumel2

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

    def can_fuse_horizontal(self, node1, node2):
        template1 = node1.get_template_node()
        template2 = node2.get_template_node()
        template_types = (DvmTemplateBuffer, ir.TritonTemplateBuffer)
        if isinstance(template1, template_types) or isinstance(
            template2, template_types
        ):
            return False
        if not dvm_config.disable_post_reduce_fusion:
            return super().can_fuse_horizontal(node1, node2)
        return False

    def get_fusion_pair_priority(self, node1, node2):
        priority = super().get_fusion_pair_priority(node1, node2)
        if (
            isinstance(node1.get_template_node(), DvmTemplateBuffer)
            and len(node1.get_nodes()) == 1
        ):
            return priority + 1
        return priority

    def codegen_template(
        self,
        template_node: SchedulerNode,
        epilogue_nodes: List[SchedulerNode],
        prologue_nodes: List[SchedulerNode] = (),
    ):
        template_buffer = template_node.get_template_node()
        if isinstance(template_buffer, ir.TritonTemplateBuffer):
            if prologue_nodes:
                raise RuntimeError(
                    "Triton templates under DVM do not support prologue fusion"
                )
            return self._get_npu_triton_scheduling().codegen_template(
                template_node, epilogue_nodes
            )
        if not isinstance(template_buffer, DvmTemplateBuffer):
            return super().codegen_template(
                template_node, epilogue_nodes, prologue_nodes
            )
        if prologue_nodes:
            raise RuntimeError(
                "DVM matmul template only supports pointwise epilogue fusion"
            )

        if (
            is_ascend950
            and not epilogue_nodes
            and template_buffer.op in (aten.mm.default, aten.bmm.default)
        ):
            with V.set_kernel_handler(Kernel(increase_kernel_count=False)):
                template_node.mark_run()
            template_buffer.codegen_unfused(V.graph.wrapper_code)
            self.free_buffers_in_scheduler()
            return

        snodes = [template_node, *epilogue_nodes]
        fused_node_names = set()
        for snode in snodes:
            fused_node_names.update(snode.get_operation_names())
        removed_buffers = {
            name
            for snode in snodes
            for name in snode.get_buffer_names()
            if self.scheduler.can_buffer_be_removed_through_fusion(
                name, fused_node_names
            )
        }
        V.graph.removed_buffers |= removed_buffers

        traced_graph, call_args, compile_kwargs = (
            create_fx_from_snodes_by_traced_graph(snodes)
        )
        mlir_kernel = self.meta_kernel_type(
            traced_graph, snodes, call_args, **compile_kwargs
        )
        with V.set_kernel_handler(mlir_kernel):
            src_code = mlir_kernel.codegen_kernel()

        need_trans_input = getattr(
            mlir_kernel.dvm_codegen, "need_trans_input", ()
        )
        for index, arg in enumerate(call_args):
            if arg in template_buffer.input_bindings:
                arg = V.graph.wrapper_code.val_to_arg_str(
                    template_buffer.input_bindings[arg]
                )
            if index < len(need_trans_input) and need_trans_input[index]:
                arg += ".mT"
            call_args[index] = arg

        kernel_name = self.define_kernel(src_code, mlir_kernel, traced_graph)
        with V.set_kernel_handler(mlir_kernel):
            for node in snodes:
                node.mark_run()
        self.codegen_comment(snodes)
        mlir_kernel.call_kernel(kernel_name, template_node.node)
        self.free_buffers_in_scheduler()


def _patch_lowering_type_checks():
    import torch._inductor.graph as inductor_graph
    import torch._inductor.lowering as inductor_lowering
    from torch_npu._inductor.dvm.inductor_patch import lowering as npu_lowering_mod
    import torch_npu._inductor.graph as npu_graph_mod

    fallback_node_due_to_unsupported_type = (
        inductor_lowering.fallback_node_due_to_unsupported_type
    )

    def _fallback_node_due_to_unsupported_type(
        node: torch.fx.Node, allow_cpu_inputs=True
    ):
        if node.target in (
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        ):
            # These HOPs use NPU Triton template lowerings under DVM.  Check
            # this before Inductor's generic HOP test replaces the registered
            # template lowering with a FallbackKernel.
            return False
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
    from torch_npu._inductor.dvm.inductor_patch.lowering import (
        _make_reduction_inner as original_make_reduction_inner,
        make_reduction as original_make_reduction,
        to_dtype,
    )

    def _traced_graph_contains_cat(traced_graph):
        graph = getattr(traced_graph, "graph", None)
        return graph is not None and any(
            node.op == "call_function"
            and node.target in (aten.cat, aten.cat.default)
            for node in graph.nodes
        )

    def _make_reduction_inner(x, *, axis, keepdims, dtype, override_return_dtype):
        # Materialize before the Reduction IR captures the pointwise cat loader;
        # the scheduler veto preserves the resulting boundary.
        if _traced_graph_contains_cat(x.get_traced_graph()):
            x.realize()
        return original_make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
        )

    def _make_reduction(reduction_type, override_return_dtype=None):
        if reduction_type != "sum":
            return original_make_reduction(reduction_type, override_return_dtype)

        reduce_sum = original_make_reduction(
            reduction_type, override_return_dtype=torch.float32
        )

        def reduce_sum_in_float32(x, axis=None, keepdims=False, *, dtype=None):
            if isinstance(axis, int):
                axis = [axis]
            elif axis is not None:
                axis = list(axis)
            out_dtype = override_return_dtype
            if out_dtype is None:
                out_dtype = x.get_dtype() if dtype is None else dtype

            result = reduce_sum(x, axis, keepdims, dtype=torch.float32)
            if out_dtype != torch.float32:
                result = to_dtype(result, out_dtype)
            return result

        return reduce_sum_in_float32

    npu_lowering._make_reduction_inner = _make_reduction_inner
    npu_lowering.make_reduction = _make_reduction


class DvmMlirFusionPatch:
    _enabled = False

    @staticmethod
    def enable() -> None:
        if DvmMlirFusionPatch._enabled:
            return
        from torch._dynamo import config as dynamo_config
        from torch._inductor import config as inductor_config

        # Enable float specialization until launch with scalar is supported.
        dynamo_config.specialize_float = True
        # Disable unroll reductions.
        inductor_config.unroll_reductions_threshold = 1
        # NPU ops always return contiguous tensors, which can differ from meta outputs.
        inductor_config.size_asserts = False
        inductor_config.allow_buffer_reuse = False
        inductor_config.comprehensive_padding = False
        patch_decomp()
        _patch_lowering_type_checks()
        _patch_lowering()
        _disable_addmm_fusion_for_dvm()
        if dvm_config.enable_matmul_fusion:
            patch_dvm_matmul_template_fusion()
        from torch_npu._inductor.compile_options import register_npu_backend

        register_npu_backend(NpuDvmScheduling, NpuDvmWrapperCodeGen)
        DvmMlirFusionPatch._enabled = True


DvmMlirFusionPatch.enable()
