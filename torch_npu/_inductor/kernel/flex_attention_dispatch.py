from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from torch._inductor import ir
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.select_algorithm import (
    SymbolicGridFn,
    TritonTemplateCaller,
)
from torch._inductor.virtualized import V
from torch_npu._inductor.kernel.flex_attention_tasklist import (
    DKDV_TASKLIST_HELPER_SOURCE,
    FlexAttentionDkdvDispatchSpec,
)


log = logging.getLogger("torch._inductor")
FLEXATTENTION_DISPATCH_STRATEGIES = (None, "fwd", "bwd_dkdv")


class FlexAttentionFwdDispatchKey(str, Enum):
    ALL_SPARSE_BLOCKS = "ALL_SPARSE_BLOCKS"


FLEX_ATTENTION_FWD_DISPATCH_HELPER_SOURCE = f"""\
def dynamic_flexattention_dispatch(
    arg_KV_NUM_BLKS,
    arg_KV_IDX,
    arg_FULL_KV_NUM_BLKS,
    arg_FULL_KV_IDX,
):
    if arg_FULL_KV_NUM_BLKS is None:
        return None
    if not torch.all(arg_FULL_KV_NUM_BLKS == 0).item():
        return None
    return {FlexAttentionFwdDispatchKey.ALL_SPARSE_BLOCKS.value!r}
"""


def get_flexattention_dispatch_strategy():
    from torch_npu._inductor import config as npu_config

    strategy = npu_config.flex_attention.flexattention_dispatch_strategy
    if strategy not in FLEXATTENTION_DISPATCH_STRATEGIES:
        raise ValueError(
            "Unsupported flexattention_dispatch_strategy="
            f"{strategy!r}; expected one of "
            f"{FLEXATTENTION_DISPATCH_STRATEGIES!r}"
        )
    return strategy


@dataclass(frozen=True)
class FlexAttentionDispatchContext:
    cpp_wrapper: bool
    aot_mode: bool
    is_subgraph: bool

    @classmethod
    def from_graph(cls, graph):
        return cls(
            cpp_wrapper=graph.cpp_wrapper,
            aot_mode=getattr(graph, "aot_mode", False),
            is_subgraph=hasattr(graph, "parent"),
        )


@dataclass(frozen=True)
class FlexAttentionDispatchCapabilities:
    python: bool
    cpp: bool
    aot: bool
    subgraph: bool

    def supports(self, context):
        if context.is_subgraph and not self.subgraph:
            return False
        if context.aot_mode and not self.aot:
            return False
        if context.cpp_wrapper and not self.cpp:
            return False
        if not (
            context.cpp_wrapper or context.aot_mode or context.is_subgraph
        ):
            return self.python
        return True


PYTHON_ONLY = FlexAttentionDispatchCapabilities(True, False, False, False)
ALL_CODEGEN_CONTEXTS = FlexAttentionDispatchCapabilities(
    True, True, True, True
)


class FlexAttentionDispatchStrategy(Protocol):
    def fallback_reason(self) -> str | None:
        raise NotImplementedError

    def build_runtime_renderers(self, output_node, primary_choice_name):
        raise NotImplementedError

    def validate_kernels(self, primary, variants) -> None:
        raise NotImplementedError

    def emit_wrapper(self, wrapper, primary, variants) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ResolvedFlexAttentionDispatchPlan:
    primary_name: str
    strategy: FlexAttentionDispatchStrategy | None
    fallback_reason: str | None = None

    @property
    def requires_composite_codegen(self):
        return self.strategy is not None


@dataclass(frozen=True)
class FlexAttentionDispatchPlan:
    primary_name: str
    strategy: FlexAttentionDispatchStrategy | None
    capabilities: FlexAttentionDispatchCapabilities

    def resolve(self, context):
        if self.strategy is None:
            return ResolvedFlexAttentionDispatchPlan(self.primary_name, None)
        if not self.capabilities.supports(context):
            return ResolvedFlexAttentionDispatchPlan(
                self.primary_name,
                None,
                "unsupported code-generation context",
            )
        reason = self.strategy.fallback_reason()
        return ResolvedFlexAttentionDispatchPlan(
            self.primary_name,
            None if reason else self.strategy,
            reason,
        )


def resolve_flex_attention_dispatch_plan(plan, graph):
    if plan is None:
        return None
    resolved = plan.resolve(FlexAttentionDispatchContext.from_graph(graph))
    if resolved.fallback_reason:
        key = (resolved.primary_name, resolved.fallback_reason)
        logged = getattr(
            graph, "_npu_flexattention_dispatch_fallbacks", None
        )
        if logged is None:
            logged = set()
            graph._npu_flexattention_dispatch_fallbacks = logged
        if key not in logged:
            logged.add(key)
            log.info(
                "FlexAttention dispatch plan %s fell back to primary: %s",
                resolved.primary_name,
                resolved.fallback_reason,
            )
    return resolved


class FlexAttentionDispatchTemplateBuffer(ir.TritonTemplateBuffer):
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        dispatch_plan,
        mutated_inputs=None,
        allowed_prologue_inps=None,
    ):
        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=make_kernel_render,
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        self.dispatch_plan = dispatch_plan


class FlexAttentionDispatchTemplateCaller(TritonTemplateCaller):
    def __init__(self, *args, dispatch_plan, **kwargs):
        super().__init__(*args, **kwargs)
        self.dispatch_plan = dispatch_plan

    def output_node(self):
        buffer = FlexAttentionDispatchTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            dispatch_plan=self.dispatch_plan,
            mutated_inputs=self.mutated_inputs,
            allowed_prologue_inps=self.allowed_prologue_inps,
        )
        return ir.TensorBox.create(buffer)


@dataclass(frozen=True)
class FwdDispatchStrategy:
    all_sparse_renderer_factory: Any

    def fallback_reason(self):
        return None

    def build_runtime_renderers(self, output_node, primary_choice_name):
        variants = {
            FlexAttentionFwdDispatchKey.ALL_SPARSE_BLOCKS.value: (
                self.all_sparse_renderer_factory(output_node)
            )
        }
        suffix = primary_choice_name.rsplit("_", 1)[-1]
        for renderer in variants.values():
            renderer.kernel._npu_codegen_kernel_name = (
                f"{renderer.kernel._npu_codegen_kernel_name}_{suffix}"
            )
        return variants

    def validate_kernels(self, primary, variants):
        if set(variants) != {
            FlexAttentionFwdDispatchKey.ALL_SPARSE_BLOCKS.value
        }:
            raise AssertionError("forward dispatch renderer set mismatch")
        if primary.kernel.meta is None:
            raise AssertionError("forward primary meta is None")
        _assert_mutated_inputs_are_kernel_args(primary, "forward primary")
        for variant_name, variant in variants.items():
            if primary.call_args != variant.call_args:
                raise AssertionError(
                    f"forward dispatch kernel ABI mismatch: {variant_name}"
                )
            if primary.kernel.grid_fn is not variant.kernel.grid_fn:
                raise AssertionError(
                    f"forward dispatch kernel grid mismatch: {variant_name}"
                )
            if primary.kernel.call_sizes != variant.kernel.call_sizes:
                raise AssertionError(
                    "forward dispatch kernel call-size mismatch: "
                    f"{variant_name}"
                )
            if variant.kernel.meta is None:
                raise AssertionError(
                    f"forward variant meta is None: {variant_name}"
                )
            if _kernel_output_contract(
                primary.kernel
            ) != _kernel_output_contract(variant.kernel):
                raise AssertionError(
                    "forward dispatch output contract mismatch: "
                    f"{variant_name}"
                )
            if _kernel_mutation_contract(
                primary
            ) != _kernel_mutation_contract(variant):
                raise AssertionError(
                    "forward dispatch mutation contract mismatch: "
                    f"{variant_name}"
                )
            _assert_mutated_inputs_are_kernel_args(
                variant, f"forward variant {variant_name}"
            )

    def emit_wrapper(self, wrapper, primary, variants):
        wrapper.write_runtime_helper_once(
            "flex_attention_fwd_dispatch",
            FLEX_ATTENTION_FWD_DISPATCH_HELPER_SOURCE,
        )
        suffix = wrapper.next_kernel_suffix()
        dispatch_mode = f"flexattention_dispatch_{suffix}"
        device = V.graph.get_current_device_or_throw()
        stream = PythonWrapperCodegen.write_get_raw_stream(
            wrapper, device.index, V.graph
        )
        named_inputs = primary.kernel.named_input_nodes
        wrapper.writeline(
            f"{dispatch_mode} = dynamic_flexattention_dispatch("
            f"{named_inputs['KV_NUM_BLKS'].get_name()}, "
            f"{named_inputs['KV_IDX'].get_name()}, "
            f"{named_inputs['FULL_KV_NUM_BLKS'].get_name()}, "
            f"{named_inputs['FULL_KV_IDX'].get_name()})"
        )
        if not isinstance(primary.kernel.grid_fn, SymbolicGridFn):
            raise AssertionError("forward dispatch requires a symbolic grid")
        grid = primary.kernel.grid_fn.sympy_call(
            *primary.kernel.call_sizes,
            primary.kernel.meta,
        )
        all_sparse_key = FlexAttentionFwdDispatchKey.ALL_SPARSE_BLOCKS.value
        variant = variants[all_sparse_key]
        primary_args = ", ".join(
            wrapper.prepare_triton_kernel_call(
                [*primary.call_args, *grid]
            )
        )
        variant_args = ", ".join(
            wrapper.prepare_triton_kernel_call(
                [*variant.call_args, *grid]
            )
        )
        wrapper.writeline(f"if {dispatch_mode} == {all_sparse_key!r}:")
        wrapper.writeline(
            f"    {variant.name}.run({variant_args}, stream={stream})"
        )
        wrapper.writeline("else:")
        wrapper.writeline(
            f"    {primary.name}.run({primary_args}, stream={stream})"
        )


@dataclass(frozen=True)
class DkdvDispatchStrategy:
    runtime_renderer_factory: Any | None
    dispatch_spec: FlexAttentionDkdvDispatchSpec | None
    eligibility_reason: str | None = None

    def fallback_reason(self):
        return self.eligibility_reason

    def build_runtime_renderers(self, output_node, primary_choice_name):
        if self.runtime_renderer_factory is None:
            raise AssertionError()
        renderers = self.runtime_renderer_factory(output_node)
        suffix = primary_choice_name.rsplit("_", 1)[-1]
        for renderer in renderers.values():
            renderer.kernel._npu_codegen_kernel_name = (
                f"{renderer.kernel._npu_codegen_kernel_name}_{suffix}"
            )
        return renderers

    def validate_kernels(self, primary, variants):
        if self.dispatch_spec is None:
            raise AssertionError()
        if set(variants) != {"tasklist", "tasklist_no_split", "reduce"}:
            raise AssertionError("dK/dV dispatch renderer set mismatch")
        if self.dispatch_spec.launch_programs <= 0:
            raise AssertionError(
                "dK/dV dispatch requires a positive launch grid"
            )
        if primary.kernel.meta is None:
            raise AssertionError("dK/dV primary meta is None")
        expected_mutations = frozenset(primary.mutated_input_names)
        expected_dkdv = frozenset(
            (
                primary.kernel.named_input_nodes["DK"].get_name(),
                primary.kernel.named_input_nodes["DV"].get_name(),
            )
        )
        if expected_mutations != expected_dkdv:
            raise AssertionError("dK/dV primary mutation contract mismatch")
        _assert_mutated_inputs_are_kernel_args(primary, "dK/dV primary")

        tasklist_runtime_args = (
            "work_items_t",
            "task_offsets_t",
            "dk_partial",
            "dv_partial",
        )
        primary_output = _kernel_output_contract(primary.kernel)
        primary_mutation = _kernel_mutation_contract(primary)
        for name in ("tasklist", "tasklist_no_split"):
            variant = variants[name]
            if variant.kernel.meta is None:
                raise AssertionError(f"dK/dV {name} meta is None")
            _assert_same_grid_and_sizes(primary, variant, f"dK/dV {name}")
            base_args = _validate_runtime_args(
                variant, tasklist_runtime_args, f"dK/dV {name}"
            )
            if base_args != primary.call_args:
                raise AssertionError(f"dK/dV {name} base ABI mismatch")
            if _kernel_output_contract(variant.kernel) != primary_output:
                raise AssertionError(
                    f"dK/dV {name} output contract mismatch"
                )
            if _kernel_mutation_contract(variant) != primary_mutation:
                raise AssertionError(
                    f"dK/dV {name} mutation contract mismatch"
                )
            _assert_mutated_inputs_are_kernel_args(
                variant, f"dK/dV {name}"
            )

        reduce = variants["reduce"]
        if reduce.kernel.meta is None:
            raise AssertionError("dK/dV reduce meta is None")
        _assert_same_grid_and_sizes(primary, reduce, "dK/dV reduce")
        reduce_base_args = _validate_runtime_args(
            reduce,
            ("dk_partial", "dv_partial", "split_bases_t"),
            "dK/dV reduce",
        )
        expected_reduce_args = (
            primary.kernel.named_input_nodes["DK"].get_name(),
            primary.kernel.named_input_nodes["DV"].get_name(),
        )
        if reduce_base_args != expected_reduce_args:
            raise AssertionError("dK/dV reduce base ABI mismatch")
        if _kernel_output_contract(reduce.kernel) != primary_output:
            raise AssertionError("dK/dV reduce output contract mismatch")
        if reduce.mutated_input_names != primary.mutated_input_names:
            raise AssertionError("dK/dV reduce mutation contract mismatch")
        if _kernel_mutation_state(
            reduce.kernel, include_reset=False
        ) != _kernel_mutation_state(primary.kernel, include_reset=False):
            raise AssertionError("dK/dV reduce mutation contract mismatch")
        if reduce.kernel.reset_to_zero_arg_names is not None:
            raise AssertionError(
                "dK/dV reduce must not reset accumulated outputs"
            )
        _assert_mutated_inputs_are_kernel_args(reduce, "dK/dV reduce")

    def emit_wrapper(self, wrapper, primary, variants):
        if self.dispatch_spec is None:
            raise AssertionError()
        wrapper.write_runtime_helper_once(
            "flex_attention_dkdv_tasklist",
            DKDV_TASKLIST_HELPER_SOURCE,
        )
        suffix = wrapper.next_kernel_suffix()
        prefix = f"dkdv_tasklist_{suffix}"
        use_tasklist = f"{prefix}_use_tasklist"
        max_sub = f"{prefix}_max_sub"
        work_items = f"{prefix}_work_items_t"
        task_offsets = f"{prefix}_task_offsets_t"
        split_bases = f"{prefix}_split_bases_t"
        dk_partial = f"{prefix}_dk_partial"
        dv_partial = f"{prefix}_dv_partial"
        num_split_bases = f"{prefix}_num_split_bases"
        spec = self.dispatch_spec
        named_inputs = primary.kernel.named_input_nodes
        dk_name = named_inputs["DK"].get_name()
        dv_name = named_inputs["DV"].get_name()
        device = V.graph.get_current_device_or_throw()
        stream = PythonWrapperCodegen.write_get_raw_stream(
            wrapper, device.index, V.graph
        )
        wrapper.writeline(
            f"{use_tasklist}, {work_items}, {task_offsets}, "
            f"{split_bases}, {max_sub} = _get_or_build_dkdv_task_list("
            f"{named_inputs['Q_NUM_BLKS'].get_name()}, "
            f"{named_inputs['FULL_Q_NUM_BLKS'].get_name()}, "
            f"{spec.batch_size}, {spec.num_kv_heads}, "
            f"{spec.num_kv_blocks}, {spec.sparse_kv_multiple}, "
            f"{spec.launch_programs}, {dk_name}.device)"
        )
        wrapper.writeline(f"if {use_tasklist}:")
        wrapper.writeline(f"    {num_split_bases} = {split_bases}.size(0)")
        wrapper.writeline(f"    if {num_split_bases} > 0:")
        wrapper.writeline(
            f"        {dk_partial} = torch.empty_strided("
            f"({max_sub}, *{dk_name}.size()), "
            f"({spec.partial_dk_stride}, *{dk_name}.stride()), "
            f"dtype=torch.float32, device={dk_name}.device)"
        )
        wrapper.writeline(f"        {dk_partial}.zero_()")
        wrapper.writeline(
            f"        {dv_partial} = torch.empty_strided("
            f"({max_sub}, *{dv_name}.size()), "
            f"({spec.partial_dv_stride}, *{dv_name}.stride()), "
            f"dtype=torch.float32, device={dv_name}.device)"
        )
        wrapper.writeline(f"        {dv_partial}.zero_()")
        wrapper.writeline("    else:")
        wrapper.writeline(f"        {split_bases} = None")
        wrapper.writeline(f"        {dk_partial} = {dk_name}")
        wrapper.writeline(f"        {dv_partial} = {dv_name}")

        runtime_values = {
            "work_items_t": work_items,
            "task_offsets_t": task_offsets,
            "split_bases_t": split_bases,
            "dk_partial": dk_partial,
            "dv_partial": dv_partial,
        }

        def replace_runtime_args(rendered):
            replacements = {
                outer_name: runtime_values[wrapper_name]
                for outer_name, wrapper_name in rendered.runtime_arg_names.items()
            }
            return [
                replacements.get(str(arg), arg) for arg in rendered.call_args
            ]

        tasklist = variants["tasklist"]
        tasklist_no_split = variants["tasklist_no_split"]
        reduce = variants["reduce"]
        tasklist_grid = [spec.launch_programs, 1, 1]
        tasklist_args = ", ".join(
            wrapper.prepare_triton_kernel_call(
                [*replace_runtime_args(tasklist), *tasklist_grid]
            )
        )
        no_split_args = ", ".join(
            wrapper.prepare_triton_kernel_call(
                [
                    *replace_runtime_args(tasklist_no_split),
                    *tasklist_grid,
                ]
            )
        )
        wrapper.writeline(f"    if {num_split_bases} > 0:")
        wrapper.writeline(
            f"        {tasklist.name}.run({tasklist_args}, stream={stream})"
        )
        wrapper.writeline("    else:")
        wrapper.writeline(
            f"        {tasklist_no_split.name}.run("
            f"{no_split_args}, stream={stream})"
        )
        reduce_args = ", ".join(
            wrapper.prepare_triton_kernel_call(
                [
                    *replace_runtime_args(reduce),
                    num_split_bases,
                    1,
                    1,
                ]
            )
        )
        wrapper.writeline(f"    if {num_split_bases} > 0:")
        wrapper.writeline(
            f"        {reduce.name}.run({reduce_args}, stream={stream})"
        )
        wrapper.writeline(
            f"    del {work_items}, {task_offsets}, {split_bases}, "
            f"{dk_partial}, {dv_partial}"
        )
        wrapper.writeline("else:")
        primary_args = ", ".join(
            wrapper.prepare_triton_kernel_call(
                [*primary.call_args, spec.launch_programs, 1, 1]
            )
        )
        wrapper.writeline(
            f"    {primary.name}.run({primary_args}, stream={stream})"
        )


@dataclass(frozen=True)
class RenderedFlexAttentionKernel:
    name: str
    kernel: Any
    call_args: tuple[Any, ...]
    runtime_arg_names: dict[str, str]
    mutated_input_names: tuple[str, ...]


def _kernel_output_contract(kernel):
    return (
        tuple(kernel.args.output_buffers.items()),
        tuple(kernel.args.inplace_buffers.items()),
    )


def _kernel_mutation_state(kernel, *, include_reset=True):
    state = (
        frozenset(kernel.mutations),
        frozenset(kernel.removed_buffers),
        frozenset(kernel.inplaced_to_remove),
    )
    if include_reset:
        state += (tuple(kernel.reset_to_zero_arg_names or ()),)
    return state


def _kernel_mutation_contract(rendered):
    return (
        rendered.mutated_input_names,
        _kernel_mutation_state(rendered.kernel),
    )


def _assert_mutated_inputs_are_kernel_args(rendered, label):
    call_args = frozenset(map(str, rendered.call_args))
    if not frozenset(rendered.mutated_input_names) <= call_args:
        raise AssertionError(
            f"{label} mutation arguments are missing from the kernel ABI"
        )


def _assert_same_grid_and_sizes(primary, variant, label):
    if primary.kernel.grid_fn is not variant.kernel.grid_fn:
        raise AssertionError(f"{label} grid mismatch")
    if primary.kernel.call_sizes != variant.kernel.call_sizes:
        raise AssertionError(f"{label} call-size mismatch")


def _validate_runtime_args(rendered, expected_wrapper_names, label):
    runtime_names = rendered.runtime_arg_names
    if tuple(runtime_names.values()) != expected_wrapper_names:
        raise AssertionError(f"{label} runtime ABI mismatch")
    runtime_outer_names = tuple(runtime_names)
    call_arg_names = tuple(map(str, rendered.call_args))
    if not all(
        call_arg_names.count(name) == 1 for name in runtime_outer_names
    ):
        raise AssertionError(f"{label} runtime ABI mismatch")
    return tuple(
        arg
        for arg in rendered.call_args
        if str(arg) not in runtime_names
    )


def _render_source(kernel, render):
    with kernel:
        partial_code = render()
    if not isinstance(partial_code, str):
        partial_code.finalize_hook("<DEF_KERNEL>")
        partial_code.finalize_hook("<ARGDEFS>", strict=False)
    with V.set_kernel_handler(kernel):
        if not isinstance(partial_code, str):
            if "<STORE_OUTPUT>" in kernel.subgraph_bodies:
                with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                    partial_code.finalize_hook("<STORE_OUTPUT>")
            return partial_code.code
        return partial_code


def codegen_flex_attention_dispatch_template(
    scheduling, template_node, epilogue_nodes
):
    if epilogue_nodes:
        raise NotImplementedError(
            "epilogue fusion is unsupported for composite FlexAttention "
            "dispatch templates"
        )

    composite = template_node.node
    strategy = composite.dispatch_plan.strategy
    if strategy is None:
        raise AssertionError("composite FlexAttention plan has no strategy")

    primary_kernel, primary_render = composite.make_kernel_render(composite)
    primary_source = _render_source(primary_kernel, primary_render)
    primary_name, _ = scheduling.define_kernel(
        primary_source, [template_node], primary_kernel, None
    )
    _, primary_args, _, _ = primary_kernel.args.python_argdefs()
    primary = RenderedFlexAttentionKernel(
        primary_name,
        primary_kernel,
        tuple(primary_args),
        {},
        tuple(
            node.get_name() for node in composite.mutated_inputs or ()
        ),
    )

    renderers = strategy.build_runtime_renderers(
        composite,
        primary_kernel._npu_codegen_kernel_name,
    )
    variants = {}
    for variant_name, renderer in renderers.items():
        with renderer.patch_runtime_args():
            source = _render_source(renderer.kernel, renderer.render)
        kernel_name, _ = scheduling.define_kernel(
            source, [template_node], renderer.kernel, None
        )
        _, call_args, _, _ = renderer.python_argdefs()
        variants[variant_name] = RenderedFlexAttentionKernel(
            kernel_name,
            renderer.kernel,
            tuple(call_args),
            dict(renderer.runtime_arg_names),
            primary.mutated_input_names,
        )

    strategy.validate_kernels(primary, variants)
    template_node.mark_run()
    wrapper = V.graph.wrapper_code
    wrapper.write_triton_header_once()
    strategy.emit_wrapper(wrapper, primary, variants)

    for rendered in (primary, *variants.values()):
        V.graph.removed_buffers |= rendered.kernel.removed_buffers
        V.graph.inplaced_to_remove |= rendered.kernel.inplaced_to_remove
    scheduling.codegen_comment([template_node])
    scheduling.scheduler.free_buffers()
