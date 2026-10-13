"""Adapt the shared FlexAttention lowering without changing default registrations."""

import ast
import inspect
from collections import Counter

from torch_npu._inductor import config as npu_config
from torch_npu._inductor.kernel import flex_attention as default
from . import flex_attention_template as templates
from .flex_attention_config_generator import (
    generate_bwd_candidate_configs,
    generate_fwd_candidate_configs,
)


def _statement(source):
    body = ast.parse(source).body
    if len(body) != 1:
        raise AssertionError("expected one FlexAttention statement")
    return body[0]


class _ExperimentalChanges(ast.NodeTransformer):

    def __init__(self):
        self.counts = Counter()
        self.current_function = None

    def replace(self, node, old, new, label):
        if ast.dump(node, include_attributes=False) != ast.dump(
            _statement(old), include_attributes=False
        ):
            return node
        self.counts[label] += 1
        return ast.copy_location(_statement(new), node)

    def visit_Assign(self, node):
        self.generic_visit(node)
        node = self.replace(
            node,
            (
                "static_workspace_path = (not getattr(V.graph, 'aot_mode', False) "
                "and not is_multi_stream() and not _ir_has_dynamic_shape("
                "query, key, value, kv_num_blocks, kv_indices))"
            ),
            "static_workspace_path = False",
            "static_workspace",
        )
        if ast.unparse(node) == "kernel_options = dict(kernel_options)":
            graph = {"flex_attention": "subgraph", "flex_attention_backward": "joint_graph"}.get(self.current_function)
            if graph:
                self.counts["options"] += 1
                return ast.copy_location(_statement(
                    (
                        "kernel_options = policy.prepare_kernel_options("
                        f"kernel_options, score_mod_graph={graph}, "
                        "mask_mod_graph=mask_graph)"
                    )
                ), node)
        targets = {
            "dq_subgraphs": "dq_kernel_options['RUN_CAPTURED_GRADS'] = False",
            "dkdv_subgraphs": "dkdv_kernel_options['RUN_CAPTURED_GRADS'] = bool(joint_outputs.mutated_grads)",
            "dkdv_inputs_for_autotuning": "dkdv_inputs_for_autotuning += list(joint_outputs.mutated_grads)",
        }
        if isinstance(node.targets[0], ast.Tuple):
            target = node.targets[0].elts[0]
        else:
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in targets:
            self.counts[target.id] += 1
            return [node, ast.copy_location(_statement(targets[target.id]), node)]
        return node

    def visit_FunctionDef(self, node):
        if node.name == "make_bwd_subgraphs_and_mutations":
            self.counts["captured_subgraphs"] += 1
            return ast.copy_location(_statement("""
def make_bwd_subgraphs_and_mutations(base_mutated_inputs: list[Any]):
    subgraphs = [fw_subgraph_buffer, joint_outputs.grad_input, mask_graph_buffer]
    mutated_inputs = list(base_mutated_inputs)
    if len(base_mutated_inputs) == 2 and joint_outputs.mutated_grads:
        subgraphs.append(joint_outputs.captured_grads_compute)
        mutated_inputs.extend(joint_outputs.mutated_grads)
    return subgraphs, mutated_inputs
"""), node)
        previous = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = previous
        return node

    def visit_Assert(self, node):
        self.generic_visit(node)
        return self.replace(
            node,
            (
                "assert not joint_outputs.mutated_grads, ("
                "'FlexAttention backward mask-out requires an identity score_mod ' "
                "'without captured-gradient side effects')"
            ),
            "kernel_options['TORCHINDUCTOR_FLEXATTENTION_MASKOUT'] = True",
            "captured_grad_guard",
        )

    def visit_Call(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "maybe_append_choice"
            and any(keyword.arg == "dispatch_plan" for keyword in node.keywords)
        ):
            node.keywords = [
                keyword for keyword in node.keywords if keyword.arg != "dispatch_plan"
            ]
            self.counts["dispatch_plan"] += 1
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_build_runtime_compact_sparse_mask_mapping"
        ):
            node.keywords = [
                keyword for keyword in node.keywords if keyword.arg != "use_workspace"
            ]
            self.counts["mapping"] += 1
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_get_split_backward_subgraph_outputs"
        ):
            for keyword in node.keywords:
                if keyword.arg == "owns_captured_grads":
                    owner = "dkdv" if self.counts["captured_owner"] == 0 else "dq"
                    keyword.value = ast.parse(
                        "bool(joint_outputs.mutated_grads)" if owner == "dkdv" else "False",
                        mode="eval",
                    ).body
                    self.counts["captured_owner"] += 1
        return node


class _ZeroGradValueInit(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (
            len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "broadcasted_grad_value"
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "empty_strided"
        ):
            self.count += 1
            return [node, ast.copy_location(_statement(
                "broadcasted_grad_value = _force_fixed_layout("
                "lowerings[aten.fill_](broadcasted_grad_value, 0), value_strides)"
            ), node)]
        return node


def _compile_copy(function, globals_override, *, transform=False, zero_grad_value=False):
    source = inspect.getsource(function)
    tree = ast.parse(source)
    if zero_grad_value:
        changes = _ZeroGradValueInit()
        tree = changes.visit(tree)
        if changes.count != 1:
            raise RuntimeError(
                "default FlexAttention backward source changed: "
                f"expected one dV initialization, got {changes.count}"
            )
    changes = _ExperimentalChanges()
    if transform:
        tree = changes.visit(tree)
        expected = dict(
            static_workspace=1, captured_grad_guard=1, dispatch_plan=3,
            mapping=2, captured_owner=2, captured_subgraphs=1, options=2,
            dq_subgraphs=1, dkdv_subgraphs=1, dkdv_inputs_for_autotuning=1,
        )
        if dict(changes.counts) != expected:
            raise RuntimeError(
                f"default FlexAttention source changed: {dict(changes.counts)} != {expected}"
            )
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
    namespace = dict(default.__dict__)
    namespace.update(globals_override)
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    return namespace[function.__name__]


def _globals(policy, offsets, mapping):
    replacements = {
        "_supports_flex_attention_mask_out": lambda: bool(getattr(npu_config.flex_attention, "flexattention_mask_out", False)),
        "get_flexattention_dispatch_strategy": lambda: "none",
        "is_dkdv_tasklist_codegen_compatible": lambda **kw: (
            bool(getattr(npu_config.flex_attention, "bwd_dkdv_tasklist", False))
            and default.is_dkdv_tasklist_codegen_compatible(**kw)
        ),
        "_build_runtime_compact_sparse_mask_offsets": offsets,
        "_build_runtime_compact_sparse_mask_mapping": mapping,
        "generate_fwd_candidate_configs": generate_fwd_candidate_configs,
        "generate_bwd_candidate_configs": generate_bwd_candidate_configs,
        "flex_attention_template": templates.flex_attention_template,
        "flex_attention_backward_template": templates.flex_attention_backward_template,
        "flex_decoding_template": templates.flex_decoding_template,
        "policy": policy,
    }
    for attribute, symbol in (
        ("forward_mask_out_template", "flex_attention_fwd_mask_out"),
        ("forward_mask_compact_template", "flex_attention_fwd_mask_compact"),
        ("backward_dq_mask_out_template", "flex_attention_bwd_dq_mask_out"),
        ("backward_dkdv_mask_out_template", "flex_attention_bwd_dkdv_mask_out"),
        ("backward_mask_compact_template", "flex_attention_bwd_mask_compact"),
        ("backward_mask_pos_template", "flex_attention_bwd_mask_pos"),
        ("backward_tasklist_template", "flex_attention_bwd_dkdv_tasklist"),
        ("backward_tasklist_no_split_template", "flex_attention_bwd_dkdv_tasklist_no_split"),
        ("backward_reduce_template", "flex_attention_bwd_dkdv_reduce"),
    ):
        replacements[symbol] = getattr(policy, attribute)
    return replacements


def _decoding_copy(function, globals_override, extra_configs):
    tree = ast.parse(inspect.getsource(function))
    matches = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "configs"
            and isinstance(node.value, ast.List)
        ):
            node.value.elts.extend(
                ast.Tuple(elts=[ast.Constant(value=value) for value in config], ctx=ast.Load())
                for config in extra_configs
            )
            matches += 1
    if matches != 1:
        raise RuntimeError("default FlexAttention decoding configs changed")
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
    namespace = dict(default.__dict__)
    namespace.update(globals_override)
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    return namespace[function.__name__]


def register_experimental_flex_attention(policy, offsets, mapping):
    replacements = _globals(policy, offsets, mapping)
    for name in ("_lower_flex_attention_mask_in", "_lower_flex_attention_backward_mask_in"):
        replacements[name] = _compile_copy(
            getattr(default, name),
            replacements,
            zero_grad_value=name == "_lower_flex_attention_backward_mask_in",
        )
    replacements["_create_npu_flex_decoding_kernel"] = _decoding_copy(
        default._create_npu_flex_decoding_kernel,
        replacements,
        policy.extra_decode_configs,
    )
    register = _compile_copy(
        default._register_npu_inductor_flex_attention, replacements, transform=True
    )
    register()
