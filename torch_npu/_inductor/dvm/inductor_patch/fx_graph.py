"""FX transformations, post-grad integration and export helpers for DVM and MLIR.

This module only defines helpers; graph mutation and export happen when called.
Keep imports independent of either backend's initialization and execution code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.fx.graph_module import _addindent, _custom_builtins, warnings
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files


_NPU_DTYPE_CAST_OPS = [
    torch.ops.npu.npu_dtype_cast.default,
    torch.ops.npu.npu_dtype_cast_backward.default,
    torch.ops.npu._npu_dtype_cast.default,
    torch.ops.npu._npu_dtype_cast_backward.default,
]


def get_num_call_functions(graph):
    num_call_functions = 0
    for node in graph.graph.nodes:
        if node.op == "call_function" and node.target != torch.ops.aten.reshape.default:
            num_call_functions += 1
        if num_call_functions > 1:
            break
    return num_call_functions


def view_to_reshape(gm: torch.fx.GraphModule):
    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.view.default
    ):
        nd.target = torch.ops.aten.reshape.default

    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.div.Tensor
    ):
        if not (isinstance(nd.args[1], torch.fx.node.Node) and
                isinstance(nd.args[1].meta['val'], torch.Tensor)):
            nd.target = torch.ops.aten.div.Scalar

    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.add.Tensor
    ):
        if not (isinstance(nd.args[1], torch.fx.node.Node) and
                isinstance(nd.args[1].meta['val'], torch.Tensor)):
            nd.target = torch.ops.aten.add.Scalar

    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.sub.Tensor
    ):
        if not (isinstance(nd.args[1], torch.fx.node.Node) and
                isinstance(nd.args[1].meta['val'], torch.Tensor)):
            nd.target = torch.ops.aten.sub.Scalar

    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.mul.Tensor
    ):
        if not (isinstance(nd.args[1], torch.fx.node.Node) and
                isinstance(nd.args[1].meta['val'], torch.Tensor)):
            nd.target = torch.ops.aten.mul.Scalar

    for nd in gm.graph.find_nodes(
        op="call_function", target=torch.ops.prims.convert_element_type.default
    ):
        nd.target = torch.ops.npu.npu_dtype_cast.default


def modify_gm_for_acc_comp(gm: torch.fx.GraphModule):
    """
    In precision comparison mode, if the second argument of npu_dtype_cast is torch.bfloat16, change it to torch.float32.
    """
    for nd in gm.graph.nodes:
        if nd.target in _NPU_DTYPE_CAST_OPS:
            if nd.args[1] == torch.bfloat16:
                new_args = list(nd.args)
                new_args[1] = torch.float32
                nd.args = tuple(new_args)


def to_folder(
        gm: torch.fx.GraphModule,
        folder: Union[str, os.PathLike],
        graph_hash: str,
        module_name: str = "FxModule"):
    """Dumps out module to ``folder`` with ``module_name`` so that it can be
    imported with ``from <folder> import <module_name>``

    Args:

        folder (Union[str, os.PathLike]): The folder to write the code out to

        module_name (str): Top-level name to use for the ``Module`` while
            writing out the code
    """
    folder = Path(folder)
    Path(folder).mkdir(exist_ok=True)
    tab = " " * 4
    custom_builtins = "\n".join([v.import_str for v in _custom_builtins.values()])
    model_str = f"""
import torch
{custom_builtins}

from torch.nn import *

class {module_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
"""

    def _gen_model_repr(module_name: str, module: torch.nn.Module) -> Optional[str]:
        safe_reprs = [
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
        ]
        if type(module) in safe_reprs:
            return f"{module.__repr__()}"
        else:
            return None

    blobified_modules = []
    for module_name, module in gm.named_children():
        module_str = _gen_model_repr(module_name, module)
        if module_str is None:
            module_file = folder / f"{module_name}.pt"
            torch.save(module, module_file)
            blobified_modules.append(module_name)
            module_repr = module.__repr__().replace("\r", " ").replace("\n", " ")
            # weights_only=False as this is legacy code that saves the model
            module_str = (
                f"torch.load(r'{module_file}', weights_only=True) # {module_repr}"
            )
        model_str += f"{tab * 2}self.{module_name} = {module_str}\n"

    for buffer_name, buffer in gm._buffers.items():
        if buffer is None:
            continue
        model_str += f"{tab * 2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}, dtype={buffer.dtype}))\n"  # noqa: B950

    for param_name, param in gm._parameters.items():
        if param is None:
            continue
        model_str += f"{tab * 2}self.{param_name} = torch.nn.Parameter(torch.empty({list(param.shape)}, dtype={param.dtype}))\n"  # noqa: B950

    model_str += f"{_addindent(gm.code, 4)}\n"

    module_file = folder / f"{graph_hash}.py"
    module_file.write_text(model_str)

    if len(blobified_modules) > 0:
        warnings.warn(
            "Was not able to save the following children modules as reprs - "
            f"saved as pickled files instead: {blobified_modules}"
        )


def replace_iota_int64_to_int32(nd: torch.fx.Node):
    """
    Replace iota dtype from int64 to int32.
    """
    if nd.target in [torch.ops.prims.iota.default] and nd.kwargs['dtype'] == torch.int64:
        new_args = dict(nd.kwargs)
        new_args['dtype'] = torch.int32
        nd.kwargs = new_args


def npu_optimize_fx_graph(gm: torch.fx.GraphModule):
    """
    optimize fx graph for npu
    """
    aten_empty_nodes = set()
    for nd in gm.graph.nodes:
        replace_iota_int64_to_int32(nd)
        # Replace npu type_as ops in the GraphModule to cast ops.
        if nd.target == torch.ops.aten.empty.memory_format and len(nd.users) == 1:
            aten_empty_nodes.add(nd)
        if nd.target == torch.ops.aten.copy.default:
            node0 = nd.args[0]
            if node0 in aten_empty_nodes:
                with gm.graph.inserting_after(nd):
                    dtype = node0.kwargs.get('dtype')
                    op_target = torch.ops.npu.npu_dtype_cast.default
                    args = (nd.args[1], dtype)
                    new_node = gm.graph.call_function(op_target, args=args)
                    new_node.name = nd.name
                nd.replace_all_uses_with(new_node)
                gm.graph.erase_node(nd)
                aten_empty_nodes.remove(node0)
                gm.graph.erase_node(node0)

    gm.recompile()


def fold_sum_cast_to_dtype(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = gm.graph
    changed = False
    aten = torch.ops.aten
    prims = torch.ops.prims

    for cast_node in list(graph.nodes):
        if cast_node.op != "call_function":
            continue
        if cast_node.target != prims.convert_element_type.default:
            continue
        if len(cast_node.args) < 2:
            continue

        sum_node, target_dtype = cast_node.args[:2]
        if not isinstance(sum_node, torch.fx.Node) or not isinstance(target_dtype, torch.dtype):
            continue
        if sum_node.op != "call_function":
            continue
        if sum_node.target not in (aten.sum.default, aten.sum.dim_IntList):
            continue
        if len(sum_node.users) != 1:
            continue
        if sum_node.kwargs.get("dtype") is not None:
            continue

        if not sum_node.args or not isinstance(sum_node.args[0], torch.fx.Node):
            continue
        input_val = sum_node.args[0].meta.get("val", None)
        sum_val = sum_node.meta.get("val", None)
        cast_val = cast_node.meta.get("val", None)
        if (
            not isinstance(input_val, torch.Tensor)
            or not isinstance(sum_val, torch.Tensor)
            or not isinstance(cast_val, torch.Tensor)
        ):
            continue
        if input_val.dtype != target_dtype or cast_val.dtype != target_dtype:
            continue

        sum_node.kwargs = {**sum_node.kwargs, "dtype": target_dtype}
        sum_node.meta["val"] = cast_val
        if "tensor_meta" in cast_node.meta:
            sum_node.meta["tensor_meta"] = cast_node.meta["tensor_meta"]
        cast_node.replace_all_uses_with(sum_node)
        graph.erase_node(cast_node)
        changed = True

    if changed:
        graph.lint()
        gm.recompile()
    return gm


def _iter_custom_graph_passes(custom_pass):
    if custom_pass is None:
        return ()
    if isinstance(custom_pass, (list, tuple)):
        return tuple(custom_pass)
    return (custom_pass,)


class DvmMlirPostGradPass(CustomGraphPass):
    def __init__(self, previous_pass=None) -> None:
        self.previous_pass = previous_pass

    def __call__(self, graph: torch.fx.Graph) -> None:
        for pass_ in _iter_custom_graph_passes(self.previous_pass):
            pass_(graph)
        fold_sum_cast_to_dtype(graph.owning_module)

    def uuid(self):
        previous_uuids = []
        for pass_ in _iter_custom_graph_passes(self.previous_pass):
            if not isinstance(pass_, CustomGraphPass):
                return None
            uuid = pass_.uuid()
            if uuid is None:
                return None
            previous_uuids.append(uuid)
        return get_hash_for_files(
            (__file__,),
            extra=f"DvmMlirPostGradPass:{previous_uuids!r}",
        )


def patch_dvm_mlir_post_grad_pass(inductor_config=None) -> None:
    if inductor_config is None:
        from torch._inductor import config as inductor_config

    post_grad_custom_post_pass = inductor_config.post_grad_custom_post_pass
    if isinstance(post_grad_custom_post_pass, DvmMlirPostGradPass):
        return
    if any(
        isinstance(pass_, DvmMlirPostGradPass)
        for pass_ in _iter_custom_graph_passes(post_grad_custom_post_pass)
    ):
        return
    inductor_config.post_grad_custom_post_pass = DvmMlirPostGradPass(
        post_grad_custom_post_pass
    )
