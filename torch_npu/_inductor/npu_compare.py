import importlib
import os
import sys
from typing import Any, Iterable, Mapping

import torch
from torch._inductor.compile_fx import clone_preserve_strides


def clone_for_accuracy(arg):
    if not isinstance(arg, torch.Tensor):
        return arg
    cloned = clone_preserve_strides(arg)
    return cloned.float() if cloned.dtype == torch.bfloat16 else cloned


def compare_outputs(
    actual_outputs: Iterable[Any],
    expected_outputs: Iterable[Any],
    kernel_name: str,
    tolerances: Mapping[Any, Mapping[str, float]],
):
    failed_indices = []
    for idx, (actual, expected) in enumerate(zip(actual_outputs, expected_outputs)):
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            continue
        if actual.dtype != expected.dtype:
            expected = expected.to(actual.dtype)
        
        tol = tolerances.get(actual.dtype, tolerances["default"])
        rtol, atol = tol["rtol"], tol["atol"]
        matches = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        if not matches.all():
            _report_mismatch(idx, actual, expected, matches, rtol, atol, kernel_name)
            failed_indices.append(idx)
        del matches
    
    return not failed_indices


def _report_mismatch(idx, actual, expected, matches, rtol, atol, kernel_name):
    try:
        abs_diff = torch.abs(actual - expected)
    except RuntimeError:
        abs_diff = torch.abs(actual.to(torch.float32) - expected.to(torch.float32))
    expected_abs = torch.abs(expected)
    if not expected_abs.is_floating_point() and not expected_abs.is_complex():
        expected_abs = expected_abs.to(torch.float32)
    rel_diff = abs_diff / torch.clamp(expected_abs, min=1e-20)
    rel_diff.masked_fill_(matches, 0)
    number_of_elements = matches.numel()
    total_mismatches = number_of_elements - int(torch.sum(matches))
    msg = (
        "CHECK ACCURACY FAILED! "
        f"Kernel: {kernel_name}, Output idx: {idx}, "
        f"Mismatched: {total_mismatches}/{number_of_elements} "
        f"({total_mismatches / number_of_elements:.1%}), "
        f"Greatest Rel Diff: {rel_diff.max().item()}, "
        f"Greatest Abs Diff: {abs_diff.max().item()}, "
        f"rtol: {rtol}, atol: {atol}"
    )
    print(msg, flush=True)
    del abs_diff, rel_diff


def get_triton_fx_graph_call(inductor_meta, auto_fallback=False):
        kernel_name = inductor_meta.get("kernel_name", "triton_")
        traced_graph_hash = inductor_meta.get("traced_graph_hash")
        dump_dir = inductor_meta.get("traced_graph_dir", "")
        dump_path = os.path.join(dump_dir, traced_graph_hash)
        if dump_dir == "" or not os.path.exists(dump_path):
            return None, None, None, None
        sys.path.append(dump_path)
        fx_module = importlib.import_module(traced_graph_hash)
        sys.path.remove(dump_path)

        model = fx_module.model
        num_inputs = fx_module.num_inputs
        num_outputs = fx_module.num_outputs
        non_contiguous_indices = fx_module.non_contiguous_indices
        mismatch_indices_shapes = fx_module.mismatch_indices_shapes

        def fx_graph_call(*fx_args):
            fx_inputs = [fx_args[idx].contiguous() if idx in non_contiguous_indices['inputs'] else \
                             fx_args[idx] for idx in range(num_inputs)]
            if len(mismatch_indices_shapes):
                for ind, shape in mismatch_indices_shapes.items():
                    if ind >= num_inputs:
                        break
                    fx_inputs[ind] = fx_inputs[ind].reshape(shape)
            model_outputs = model.forward(*fx_inputs)
            for idx, (out1, out2) in enumerate(zip(model_outputs, fx_args[num_inputs:(num_inputs + num_outputs)])):
                out1 = out1.reshape(out2.shape)
                if idx in non_contiguous_indices['outputs']:
                    out2.copy_(out1)
                else:
                    out2.data = out1.data

        def fallback_call(*args):
            fx_args = [args[idx] for idx in fx_module.call_args_mapping]
            return fx_graph_call(*fx_args)

        if auto_fallback:
            return fallback_call, kernel_name, None, None
        return fx_graph_call, kernel_name, dump_path, fx_module


def check_accuracy_triton(*args, launcher, grid, stream, inductor_meta, **kwargs):
    import torch_npu._inductor.config as npu_config
    fx_graph_call, kernel_name, dump_path, fx_module = get_triton_fx_graph_call(inductor_meta)
    if not fx_graph_call:
        return None
    call_outputs_indices = fx_module.call_args_mapping[fx_module.num_inputs:]

    fx_args = []
    for idx in fx_module.call_args_mapping:
        arg = args[idx]
        if isinstance(arg, torch.Tensor):
            fx_args.append(clone_for_accuracy(arg))
    
    fx_graph_call(*fx_args)

    launcher(*args, **kwargs, stream=stream)

    compare_outputs(
        [args[i] for i in call_outputs_indices],
        fx_args[fx_module.num_inputs:],
        kernel_name=kernel_name,
        tolerances=npu_config.acc_comp_tol,
    )

    for arg in fx_args:
        del arg
    return True


def check_accuracy_mlir(*args, kernel_name, launchers, num_outputs, dynamic, **kwargs):
    from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as anir_config
    launcher_fx = launchers[1]
    launcher = launchers[0]

    num_inputs = len(args) - num_outputs
    fx_outputs = [clone_for_accuracy(arg) for arg in args[num_inputs:]]
    fx_inputs = [clone_for_accuracy(arg) for arg in args[:num_inputs]]
    fx_args = fx_inputs + fx_outputs

    launcher_fx(*fx_args, **kwargs)

    if dynamic:
        args_new = ()
        for arg in args:
            if not torch.is_tensor(arg):
                args_new = args_new + (arg,)
                continue
            args_new = args_new + (arg, arg, 0) + arg.size() + arg.stride()
    else:
        args_new = args
    
    output = launcher(*args_new, **kwargs)
    result = compare_outputs(
        args[num_inputs:],
        fx_outputs,
        kernel_name=kernel_name,
        tolerances=anir_config.acc_comp_tol,
    )
    del fx_inputs
    return (output, result)


def _load_fx_model(acc_meta):
    """Load the traced FX GraphModule from disk for accuracy comparison."""
    if acc_meta.get('_fx_model') is not None:
        return acc_meta['_fx_model']
    dump_path = os.path.join(
        os.getenv("TORCHINDUCTOR_CACHE_DIR"),
        acc_meta['traced_graph_cache'],
        str(acc_meta['device_index']),
        acc_meta['traced_graph_hash'],
    )
    sys.path.insert(0, dump_path)
    try:
        module = importlib.import_module(acc_meta['traced_graph_hash'])
    finally:
        sys.path.remove(dump_path)
    Model = getattr(module, acc_meta['traced_graph_hash'])
    model = Model()
    acc_meta['_fx_model'] = model
    return model
 
 
def check_accuracy_dvm(kobj, acc_meta, kernel_name, args):
    """Run DVM kernel then compare outputs against FX graph reference."""
    from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir import config as anir_config

    fx_model = _load_fx_model(acc_meta)

    num_outputs = acc_meta['num_outputs']
    num_inputs = len(args) - num_outputs

    fx_inputs = [clone_for_accuracy(arg) for arg in args[:num_inputs]]
    fx_outputs = fx_model.forward(*fx_inputs)
    if not isinstance(fx_outputs, (tuple, list)):
        fx_outputs = (fx_outputs,)

    kobj.run(*args)

    compare_outputs(
        args[num_inputs:],
        fx_outputs,
        kernel_name=kernel_name,
        tolerances=anir_config.acc_comp_tol,
    )
