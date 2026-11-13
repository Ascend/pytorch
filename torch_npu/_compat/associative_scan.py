import functools
import importlib

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree

from torch_npu._compat._impl import compat_impl, compat_impl_container

__all__ = ["associative_scan"]

@compat_impl(key="associative_scan_npu_allowlist", lt=(2, 14))
def _build_associative_scan():
    upstream = importlib.import_module("torch._higher_order_ops.associative_scan")

    wrap_combine_fn_flat = upstream.wrap_combine_fn_flat
    generic_associative_scan = upstream.generic_associative_scan
    associative_scan_op = upstream.associative_scan_op
    _maybe_compile_and_run_fn = upstream._maybe_compile_and_run_fn

    def associative_scan(
        combine_fn,
        xs,
        dim,
        reverse=False,
        combine_mode="pointwise",
    ):
        leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

        def _validate_input(cfn, lxs, d, r, cm):
            # Basic arguments check
            if not callable(cfn):
                raise ValueError(f"Combine_fn must be a callable, but got {cfn}")
            if not isinstance(d, int):
                raise ValueError("Dim must be an int, but got " + str(type(d)))
            if not isinstance(r, bool):
                raise RuntimeError("Reverse must be a bool, but got " + str(type(r)))
            if cm not in ["pointwise", "generic"]:
                raise ValueError(
                    f"Combine_mode must either 'pointwise' or 'generic', but got {cm}"
                )
            if cm == "pointwise" and not all(
                l.device.type in ("cuda", "xpu", "npu") for l in lxs
            ):
                raise ValueError(
                    "For combine_mode='pointwise', all input tensors need to be on CUDA or XPU"
                )

            # Checks for xs
            if len(lxs) == 0:
                raise ValueError("Expected at least 1 xs leaf")
            if any(not isinstance(x, torch.Tensor) for x in lxs):
                raise ValueError("xs leaves must be a Tensor")
            if any(x.is_sparse for x in lxs):
                raise ValueError(
                    "xs leaves must dense Tensors, consider using `to_dense()`"
                )
            if any(x.ndim <= d for x in lxs):
                raise ValueError(
                    "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
                )
            if any(x.shape[d] == 0 for x in lxs):
                raise ValueError(
                    "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
                )

        ndim = leaves_xs_orig[0].ndim
        dim = utils.canonicalize_dim(ndim, dim)

        _validate_input(combine_fn, leaves_xs_orig, dim, reverse, combine_mode)

        # Move scan dim to 0 and always perform scan on dim 0
        leaves_xs = [torch.movedim(elem, dim, 0) for elem in leaves_xs_orig]

        if reverse:
            leaves_xs = [torch.flip(elem, [0]) for elem in leaves_xs]

        if combine_mode == "generic":
            combine_fn = functools.partial(
                wrap_combine_fn_flat,
                combine_fn=torch.vmap(
                    combine_fn,
                    in_dims=(
                        pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                        pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                    ),
                    out_dims=0,
                ),
                spec=spec_xs,
                num_leaves=len(leaves_xs),
            )
            out = generic_associative_scan(combine_fn, leaves_xs, additional_inputs=())
            out = pytree.tree_unflatten(out, spec_xs)
        else:
            combine_fn = functools.partial(
                wrap_combine_fn_flat,
                combine_fn=combine_fn,
                spec=spec_xs,
                num_leaves=len(leaves_xs),
            )

            def run_flattened_associative_scan(combine_fn, leaves_xs):
                return associative_scan_op(combine_fn, leaves_xs, additional_inputs=())

            out = _maybe_compile_and_run_fn(
                run_flattened_associative_scan,
                combine_fn,
                leaves_xs,
            )

        if reverse:
            out = pytree.tree_map(lambda elem: elem.flip([0]), out)

        out = pytree.tree_map(lambda elem: torch.movedim(elem, 0, dim), out)

        return out

    upstream.associative_scan = associative_scan
    return associative_scan

@compat_impl(key="associative_scan_npu_allowlist", ge=(2, 14))
def _identify_associative_scan():
    upstream = importlib.import_module("torch._higher_order_ops.associative_scan")
    return upstream.associative_scan

associative_scan = compat_impl_container["associative_scan_npu_allowlist"].apply()
