"""External runtime wiring for graph execution domains.

The graph tree core files are never edited.  Four module-level targets are
wrapped, each idempotently and preserving the original function via
``functools.wraps``:

1. ``torch_npu.npu._graph_tree.get_container`` -- when a domain is active,
   serve the domain's own native container; otherwise delegate to the
   original thread-local path untouched.
2. ``torch_npu.npu._graph_tree.reset_npugraph_trees`` -- coordinate the
   teardown of explicit domains before the original global reset runs.
3. ``torch._inductor.compile_fx.cudagraphify`` -- while a domain is active,
   create the graph callable through the context-aware :func:`npugraphify`
   below; ordinary compilations keep returning the original callable.
4. ``torch._inductor.cudagraph_trees.reset_cudagraph_trees`` -- invalidate
   per-domain caches on the ordinary Inductor reset path.

Once installed the wrappers stay for the lifetime of the process; there is no
uninstall and no per-call install/uninstall churn.
"""

import functools
import threading

from torch_npu._experimental.dual_stream_graph.domain import (
    active_domain_container,
    current_domain,
    reset_graph_execution_domains,
)


_core_hooks_lock = threading.Lock()


def _wrap_reset(original_reset):
    """Return ``original_reset`` extended with explicit-domain teardown."""
    if getattr(original_reset, "_resets_npu_execution_domains", False):
        return original_reset

    @functools.wraps(original_reset)
    def reset_with_domains():
        reset_graph_execution_domains()
        return original_reset()

    reset_with_domains._resets_npu_execution_domains = True
    return reset_with_domains


def install_graph_tree_domain_hooks(graph_tree_module):
    """Wrap container lookup and tree reset without editing the core module.

    The core calls ``get_container`` by module-global name, so wrapping the
    module attribute is sufficient; aliases captured before installation keep
    their original behaviour by design and must not be relied upon.
    """
    with _core_hooks_lock:
        original_get_container = graph_tree_module.get_container
        if not getattr(original_get_container, "_routes_npu_execution_domains", False):
            @functools.wraps(original_get_container)
            def get_container(device_index):
                container = active_domain_container(device_index)
                if container is not None:
                    return container
                return original_get_container(device_index)

            get_container._routes_npu_execution_domains = True
            graph_tree_module.get_container = get_container
        graph_tree_module.reset_npugraph_trees = _wrap_reset(
            graph_tree_module.reset_npugraph_trees
        )


def context_aware_npugraphify_callable(
    create_wrapper, *, use_trees, device_index, is_inference, is_backward,
    mutated_input_idxs,
):
    """One default lazy callable plus one per active domain and artifact.

    ``create_wrapper(first_inputs)`` must build a fresh original lazy
    callable.  The domain cache is keyed by ``artifact`` identity, so two
    compiled artifacts never share graph state, and two domains never share a
    manager-bound callable.
    """
    default_callable = None
    artifact = object()

    def run(new_inputs):
        nonlocal default_callable
        domain = current_domain()
        if domain is not None:
            if not use_trees or not is_inference or is_backward:
                raise RuntimeError(
                    "NPU graph execution domains require inference mode with graph trees"
                )
            if mutated_input_idxs:
                raise RuntimeError(
                    "NPU graph execution domains do not support input mutation"
                )
            if device_index != domain.device_index:
                raise RuntimeError(
                    "Compiled graph and execution domain must use the same device"
                )
            wrapper = domain.get_or_create_callable(
                artifact, lambda: create_wrapper(new_inputs)
            )
            return wrapper(new_inputs)
        if default_callable is None:
            default_callable = create_wrapper(new_inputs)
        return default_callable(new_inputs)

    return run


def npugraphify(
    model, static_input_idxs=(), *, device_index, stack_traces, is_backward,
    is_inference, constants=(), placeholders=(), mutated_input_idxs=(),
):
    """Context-aware drop-in for ``torch_npu.utils._graph_tree.npugraphify``.

    Same signature as the original.  Each execution domain receives its own
    original lazy callable, so the original's ``compiled_fn`` closure is never
    shared across domains.  The original factory selects its implementation at
    creation time, which may now happen after the compiler's config scope has
    exited, so only the tree-selection switch is restored while creating it.
    """
    from torch._inductor import config as inductor_config
    from torch_npu.utils._graph_tree import npugraphify as original_npugraphify

    use_trees = inductor_config.triton.cudagraph_trees

    def create_wrapper(_inputs):
        with inductor_config.patch({"triton.cudagraph_trees": use_trees}):
            return original_npugraphify(
                model, static_input_idxs, device_index=device_index,
                stack_traces=stack_traces, is_backward=is_backward,
                is_inference=is_inference, constants=constants,
                placeholders=placeholders, mutated_input_idxs=mutated_input_idxs,
            )

    return context_aware_npugraphify_callable(
        create_wrapper, use_trees=use_trees, device_index=device_index,
        is_inference=is_inference, is_backward=is_backward,
        mutated_input_idxs=mutated_input_idxs,
    )


def _inductor_routed_npugraphify(*args, **kwargs):
    # Ordinary compilations (no active domain) keep returning the original
    # callable, without paying any per-replay domain dispatch.  Ordinary
    # callables created this way cannot later be used as dual-stream artifacts.
    if current_domain() is None:
        from torch_npu.utils._graph_tree import npugraphify as original_npugraphify

        return original_npugraphify(*args, **kwargs)
    return npugraphify(*args, **kwargs)


def _ensure_baseline_inductor_setup():
    """Finish baseline Inductor setup for the running torch_npu frontend.

    v2.7.1 frontends expose a ``run_once``-guarded ``_lazy_inductor_setup``;
    older frontends apply the graph-tree binding through the patch framework
    at import time and must not re-run it (its backend registry assert is not
    idempotent), so the binding is only applied when still missing.
    """
    from torch._inductor import compile_fx
    from torch_npu.utils import _graph_tree as utils_graph_tree

    try:
        from torch_npu.utils._dynamo import _lazy_inductor_setup as lazy_setup
    except ImportError:
        lazy_setup = None
    if lazy_setup is not None:
        lazy_setup()
        return
    if compile_fx.cudagraphify is not utils_graph_tree.npugraphify:
        utils_graph_tree._apply_npugraph_tree_methods()


_compile_hooks_lock = threading.Lock()
_compile_hooks_installed = False


def install_domain_compile_hooks():
    """Opt into the experimental compile bindings, idempotently.

    Baseline Inductor setup runs first so it can never overwrite the
    experimental binding afterwards.  Only the existing
    ``compile_fx.cudagraphify`` target is rebound, plus a one-time extension
    of the Inductor reset; ``utils._graph_tree.npugraphify`` itself is never
    replaced.
    """
    global _compile_hooks_installed
    with _compile_hooks_lock:
        if _compile_hooks_installed:
            return
        _ensure_baseline_inductor_setup()
        from torch._inductor import compile_fx, cudagraph_trees

        compile_fx.cudagraphify = _inductor_routed_npugraphify
        if not getattr(
            cudagraph_trees.reset_cudagraph_trees, "_resets_npu_execution_domains", False
        ):
            cudagraph_trees.reset_cudagraph_trees = _wrap_reset(
                cudagraph_trees.reset_cudagraph_trees
            )
        _compile_hooks_installed = True
