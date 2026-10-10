# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Shared NPU grid and Python source generation, independent of runner ABI.

Like upstream CompileResult._gen_launcher_code, callers supply the runner
arguments. Each compile-result class owns its loading, argument selection and
binding policy; these helpers do not select ordinary, static or fast launchers.
"""


def _can_clamp_1d_grid(inductor_meta, def_args, arg_names):
    """Whether a Grid1D pointwise launch can derive its grid from xnumel.

    A zero free-x-node kernel is the scalar form: its generated body has no
    per-program x index, so launching every NPU core would make all programs
    access the same scalar.  Clamp it just like the one-free-node form.
    Missing metadata remains conservative for custom/legacy kernels.
    """
    npu_num_x_nodes = inductor_meta.get("npu_num_x_nodes")
    return (
        npu_num_x_nodes in (0, 1)
        and inductor_meta.get("grid_type", "Grid1D") == "Grid1D"
        and "xnumel" in def_args
        and "R0_BLOCK" not in set(arg_names)
    )


def _gen_grid_code(scope, def_args, arg_names, cfg, inductor_meta, *, num_cores, is_a5):
    """Build NPU grid statements and any persistent grid-cache state."""
    xblock_val = cfg.kwargs.get("XBLOCK", 1)
    # Clamp grid_0 for scalar or single-free-node 1D pointwise kernels. The
    # grid expression below is derived only from xnumel, so it is not valid
    # for Grid2D/3D or reductions. Keep the separate unsplit scalar
    # reduction case from master: partial rsplit kernels must use the
    # regular dispatch path.
    npu_num_x_nodes = inductor_meta.get("npu_num_x_nodes", 0)
    grid_type = inductor_meta.get("grid_type", "Grid1D")
    npu_rsplit_partial = inductor_meta.get(
        "npu_rsplit_partial", "ws_ptr" in arg_names
    )
    is_simple_1d = _can_clamp_1d_grid(
        inductor_meta, def_args, arg_names
    )
    is_unsplit_scalar_reduction = (
        # Non-linearize bodies use pid * XBLOCK without group-dispatch
        # folding, so they must not take the grid=1 scalar shortcut.
        inductor_meta.get("npu_linearize", True)
        and npu_num_x_nodes == 0
        and grid_type == "Grid1D"
        and "R0_BLOCK" in set(arg_names)
        and not npu_rsplit_partial
    )

    # A5 (910_95) one-program-per-tile: kernel emits group_size=1/group_base=program_id
    # per free-x shape, so the launcher must launch EXACTLY total_blocks (over/under
    # aliases or drops tiles — odometer periodic modulo total_blocks). Codegen injects a
    # recipe reproducing it host-side (references XBLOCK literal + <x>numel args); over
    # 65535 coreDim folds logical→physical. Falls through to group-dispatch when absent.
    is_te_combo = inductor_meta.get("te_combo_meta") is not None
    _recipe = inductor_meta.get("npu_dispatch_recipe")
    _grid_recipe_lines = None
    if _recipe and is_a5 and not is_te_combo:
        # Splice every tile-block constexpr the recipe may reference
        # (XBLOCK / YBLOCK / ZBLOCK for Grid1D/2D/3D) as a literal, so the
        # recipe lines exec with the same block sizes the kernel was JIT'd
        # with. Missing one (e.g. YBLOCK on a Grid2D recipe) would NameError
        # in the grid computation and fail every config.
        _rl = [f"    {_bn} = {_bv}" for _bn, _bv in cfg.kwargs.items()
               if _bn.endswith("BLOCK")]
        _rl += [f"    {ln}" for ln in _recipe["lines"]]
        _tb = " * ".join(_recipe["factors"])
        _rl.append(f"    grid_0 = max(1, {_tb})")
        _grid_recipe_lines = _rl

    if is_te_combo:
        # TE combo dispatch divides one concatenated tile space across every
        # vector core inside the kernel, so it always launches the full core count.
        grid_0_expr = str(num_cores)
        grid_0_is_memoized = False
    elif _grid_recipe_lines is not None:
        # Not memoized: total_blocks depends on multiple <x>numel args, and
        # the recipe arithmetic is cheap relative to correctness clarity.
        grid_0_expr = None
        grid_0_is_memoized = False
    elif is_unsplit_scalar_reduction:
        grid_0_expr = "1"
        grid_0_is_memoized = False
    elif is_simple_1d:
        # Clamp to num_cores: below it, the full count wastes overhead on idle cores;
        # the group-dispatch body is well-defined for grid < total_thread (excess lanes
        # get group_size=0). Lower-bound at 1: an unbacked size can be 0 (speech_transformer
        # empty slice → xnumel==0) and CANN rejects coreDim==0 (EE1003); one program is a
        # correct no-op (mask all False). A5 with a free x-axis takes the recipe path.
        grid_0_expr = (
            f"max(1, min((xnumel + {xblock_val} - 1) // {xblock_val}, {num_cores}))"
        )
        # grid_0 depends only on xnumel (XBLOCK and num_cores are baked in). It's a
        # constant literal for static shapes and usually stable for dynamic ones, so
        # memoize on the last-seen xnumel with a single-slot cache: the arithmetic +
        # max/min run only when xnumel changes, still correct for any value.
        # The caller decides whether the cache is a global or a bound default.
        grid_0_is_memoized = True
    elif (
        not inductor_meta.get("npu_linearize", True)
        and grid_type == "Grid1D"
        and "xnumel" in def_args
    ):
        # Match upstream b45e5d12ea: one program handles one tile on this
        # path. Overlaunch can read past unmasked inputs; a core-count cap
        # silently drops tiles when ceil(xnumel / XBLOCK) exceeds num_cores.
        # Keep the exact tile count, without an upper clamp.
        grid_0_expr = f"max(1, (xnumel + {xblock_val} - 1) // {xblock_val})"
        grid_0_is_memoized = True
    else:
        grid_0_expr = str(num_cores)
        grid_0_is_memoized = False

    if _grid_recipe_lines is not None:
        # A5 one-program-per-tile: the injected recipe computes grid_0 as the
        # exact total_blocks (see above). Not memoized -- it depends on
        # multiple <x>numel args, so a single-slot xnumel cache would be wrong.
        grid_lines = _grid_recipe_lines
    elif grid_0_is_memoized:
        # Single-slot grid cache: [last_xnumel, last_grid_0]. The caller
        # chooses the binding policy. ``-1`` cannot equal a real xnumel, so
        # the first call always misses and populates the slot.
        scope["_grid_cache"] = [-1, 0]
        grid_lines = [
            "    if xnumel == _grid_cache[0]:",
            "        grid_0 = _grid_cache[1]",
            "    else:",
            f"        grid_0 = {grid_0_expr}",
            "        _grid_cache[0] = xnumel",
            "        _grid_cache[1] = grid_0",
        ]
    else:
        grid_lines = [f"    grid_0 = {grid_0_expr}"]
    return grid_lines


def _gen_launcher_code(scope, def_args, runner_args, grid_lines, *, bound_names=(), call_lines=None):
    """Emit a launcher using the caller's ABI, binding names and call statements."""
    scope.setdefault("max", max)
    scope.setdefault("min", min)
    hidden = "".join(f", {name}={name}" for name in bound_names if name in scope)
    if call_lines is None:
        call_lines = [f"    runner({', '.join(runner_args)})"]
    lines = [
        f"def launcher({', '.join([*def_args, 'stream'])}{hidden}):",
        *grid_lines,
        "    grid_1 = 1",
        "    grid_2 = 1",
        *call_lines,
    ]
    exec("\n".join(lines), scope)
    launcher = scope["launcher"]
    launcher._expected_positional_count = len(def_args)
    launcher._npu_def_args = list(def_args)
    return launcher
