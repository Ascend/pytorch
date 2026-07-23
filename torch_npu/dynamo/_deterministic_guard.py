def install_npu_deterministic_level_guard() -> bool:
    """Guard the NPU deterministic_level to trigger recompilation when level changes."""
    import torch._guards as _guards
    from torch._dynamo.guards import get_verbose_code_parts
    from torch._dynamo.source import GlobalStateSource
    import torch_npu

    tc = _guards.TracingContext.try_get()
    if tc is None:
        return False

    captured_level = torch_npu.npu._get_deterministic_level()

    def _create_guard_fn(builder, guard):
        code = [f"torch_npu.npu._get_deterministic_level() == {captured_level}"]

        def check_fn(_):
            return torch_npu.npu._get_deterministic_level() == captured_level

        builder.guard_manager.root.add_lambda_guard(
            check_fn,
            get_verbose_code_parts(code, guard),
        )

    tc.guards_context.dynamo_guards.add(GlobalStateSource().make_guard(_create_guard_fn))
    return True
