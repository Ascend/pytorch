from __future__ import annotations

from typing import Any

from torch._inductor.virtualized import V

from .codegen import build_callsite_metadata


class FastLaunchWrapperEmitter:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.callsite_index = 0

    def emit_triton_call(
        self,
        *,
        kernel_name: str,
        call_args: Any,
        triton_meta: Any,
        call_args_str: str,
        stream_name: str,
        indent: str = "",
    ) -> None:
        index = self.callsite_index
        self.callsite_index += 1
        graph_id = str(
            getattr(V.graph, "graph_id", None)
            or getattr(V.graph, "name", None)
            or "graph"
        )
        metadata = build_callsite_metadata(
            kernel_name=kernel_name,
            call_args=call_args,
            triton_meta=triton_meta,
            graph_id=graph_id,
            callsite_index=index,
        )
        metadata_name = f"_npu_fast_launch_metadata_{index}"
        call_slot = f"_npu_fast_launch_call_{index}"
        call_local = f"{call_slot}_fn"
        self.owner.header.writeline(f"{metadata_name} = {metadata!r}")
        self.owner.header.writeline(f"{call_slot} = [None]")
        self.owner.writeline(f"{indent}{call_local} = {call_slot}[0]")
        self.owner.writeline(f"{indent}if {call_local} is None:")
        self.owner.writeline(
            f"{indent}    {call_local} = bind_python_wrapper_kernel_fast("
            f"{metadata_name}, {kernel_name}, call_slot={call_slot})"
        )
        self.owner.writeline(f"{indent}    {call_slot}[0] = {call_local}")
        args_prefix = f"{call_args_str}, " if call_args_str else ""
        self.owner.writeline(f"{indent}{call_local}({args_prefix}stream={stream_name})")


__all__ = ["FastLaunchWrapperEmitter"]
