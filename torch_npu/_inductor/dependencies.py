from typing import Any, Callable, Sequence

import sympy
import torch._inductor.dependencies as dependencies

origin_extract_read_writes = dependencies.extract_read_writes

def patch_extract_read_writes():
    def extract_read_writes(
            fn: Callable[..., Any],
            *argsizes: Sequence[sympy.Expr],
            normalize: bool = False,
            prefix: str = "d",
            hidden_args: Sequence[list[sympy.Expr]] = (),
    ) -> dependencies.ReadWrites:
        # NPU does not support normalize load/store
        return origin_extract_read_writes(
            fn,
            *argsizes,
            normalize=False,
            prefix=prefix,
            hidden_args=hidden_args
        )

    dependencies.extract_read_writes = extract_read_writes
    import torch._inductor.ir as ir
    import torch._inductor.scheduler as scheduler
    ir.extract_read_writes = extract_read_writes
    scheduler.extract_read_writes = extract_read_writes
