from __future__ import annotations

import logging

import torch


def patch_async_compile():
    from .codecache import CATLASSCodeCache

    log = logging.getLogger("torch._inductor")

    def catlass(self, source_code, dst_file_ext, aot_compile=False, is_mix=False):
        log.info("CATLASS Kernel:\n%s", source_code)

        def task():
            if aot_compile:
                # We rely on JITInductor to compile the CUDA code,
                # so that we can load it into AOTInductor.
                CATLASSCodeCache.compile(source_code, "o", is_mix=is_mix)
            return CATLASSCodeCache.load(source_code, dst_file_ext, is_mix)[0]

        return self.submit(task)

    torch._inductor.async_compile.AsyncCompile.catlass = catlass
