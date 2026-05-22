from itertools import count
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V

from ...npu.codegen.meta_kernel import NpuMetaKernel, NpuMetaScheduling


id_iter = count()


class AkgKernel(NpuMetaKernel):

    def call_kernel(self, name: str, node=None):
        wrapper = V.graph.wrapper_code
        call_args = self.get_call_args()

        if len(call_args) > 0:
            wrapper.generate_kernel_call(
                name,
                call_args,
            )


class AkgScheduling(NpuMetaScheduling):
    meta_kernel_type = AkgKernel

    def __init__(self, sched: Scheduler):
        super().__init__(sched)

    def _get_compile_api(self) -> str:
        return "akg_auto_fallback"
