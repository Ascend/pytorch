from torch._inductor import config
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.wrapper import (
    NpuMlirWrapperCodeGen,
)
from torch_npu._inductor.triton_experimental.codegen.wrapper import NPUWrapperCodeGen


class NpuDvmWrapperCodeGen(NpuMlirWrapperCodeGen):
    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        if is_subgraph:
            return NpuMlirWrapperCodeGen.create(
                is_subgraph, subgraph_name, parent_wrapper, partition_signatures
            )
        return NpuDvmWrapperCodeGen()

    def write_header(self):
        super().write_header()
        self.header.writeline("empty_strided_npu = torch_npu._C._empty_strided_npu")

    def make_allocation(
        self, name, device, dtype, shape, stride, allocation_shape=None, is_pinned=False
    ):
        if device.type != "npu" or config.test_configs.track_memory_lifecycle or is_pinned:
            return super().make_allocation(
                name, device, dtype, shape, stride, allocation_shape, is_pinned
            )
        return NPUWrapperCodeGen.make_allocation(
            self, name, device, dtype, shape, stride, allocation_shape, is_pinned
        )
