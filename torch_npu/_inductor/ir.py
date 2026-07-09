from torch._inductor.virtualized import V
from torch._inductor import config


def patch_extern_kernel_codegen_size_asserts():
    from torch._inductor.ir import ExternKernel
    from . import config as npu_config
    original_codegen_size_asserts = ExternKernel.codegen_size_asserts

    def npu_codegen_size_asserts(self, wrapper):
        fx_node = getattr(self, 'fx_node', None)
        should_skip = False
        if fx_node and fx_node.target:
            skip_config = getattr(npu_config, 'skip_specific_stride_asserts', [])
            if isinstance(skip_config, (list, tuple)):
                should_skip = fx_node.target in skip_config
        if should_skip:
            if config.size_asserts and not V.graph.cpp_wrapper:
                from torch._inductor.utils import sympy_product
                if sympy_product(self.get_size()) == 0:
                    return
                wrapper.writeline(
                    f"# NPU: Skipping stride assertion for {fx_node.target}"
                )
        else:
            original_codegen_size_asserts(self, wrapper)

    ExternKernel.codegen_size_asserts = npu_codegen_size_asserts
