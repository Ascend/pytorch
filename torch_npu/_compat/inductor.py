from torch_npu._compat.version import CURRENT_VERSION


# COMPAT(>= 2.11): CachingAutotuner moved from runtime.triton_heuristics to triton_heuristics.
# Lazy function (not module-level import): torch._inductor.triton_heuristics does not exist
# in CPU-only builds, so importing at module load time would fail.
# CAN REMOVE when MIN_SUPPORTED >= (2, 11): import from triton_heuristics directly
def get_CachingAutotuner():
    if CURRENT_VERSION >= (2, 11):
        from torch._inductor.triton_heuristics import CachingAutotuner
    else:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    return CachingAutotuner


# COMPAT(>= 2.11): sizevars.var_to_val renamed to sizevars.backed_var_to_val.
# CAN REMOVE when MIN_SUPPORTED >= (2, 11): use sizevars.backed_var_to_val directly
def get_sizevars_backed_var_to_val(sizevars):
    if hasattr(sizevars, "backed_var_to_val"):
        return sizevars.backed_var_to_val
    return sizevars.var_to_val


# COMPAT(>= 2.11): gen_common_triton_imports changed from module-level function
#   to instance method on the kernel object.
# CAN REMOVE when MIN_SUPPORTED >= (2, 11): call kernel.gen_common_triton_imports() directly
def gen_common_triton_imports(kernel):
    if CURRENT_VERSION >= (2, 11):
        return kernel.gen_common_triton_imports()
    from torch._inductor.codegen.triton import gen_common_triton_imports as _fn
    return _fn()
