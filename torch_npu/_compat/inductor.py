from torch_npu._compat.version import CURRENT_VERSION

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
