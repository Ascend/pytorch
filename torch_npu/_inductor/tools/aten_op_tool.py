"""
this tool is to compare the difference of lowerings.fallbacks between Pytorch native and torch_npu.
It is helpful to set FALLBACK_LIST in lowering_fallback_list.py.
"""

from typing import Set

from torch._inductor import lowering
from torch._ops import OpOverload, OpOverloadPacket


def write_to_file(items: Set[str], file_name: str):
    with open(file_name, "w") as f:
        for item in items:
            f.write(f"{item},\n") 
            

def get_base_packet(op):
    if isinstance(op, OpOverload):
        return op.overloadpacket
    return op


def get_overload_set(inputs, output_set):
    for fn in inputs:
        if isinstance(fn, OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                output_set.add(other_fn)
        else:
            output_set.add(fn)
            
                                                                                                                                            
def get_native_fallbacks_base() -> Set[str]:
    native_fallbacks = {str(op) for op in lowering.fallbacks}
    write_to_file(sorted(native_fallbacks), f"native_fallbacks_{len(native_fallbacks)}.txt")
                                                                                                                                                        
    native_fallbacks_base = {str(get_base_packet(op)) for op in lowering.fallbacks}
    write_to_file(sorted(native_fallbacks_base), f"native_fallbacks_base_{len(native_fallbacks_base)}.txt")
                                                                                                                                                                                         
    print(f"[native torch] len(lowering.fallbacks): {len(lowering.fallbacks)}, len(native_fallbacks_base): {len(native_fallbacks_base)}")
    return native_fallbacks, native_fallbacks_base
                                                                                                                                                                                                 
    
def get_npu_fallbacks_base() -> Set[str]:
    import torch_npu._inductor
    
    npu_fallbacks = {str(op) for op in lowering.fallbacks}
    write_to_file(sorted(npu_fallbacks), f"npu_fallbacks_{len(npu_fallbacks)}.txt")
    
    npu_fallbacks_base = {str(get_base_packet(op)) for op in lowering.fallbacks}
    write_to_file(sorted(npu_fallbacks_base), f"npu_fallbacks_base_{len(npu_fallbacks_base)}.txt")                                                                                                                                                                                                                     
    
    print(f"[torch_npu] len(lowering.fallbacks): {len(lowering.fallbacks)}, len(npu_fallbacks_base): {len(npu_fallbacks_base)}")
    return npu_fallbacks, npu_fallbacks_base

                                                                                                                                                                                                                                 
def npu_extra_fallbacks_base_diff(native_fallbacks_base, npu_fallbacks_base) -> None:
    npu_extra_fallbacks_base = npu_fallbacks_base - native_fallbacks_base
    write_to_file(sorted(npu_extra_fallbacks_base), f"npu_extra_fallbacks_base_{len(npu_extra_fallbacks_base)}.txt")
    print(f"[npu extra] len(npu_extra_fallbacks_base): {len(npu_extra_fallbacks_base)}")


def npu_extra_fallbacks_diff(native_fallbacks, npu_fallbacks) -> None:
    npu_extra_fallbacks = npu_fallbacks - native_fallbacks
    write_to_file(sorted(npu_extra_fallbacks), f"npu_extra_fallbacks_{len(npu_extra_fallbacks)}.txt")
    print(f"[npu extra] len(npu_extra_fallbacks): {len(npu_extra_fallbacks)}")


if __name__ == "__main__":
    native_fallbacks, native_fallbacks_base = get_native_fallbacks_base()
    npu_fallbacks, npu_fallbacks_base = get_npu_fallbacks_base()
    npu_extra_fallbacks_base_diff(native_fallbacks_base, npu_fallbacks_base)
    npu_extra_fallbacks_diff(native_fallbacks, npu_fallbacks)