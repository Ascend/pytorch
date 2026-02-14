"""
this tool is to get and save FALLBACK_LIST in lowering_fallback_list.py
"""

from torch_npu._inductor import lowering_fallback_list
from torch_npu._inductor.tools.aten_op_tool import write_to_file


def get_npu_fallback_list():
    npu_fallback_list = {str(op) for op in lowering_fallback_list.FALLBACK_LIST}
    print(f"len(lowering_fallback_list.FALLBACK_LIST): {len(lowering_fallback_list.FALLBACK_LIST)}")
    return npu_fallback_list


if __name__ == "__main__": 
    npu_fallback_list = get_npu_fallback_list()
    write_to_file(sorted(npu_fallback_list), f"npu_fallback_list_{len(npu_fallback_list)}.txt")