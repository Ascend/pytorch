import os

from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.global_var import GlobalVar
from ..prof_view.base_view_parser import BaseViewParser
from ..prof_common_func.constant import Constant


class StackViewParser(BaseViewParser):
    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

    def generate_view(self, output_path: str, **kwargs) -> None:
        if not GlobalVar.torch_op_tree_node:
            return
        metric = kwargs.get("metric")
        with os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT, Constant.FILE_AUTHORITY), "w") as f:
            for torch_op_node in GlobalVar.torch_op_tree_node:
                call_stack = torch_op_node.call_stack
                if not call_stack:
                    continue
                # remove ‘\n’ for each stack frame
                call_stack = ";".join(map(lambda x: x.strip(), call_stack.split(";")))
                if metric == "self_cpu_time_total":
                    total_dur = torch_op_node.host_total_dur
                else:
                    total_dur = torch_op_node.device_total_dur
                total_dur = round(float(total_dur))
                f.write(call_stack + " " + str(total_dur) + "\n")

