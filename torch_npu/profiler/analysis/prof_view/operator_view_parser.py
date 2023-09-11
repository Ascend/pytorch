from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.global_var import GlobalVar
from ..prof_view.base_view_parser import BaseViewParser


class OperatorViewParser(BaseViewParser):
    OPERATOR_HEADERS = ["Name", "Input Shapes", "Call Stack", "Host Self Duration(us)", "Host Total Duration(us)",
                        "Device Self Duration(us)", "Device Total Duration(us)", "Device Self Duration With AICore(us)",
                        "Device Total Duration With AICore(us)"]
    OPERATOR_VIEW = "operator_details.csv"

    def __init__(self, profiler_path: str):
        super().__init__(profiler_path)

    def generate_view(self, output_path: str, **kwargs) -> None:
        if not GlobalVar.torch_op_tree_node:
            return
        operator_list = [None] * len(GlobalVar.torch_op_tree_node)
        index = 0
        for torch_op_node in GlobalVar.torch_op_tree_node:
            if torch_op_node.is_profiler_step():
                continue
            operator_list[index] = [torch_op_node.event.name, torch_op_node.input_shape, torch_op_node.call_stack,
                                    torch_op_node.host_self_dur, torch_op_node.host_total_dur,
                                    torch_op_node.device_self_dur, torch_op_node.device_total_dur,
                                    torch_op_node.device_self_dur_with_ai_core,
                                    torch_op_node.device_total_dur_with_ai_core]
            index += 1
        del operator_list[index:]
        FileManager.create_csv_file(output_path, operator_list, self.OPERATOR_VIEW, self.OPERATOR_HEADERS)
