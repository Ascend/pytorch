class OpSummaryBean:
    SHOW_HEADERS = ["Op Name", "OP Type", "Task Type", "Task Start Time", "Task Duration(us)", "Task Wait Time(us)",
                    "Block Dim", "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
                    "Output Data Types", "Output Formats"]
    SAVE_HEADERS = ["Name", "Type", "Accelerator Core", "Start Time", "Duration(us)", "Wait Time(us)",
                    "Block Dim", "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
                    "Output Data Types", "Output Formats"]

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        row = []
        for field_name in self.SHOW_HEADERS:
            row.append(self._data.get(field_name, ""))
        return row

    @classmethod
    def headers(cls, step_id: list) -> list:
        step = ["Step Id"] if step_id else []
        return step + cls.SAVE_HEADERS
