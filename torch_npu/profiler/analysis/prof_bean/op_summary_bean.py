class OpSummaryBean:
    TASK_START_TIME = "Task Start Time"
    headers = []

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        row = []
        read_headers = OpSummaryBean.headers if OpSummaryBean.headers else self._data.keys()
        for field_name in read_headers:
            if field_name == self.TASK_START_TIME:
                row.append(float(self._data.get(field_name, 0)) / 1000)
            else:
                row.append(self._data.get(field_name, ""))
        return row

    @property
    def ts(self) -> float:
        return float(self._data.get(self.TASK_START_TIME, 0)) / 1000

    @property
    def all_headers(self) -> list:
        return list(self._data.keys())
