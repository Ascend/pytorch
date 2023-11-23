from ..prof_common_func.csv_headers import CsvHeaders
from ..prof_common_func.constant import convert_us2ns


class OpSummaryBean:
    headers = []

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        row = []
        read_headers = OpSummaryBean.headers if OpSummaryBean.headers else self._data.keys()
        for field_name in read_headers:
            row.append(self._data.get(field_name, ""))
        return row

    @property
    def ts(self) -> int:
        # Time unit is ns
        ts_us = self._data.get(CsvHeaders.TASK_START_TIME, 0)
        ts_ns = convert_us2ns(ts_us)
        return ts_ns

    @property
    def all_headers(self) -> list:
        return list(self._data.keys())
