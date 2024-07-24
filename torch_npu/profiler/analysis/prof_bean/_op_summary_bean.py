from ._common_bean import CommonBean
from ..prof_common_func._csv_headers import CsvHeaders

__all__ = []


class OpSummaryBean(CommonBean):
    headers = []

    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def row(self) -> list:
        row = []
        read_headers = OpSummaryBean.headers if OpSummaryBean.headers else self._data.keys()
        for field_name in read_headers:
            row.append(self._data.get(field_name, ""))
        return row

    @property
    def ts(self) -> str:
        # Time us str
        return self._data.get(CsvHeaders.TASK_START_TIME, "0")

    @property
    def all_headers(self) -> list:
        return list(self._data.keys())
