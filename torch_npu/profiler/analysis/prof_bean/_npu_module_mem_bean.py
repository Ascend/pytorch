from ._common_bean import CommonBean
from ..prof_common_func._constant import Constant

__all__ = []


class NpuModuleMemoryBean(CommonBean):
    SHOW_HEADERS = ["Component", "Timestamp(us)", "Total Reserved(MB)", "Device"]

    def __init__(self, data: dict):
        super().__init__(data)
        total_reverved = float(data.get("Total Reserved(KB)", 0))
        self._data["Total Reserved(MB)"] = str(total_reverved / Constant.KB_TO_MB)

    @property
    def row(self) -> list:
        return [self._data.get(field_name, "") for field_name in self.SHOW_HEADERS]

    @property
    def headers(self) -> list:
        return self.SHOW_HEADERS
