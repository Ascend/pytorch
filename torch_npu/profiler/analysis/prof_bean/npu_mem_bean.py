from .common_bean import CommonBean
from ..prof_common_func.constant import Constant

__all__ = []


class NpuMemoryBean(CommonBean):
    SHOW_HEADERS = ["event", "timestamp(us)", "allocated(KB)", "memory(KB)", "active", "stream_ptr"]

    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def row(self) -> list:
        row = []
        if self._data.get("event") != Constant.APP:
            return row
        for field_name in self.SHOW_HEADERS:
            if field_name == "memory(KB)":
                row.append(float(self._data.get(field_name, 0)) / Constant.KB_TO_MB)
            else:
                row.append(self._data.get(field_name, ""))
        return row
