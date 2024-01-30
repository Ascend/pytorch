from ..prof_common_func.constant import Constant


class NpuMemoryBean:
    SHOW_HEADERS = ["event", "timestamp(us)", "allocated(KB)", "memory(KB)", "active"]

    def __init__(self, data: list):
        self._data = data

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
