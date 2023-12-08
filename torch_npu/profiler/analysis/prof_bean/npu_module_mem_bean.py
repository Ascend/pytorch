from ..prof_common_func.constant import Constant


class NpuModuleMemoryBean:
    SHOW_HEADERS = ["Component", "Timestamp(us)", "Total Reserved(MB)", "Device"]

    def __init__(self, data: dict):
        total_reverved = float(data.get("Total Reserved(KB)", 0))
        data["Total Reserved(MB)"] = str(total_reverved / Constant.KB_TO_MB)
        self._data = data

    @property
    def row(self) -> list:
        return [self._data.get(field_name, "") for field_name in self.SHOW_HEADERS]

    @property
    def headers(self) -> list:
        return self.SHOW_HEADERS
