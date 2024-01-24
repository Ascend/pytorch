from ..prof_common_func.constant import Constant
from ..prof_common_func.constant import convert_ns2us_str
from ..prof_common_func.constant import convert_us2ns


class GeMemoryRecordBean:

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        return [Constant.GE, convert_ns2us_str(self.time_ns, "\t"), self.total_allocated,
                self.total_reserved, None, self.device_tag]

    @property
    def component(self) -> str:
        return self._data.get("Component")

    @property
    def time_ns(self) -> int:
        ts_us = self._data.get("Timestamp(us)")
        ts_ns = convert_us2ns(ts_us)
        return ts_ns

    @property
    def total_allocated(self) -> float:
        return float(self._data.get("Total Allocated(KB)", 0)) / Constant.KB_TO_MB

    @property
    def total_reserved(self) -> float:
        return float(self._data.get("Total Reserved(KB)", 0)) / Constant.KB_TO_MB

    @property
    def total_active(self) -> float:
        return 0

    @property
    def device_tag(self) -> float:
        return self._data.get("Device", "")
