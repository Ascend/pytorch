from ..prof_common_func.constant import Constant


class GeOpMemoryBean:
    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        return [self.name, self.size, self.allocation_time, self.release_time, self.dur,
                self.allocation_total_allocated, self.allocation_total_reserved, self.release_total_allocated,
                self.release_total_reserved, self.device]

    @property
    def name(self):
        return "cann::" + self._data.get("Name")

    @property
    def size(self):
        return self._data.get("Size(KB)")

    @property
    def allocation_time(self):
        return self._data.get("Allocation Time(us)")

    @property
    def dur(self):
        return self._data.get("Duration(us)")

    @property
    def release_time(self):
        if self.allocation_time and self.dur:
            return float(self.allocation_time) + float(self.dur)
        return None

    @property
    def allocation_total_allocated(self):
        size_kb = self._data.get("Allocation Total Allocated(KB)")
        return float(size_kb) / Constant.KB_TO_MB if size_kb else None

    @property
    def allocation_total_reserved(self):
        size_kb = self._data.get("Allocation Total Reserved(KB)")
        return float(size_kb) / Constant.KB_TO_MB if size_kb else None

    @property
    def release_total_allocated(self):
        size_kb = self._data.get("Release Total Allocated(KB)")
        return float(size_kb) / Constant.KB_TO_MB if size_kb else None

    @property
    def release_total_reserved(self):
        size_kb = self._data.get("Release Total Reserved(KB)")
        return float(size_kb) / Constant.KB_TO_MB if size_kb else None

    @property
    def device(self):
        return self._data.get("Device")
