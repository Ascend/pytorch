from ._common_bean import CommonBean

__all__ = []


class AiCpuBean(CommonBean):
    HEADERS = ["Timestamp", "Node", "Compute_time(ms)", "Memcpy_time(us)", "Task_time(ms)", "Dispatch_time(ms)",
               "Total_time(ms)", "Stream ID", "Task ID"]

    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def row(self) -> list:
        return list(self._data.values())

    @property
    def headers(self) -> list:
        return list(self._data.keys())
