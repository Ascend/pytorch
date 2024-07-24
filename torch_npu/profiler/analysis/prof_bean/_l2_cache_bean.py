from ._common_bean import CommonBean

__all__ = []


class L2CacheBean(CommonBean):
    HEADERS = ["Stream Id", "Task Id", "Hit Rate", "Victim Rate", "Op Name"]

    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def row(self) -> list:
        return list(self._data.values())

    @property
    def headers(self) -> list:
        return list(self._data.keys())
