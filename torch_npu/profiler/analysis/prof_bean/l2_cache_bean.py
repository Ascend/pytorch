class L2CacheBean:
    HEADERS = ["Stream Id", "Task Id", "Hit Rate", "Victim Rate", "Op Name"]

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        return list(self._data.values())

    @property
    def headers(self) -> list:
        return list(self._data.keys())
