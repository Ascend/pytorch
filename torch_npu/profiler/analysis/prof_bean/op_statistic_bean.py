class OpStatisticBean:
    HEADERS = ["OP Type", "Core Type", "Count", "Total Time(us)", "Min Time(us)", "Avg Time(us)",
               "Max Time(us)", "Ratio(%)"]

    def __init__(self, data: list):
        self._data = data

    @property
    def row(self) -> list:
        return list(self._data.values())

    @property
    def headers(self) -> list:
        return list(self._data.keys())
