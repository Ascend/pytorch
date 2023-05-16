class EventBean:
    def __init__(self, data: dict):
        self._origin_data = data

    @property
    def ts(self) -> float:
        return self._origin_data.get("ts")

    @property
    def corr_id(self) -> int:
        return self._origin_data.get("corr_id")
