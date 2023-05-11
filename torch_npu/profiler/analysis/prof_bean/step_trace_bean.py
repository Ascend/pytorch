class StepTraceBean:
    def __init__(self, data: dict):
        self._data = data

    @property
    def step_id(self) -> int:
        return int(self._data.get("Iteration ID", -1))
