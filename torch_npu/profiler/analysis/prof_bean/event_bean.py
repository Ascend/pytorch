from typing import Union

from ..prof_common_func.constant import Constant
from ..prof_common_func.constant import convert_us2ns


class EventBean:

    def __init__(self, data: dict):
        self._origin_data = data

    @property
    def ts(self) -> int:
        # Time unit is ns
        ts_us = self._origin_data.get("ts", 0)
        ts_ns = convert_us2ns(ts_us)
        return ts_ns

    @property
    def pid(self) -> str:
        return self._origin_data.get("pid", "")

    @property
    def tid(self) -> int:
        return self._origin_data.get("tid", 0)

    @property
    def dur(self) -> Union[float, str]:
        # Time unit is us
        return self._origin_data.get("dur", 0)

    @property
    def end_ns(self) -> int:
        # Time unit is ns
        start = convert_us2ns(self._origin_data.get("ts", 0))
        dur = int(self._origin_data.get("dur", 0) * 1000)
        return int(start + dur)

    @property
    def name(self) -> str:
        return self._origin_data.get("name", "")

    @property
    def id(self) -> any:
        return self._origin_data.get("id")

    @property
    def unique_id(self) -> str:
        return f"{self.pid}-{self.tid}-{self.ts}"

    @property
    def is_ai_core(self) -> bool:
        args = self._origin_data.get("args")
        if args:
            return args.get("Task Type") == Constant.AI_CORE
        return False
