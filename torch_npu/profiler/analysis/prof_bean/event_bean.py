from ..prof_common_func.constant import Constant


class EventBean:
    HOST_TO_DEVICE = "HostToDevice"
    START_FLOW = "s"
    END_FLOW = "f"

    def __init__(self, data: dict):
        self._origin_data = data

    @property
    def ts(self) -> float:
        return self._origin_data.get("ts", 0)

    @property
    def pid(self) -> str:
        return self._origin_data.get("pid", "")

    @property
    def tid(self) -> int:
        return self._origin_data.get("tid", 0)

    @property
    def dur(self) -> float:
        return self._origin_data.get("dur", 0)

    @property
    def name(self) -> str:
        return self._origin_data.get("name", "")

    @property
    def id(self) -> any:
        return self._origin_data.get("id")

    @property
    def unique_id(self) -> str:
        return "{}-{}-{}".format(self.pid, self.tid, self.ts)

    @property
    def is_ai_core(self) -> bool:
        args = self._origin_data.get("args")
        if args:
            return args.get("Task Type") == Constant.AI_CORE
        return False

    def is_flow_start_event(self) -> bool:
        return self._origin_data.get("cat") == self.HOST_TO_DEVICE and self._origin_data.get("ph") == self.START_FLOW

    def is_flow_end_event(self) -> bool:
        return self._origin_data.get("cat") == self.HOST_TO_DEVICE and self._origin_data.get("ph") == self.END_FLOW

    def is_x_event(self) -> bool:
        return self._origin_data.get("ph") == "X"
