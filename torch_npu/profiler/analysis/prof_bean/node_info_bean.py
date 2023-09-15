class NodeInfoBean:
    def __init__(self, kernel_list: list):
        self._device_dur = 0
        self._device_dur_with_ai_core = 0
        self._kernel_list = kernel_list
        self._min_start = None
        self._max_end = None
        self._init()

    def _init(self):
        self._device_dur = sum([kernel.dur for kernel in self._kernel_list])
        self._device_dur_with_ai_core = sum([kernel.dur for kernel in self._kernel_list if kernel.is_ai_core])
        self._min_start = min([kernel.ts for kernel in self._kernel_list])
        self._max_end = max([kernel.ts + kernel.dur for kernel in self._kernel_list])

    @property
    def kernel_list(self) -> float:
        return self._kernel_list

    @property
    def device_dur(self) -> float:
        return self._device_dur

    @property
    def device_dur_with_ai_core(self) -> float:
        return self._device_dur_with_ai_core

    @property
    def min_start(self) -> float:
        return self._min_start

    @property
    def max_end(self) -> float:
        return self._max_end
