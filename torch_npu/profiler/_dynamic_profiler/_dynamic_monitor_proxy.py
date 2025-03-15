from ..analysis.prof_common_func._singleton import Singleton
from ._dynamic_profiler_utils import DynamicProfilerUtils


@Singleton
class PyDynamicMonitorProxySingleton():
    def __init__(self):
        self._proxy = None
        self._load_proxy()

    def _load_proxy(self):
        if not self._proxy:
            try:
                from IPCMonitor import PyDynamicMonitorProxy
                self._proxy = PyDynamicMonitorProxy()
            except Exception as e:
                dynamic_profiler_utils.stdout_log(f"Import IPCMonitro module failed :{e}!",
                                                dynamic_profiler_utils.LoggerLevelEnum.WARNING)

    def get_proxy(self):
        return self._proxy