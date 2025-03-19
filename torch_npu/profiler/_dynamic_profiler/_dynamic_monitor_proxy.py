from ..analysis.prof_common_func._singleton import Singleton
from ._dynamic_profiler_utils import DynamicProfilerUtils


@Singleton
class PyDynamicMonitorProxySingleton():
    def __init__(self):
        self._proxy = None
        self._load_success = True

    def _load_proxy(self):
        if not self._proxy and self._load_success:
            try:
                from IPCMonitor import PyDynamicMonitorProxy
            except Exception as e:
                self._load_success = False
                return
            self._proxy = PyDynamicMonitorProxy()

    def get_proxy(self):
        self._load_proxy()
        return self._proxy