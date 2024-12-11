import atexit
from concurrent import futures

from ._singleton import Singleton


@Singleton
class MultiProcessPool:
    def __init__(self):
        self._pool = None
        atexit.register(self.close_pool)

    def init_pool(self, max_workers=4):
        if not self._pool:
            self._pool = futures.ProcessPoolExecutor(max_workers=max_workers)
        return self._pool

    def close_pool(self, wait: bool = True):
        if self._pool:
            self._pool.shutdown(wait=wait)
            self._pool = None

    def submit_task(self, func, *args, **kwargs):
        if not self._pool:
            return
        self._pool.submit(func, *args, **kwargs)
