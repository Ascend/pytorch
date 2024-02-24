import multiprocessing

__all__ = []


class ProfProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        return
