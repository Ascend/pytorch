from abc import ABCMeta, abstractmethod


class BaseViewParser(metaclass=ABCMeta):
    """
    prof_interface for viewer
    """

    def __init__(self, profiler_path: str):
        self._profiler_path = profiler_path

    @abstractmethod
    def generate_view(self, output_path: str, **kwargs) -> None:
        """
        summarize data to generate json or csv files
        Returns: None
        """
