__all__ = []

from ._common_bean import CommonBean


class NicBean(CommonBean):

    def __init__(self, data: dict):
        super().__init__(data)

    @property
    def row(self) -> list:
        return list(self._data.values())

    @property
    def headers(self) -> list:
        return list(self._data.keys())
