from ._constant import DbConstant
from ._singleton import Singleton

__all__ = []


@Singleton
class Str2IdManager:
    def __init__(self) -> None:
        self._str_id_map = {}
        self._curr_id = 0
    
    def set_start_id(self, start_id: int):
        self._curr_id = start_id
    
    def get_id_from_str(self, string: str) -> int:
        # 先查询；有性能影响的话直接+1，不查询了
        if not string:
            return DbConstant.DB_INVALID_VALUE
        if string in self._str_id_map.keys():
            return self._str_id_map.get(string)

        res_id = self._curr_id
        self._str_id_map[string] = self._curr_id
        self._curr_id += 1
        return res_id

    def get_all_string_2_id_data(self) -> list:
        data = []
        if not self._str_id_map:
            return data
        for k, v in self._str_id_map.items():
            data.append([v, k])
        return data


@Singleton
class ConnectionIdManager:
    def __init__(self) -> None:
        self._connecion_id_map = {}
        self._curr_id = 0

    def get_id_from_connection_ids(self, connection_ids: list) -> int:
        res_id = self._curr_id
        self._connecion_id_map[self._curr_id] = connection_ids
        self._curr_id += 1
        return res_id
    
    def get_all_connection_ids(self) -> dict:
        return self._connecion_id_map


@Singleton
class CallChainIdManager:
    def __init__(self) -> None:
        self._callchain_id_map = {}
        self._curr_id = 0

    def get_callchain_id_from_callstack(self, callstack: str) -> int:
        res_id = self._curr_id
        stacks = callstack.split(";\r\n")
        for idx, stack in enumerate(stacks):
            self._callchain_id_map.setdefault(self._curr_id, []).append([Str2IdManager().get_id_from_str(stack), idx])
        self._curr_id += 1
        return res_id

    def get_all_callchain_id(self) -> dict:
        return self._callchain_id_map
