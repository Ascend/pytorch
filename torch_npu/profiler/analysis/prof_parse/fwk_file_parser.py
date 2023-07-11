import os
import re

from ..prof_common_func.binary_decoder import BinaryDecoder
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.file_tag import FileTag
from ..prof_common_func.path_manager import PathManager
from ..prof_common_func.tlv_decoder import TLVDecoder
from ..prof_config.fwk_file_parser_config import FwkFileParserConfig


class FwkFileParser:
    def __init__(self, profiler_path: str):
        self._fwk_path = PathManager.get_fwk_path(profiler_path)
        self._file_list = {}
        self._file_dispatch()

    def get_file_data_by_tag(self, file_tag: int) -> list:
        file_path = self._file_list.get(file_tag)
        if not file_path:
            return []
        all_bytes = FileManager.file_read_all(file_path, "rb")
        file_bean = FwkFileParserConfig.FILE_BEAN_MAP.get(file_tag, {}).get("bean")
        is_tlv = FwkFileParserConfig.FILE_BEAN_MAP.get(file_tag, {}).get("is_tlv")
        struct_size = FwkFileParserConfig.FILE_BEAN_MAP.get(file_tag, {}).get("struct_size")
        if is_tlv:
            return TLVDecoder.decode(all_bytes, file_bean, struct_size)
        else:
            return BinaryDecoder.decode(all_bytes, file_bean, struct_size)

    def get_task_queue_data(self) -> tuple:
        enqueue_data_list, dequeue_data_list = [], []
        op_mark_data = self.get_file_data_by_tag(FileTag.OP_MARK)
        if not op_mark_data:
            return enqueue_data_list, dequeue_data_list
        op_mark_data.sort(key=lambda x: x.time_us)
        enqueue_start, dequeue_start = None, None
        for op_mark in op_mark_data:
            if op_mark.is_enqueue_start:
                enqueue_start = op_mark
            elif op_mark.is_enqueue_end:
                if enqueue_start and enqueue_start.tid == op_mark.tid and \
                        enqueue_start.origin_name == op_mark.origin_name:
                    op_mark.ts = enqueue_start.time_us
                    op_mark.dur = op_mark.time_us - enqueue_start.time_us
                    enqueue_data_list.append(op_mark)
                    enqueue_start = None
            elif op_mark.is_dequeue_start:
                dequeue_start = op_mark
            elif op_mark.is_dequeue_end:
                if dequeue_start and dequeue_start.corr_id == op_mark.corr_id:
                    dequeue_start.ts = dequeue_start.time_us
                    dequeue_start.dur = op_mark.time_us - dequeue_start.time_us
                    dequeue_data_list.append(dequeue_start)
                    dequeue_start = None
        return enqueue_data_list, dequeue_data_list

    def _file_dispatch(self):
        for file_name in os.listdir(os.path.realpath(self._fwk_path)):
            file_path = os.path.join(self._fwk_path, file_name)
            if not os.path.isfile(file_path):
                continue
            for file_tag, pattern in FwkFileParserConfig.FILE_DISPATCH_MAP.items():
                if re.match(pattern, file_name):
                    self._file_list.setdefault(file_tag, file_path)
