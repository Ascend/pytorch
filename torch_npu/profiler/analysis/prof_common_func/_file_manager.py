import csv
import json
import os.path

from typing import Dict, Optional
from torch_npu.utils._error_code import ErrCode, prof_error
from ....utils.path_manager import PathManager
from ..prof_common_func._constant import Constant, print_warn_msg

__all__ = []


class FileManager:
    @classmethod
    def file_read_all(cls, file_path: str, mode: str = "r") -> any:
        PathManager.check_directory_path_readable(file_path)
        if not os.path.isfile(file_path):
            return ''
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return ''
        if file_size > Constant.MAX_FILE_SIZE:
            msg = f"The file size exceeds the preset value, please check the file: {file_path}"
            print_warn_msg(msg)
            return ''
        try:
            with open(file_path, mode) as file:
                return file.read()
        except Exception as err:
            raise RuntimeError(f"Can't read file: {file_path}" + prof_error(ErrCode.UNAVAIL)) from err

    @classmethod
    def read_csv_file(cls, file_path: str, class_bean: any) -> list:
        PathManager.check_directory_path_readable(file_path)
        if not os.path.isfile(file_path):
            return []
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return []
        if file_size > Constant.MAX_CSV_SIZE:
            msg = f"The file size exceeds the preset value, please check the file: {file_path}"
            print_warn_msg(msg)
            return []
        result_data = []
        try:
            with open(file_path, newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    result_data.append(class_bean(row))
        except Exception as err:
            raise RuntimeError(f"Failed to read the file: {file_path}" + prof_error(ErrCode.UNAVAIL)) from err
        return result_data

    @classmethod
    def read_json_file(cls, file_path: str) -> Optional[Dict]:
        """Read json file and return dict data"""
        if not os.path.isfile(file_path):
            return {}
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return {}
        if file_size > Constant.MAX_FILE_SIZE:
            msg = f"The file size exceeds the preset value, please check the file: {file_path}"
            print_warn_msg(msg)
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                return data
        except Exception as err:
            raise RuntimeError(f"Failed to read the file: {file_path}" + prof_error(ErrCode.UNAVAIL)) from err

    @classmethod
    def create_csv_file(cls, output_path: str, data: list, file_name: str, headers: list = None) -> None:
        if not data:
            return
        file_path = os.path.join(output_path, file_name)
        PathManager.make_dir_safety(output_path)
        PathManager.create_file_safety(file_path)
        PathManager.check_directory_path_writeable(file_path)
        try:
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
        except Exception as err:
            raise RuntimeError(f"Can't create file: {file_path}" + prof_error(ErrCode.SYSCALL)) from err

    @classmethod
    def create_json_file(cls, output_path: str, data: list, file_name: str) -> None:
        if not data:
            return
        file_path = os.path.join(output_path, file_name)
        cls.create_json_file_by_path(file_path, data)

    @classmethod
    def create_json_file_by_path(cls, output_path: str, data: list, indent: int = None) -> None:
        if not data:
            return
        dir_name = os.path.dirname(output_path)
        PathManager.make_dir_safety(dir_name)
        PathManager.create_file_safety(output_path)
        PathManager.check_directory_path_writeable(output_path)
        try:
            with open(output_path, "w") as file:
                data = json.dumps(data, indent=indent, ensure_ascii=False)
                file.write(data)
        except Exception as err:
            raise RuntimeError(f"Can't create file: {output_path}" + prof_error(ErrCode.SYSCALL)) from err

    @classmethod
    def create_bin_file_by_path(cls, output_path: str, data: bytes) -> None:
        if not data:
            return
        dir_name = os.path.dirname(output_path)
        PathManager.make_dir_safety(dir_name)
        PathManager.create_file_safety(output_path)
        PathManager.check_directory_path_writeable(output_path)
        try:
            with os.fdopen(os.open(output_path, os.O_WRONLY, PathManager.DATA_FILE_AUTHORITY), 'wb') as file:
                file.write(data)
        except Exception as err:
            raise RuntimeError(f"Can't create file: {output_path}" + prof_error(ErrCode.SYSCALL)) from err

    @classmethod
    def append_trace_json_by_path(cls, output_path: str, data: list, new_name: str) -> None:
        PathManager.check_directory_path_writeable(output_path)
        try:
            with open(output_path, "a") as file:
                data = json.dumps(data, ensure_ascii=False)
                data = f",{data[1:]}"
                file.write(data)
        except Exception as err:
            raise RuntimeError(f"Can't create file: {output_path}" + prof_error(ErrCode.SYSCALL)) from err
        if output_path != new_name:
            os.rename(output_path, new_name)

    @classmethod
    def create_prepare_trace_json_by_path(cls, output_path: str, data: list) -> None:
        if not data:
            return
        dir_name = os.path.dirname(output_path)
        PathManager.make_dir_safety(dir_name)
        PathManager.create_file_safety(output_path)
        PathManager.check_directory_path_writeable(output_path)
        try:
            with open(output_path, "w") as file:
                data = json.dumps(data, ensure_ascii=False)
                file.write(data[:-1])
        except Exception as err:
            raise RuntimeError(f"Can't create file: {output_path}" + prof_error(ErrCode.SYSCALL)) from err

    @classmethod
    def check_db_file_vaild(cls, db_path: str) -> None:
        PathManager.check_input_file_path(db_path)
        db_size = os.path.getsize(db_path)
        if db_size < 0 or db_size > Constant.MAX_FILE_SIZE:
            raise RuntimeError(f"Invalid db file size, please check the db file: {db_path}")


class FdOpen:
    """
    creat and write file
    """

    def __init__(self: any, file_path: str, flags: int = os.O_WRONLY | os.O_CREAT, mode: int = Constant.FILE_AUTHORITY,
                 operate: str = "w", newline: str = None) -> None:
        self.file_path = file_path
        self.flags = flags
        self.newline = newline
        self.mode = mode
        self.operate = operate
        self.fd = None
        self.file_open = None

    def __enter__(self: any) -> any:
        file_dir = os.path.dirname(self.file_path)
        PathManager.check_directory_path_writeable(file_dir)

        self.fd = os.open(self.file_path, self.flags, self.mode)
        if self.newline is None:
            self.file_open = os.fdopen(self.fd, self.operate)
        else:
            self.file_open = os.fdopen(self.fd, self.operate, newline=self.newline)
        return self.file_open

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_open:
            self.file_open.close()
        elif self.fd:
            os.close(self.fd)

