import csv
import json
import os.path

from ....utils.path_manager import PathManager
from ..prof_common_func.constant import Constant, print_warn_msg


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
            raise RuntimeError(f"Can't read file: {file_path}") from err

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
            raise RuntimeError(f"Failed to read the file: {file_path}") from err
        return result_data

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
            raise RuntimeError(f"Can't create file: {file_path}") from err

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
            raise RuntimeError(f"Can't create file: {output_path}") from err

    @classmethod
    def append_trace_json_by_path(cls, output_path: str, data: list, new_name: str) -> None:
        try:
            with open(output_path, "a") as file:
                data = json.dumps(data, ensure_ascii=False)
                data = f",{data[1:]}"
                file.write(data)
        except Exception as err:
            raise RuntimeError(f"Can't create file: {output_path}") from err
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
            raise RuntimeError(f"Can't create file: {output_path}") from err
