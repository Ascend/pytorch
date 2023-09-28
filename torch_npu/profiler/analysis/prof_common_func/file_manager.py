import csv
import json
import os.path
import shutil
from warnings import warn

from ..prof_common_func.constant import Constant


class FileManager:
    @classmethod
    def file_read_all(cls, file_path: str, mode: str = "r") -> any:
        if not os.path.isfile(file_path):
            return ''
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return ''
        if file_size > Constant.MAX_FILE_SIZE:
            warn(f"The file size exceeds the preset value {Constant.MAX_FILE_SIZE / 1024 / 1024}MB, "
                 f"please check the file: {file_path}")
            return ''
        try:
            with open(file_path, mode) as file:
                return file.read()
        except Exception:
            raise RuntimeError(f"Can't read file: {file_path}")

    @classmethod
    def read_csv_file(cls, file_path: str, class_bean: any) -> list:
        if not os.path.isfile(file_path):
            return []
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return []
        if file_size > Constant.MAX_CSV_SIZE:
            warn(f"The file size exceeds the preset value {Constant.MAX_CSV_SIZE / 1024 / 1024}MB, "
                 f"please check the file: {file_path}")
            return []
        result_data = []
        try:
            with open(file_path, newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    result_data.append(class_bean(row))
        except Exception:
            raise RuntimeError(f"Failed to read the file: {file_path}")
        return result_data

    @classmethod
    def create_csv_file(cls, output_path: str, data: list, file_name: str, headers: list = None) -> None:
        if not data:
            return
        file_path = os.path.join(output_path, file_name)
        try:
            with os.fdopen(os.open(file_path, os.O_WRONLY | os.O_CREAT, Constant.FILE_AUTHORITY), "w",
                           newline="") as file:
                writer = csv.writer(file)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
        except Exception:
            raise RuntimeError(f"Can't create file: {file_path}")

    @classmethod
    def create_json_file(cls, output_path: str, data: list, file_name: str) -> None:
        if not data:
            return
        file_path = os.path.join(output_path, file_name)
        cls.create_json_file_by_path(file_path, data)

    @classmethod
    def create_json_file_by_path(cls, output_path: str, data: list, indent: int = None) -> None:
        dir_name = os.path.dirname(output_path)
        cls.make_dir_safety(dir_name)
        try:
            with os.fdopen(os.open(output_path, os.O_WRONLY | os.O_CREAT, Constant.FILE_AUTHORITY), "w") as file:
                json.dump(data, file, indent=indent)
        except Exception:
            raise RuntimeError(f"Can't create file: {output_path}")

    @classmethod
    def check_input_path(cls, path):
        """
        Function Description:
            check whether the path is valid
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        if len(path) > Constant.MAX_PATH_LENGTH:
            msg = f"The length of file path exceeded the maximum value {Constant.MAX_PATH_LENGTH}: {path}"
            raise RuntimeError(msg)

        if os.path.islink(path):
            msg = f"Invalid profiling path is soft link: {path}"
            raise RuntimeError(msg)

        if os.path.isfile(path):
            raise RuntimeError('Your profiling output path {} is a file.'.format(path))

    @classmethod
    def check_directory_path_writable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        if not os.path.exists(path):
            raise RuntimeError('The path {} is not exist.'.format(path))
        if not os.access(path, os.W_OK):
            raise RuntimeError('The path {} does not have permission to write. '
                               'Please check the path permission'.format(path))

    @classmethod
    def remove_file_safety(cls, path: str):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception:
                print(f"[WARNING] [{os.getpid()}] profiler.py: Can't remove the directory: {path}")

    @classmethod
    def make_dir_safety(cls, path: str):
        if os.path.islink(path):
            msg = f"Invalid path is soft link: {path}"
            raise RuntimeError(msg)
        if os.path.exists(path):
            return
        try:
            os.makedirs(path, mode=Constant.DIR_AUTHORITY)
            os.chmod(path, Constant.DIR_AUTHORITY)
        except Exception:
            raise RuntimeError("Can't create directory: " + path)
