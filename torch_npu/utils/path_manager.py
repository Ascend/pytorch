# Copyright (c) 2023 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import shutil
import warnings

from torch_npu.utils._error_code import ErrCode, pta_error


__all__ = ['PathManager']


class PathManager:
    MAX_PATH_LENGTH = 4096
    MAX_FILE_NAME_LENGTH = 255
    DATA_FILE_AUTHORITY = 0o640
    DATA_DIR_AUTHORITY = 0o750

    @classmethod
    def check_input_directory_path(cls, path: str):
        """
        Function Description:
            check whether the path is valid, some businesses can accept a path that does not exist,
            so the function do not verify whether the path exists
        Parameter:
            path: the path to check, whether the incoming path is absolute or relative depends on the business
        Exception Description:
            when invalid data throw exception
        """
        cls._input_path_common_check(path)

        if os.path.isfile(path):
            msg = "Invalid input path is a file path: {path}"
            raise RuntimeError(msg + pta_error(ErrCode.PARAM))

    @classmethod
    def check_input_file_path(cls, path: str):
        """
        Function Description:
            check whether the file path is valid, some businesses can accept a path that does not exist,
            so the function do not verify whether the path exists
        Parameter:
            path: the file path to check, whether the incoming path is absolute or relative depends on the business
        Exception Description:
            when invalid data throw exception
        """
        cls._input_path_common_check(path)

        if os.path.isdir(path):
            msg = "Invalid input path is a directory path: {path}"
            raise RuntimeError(msg + pta_error(ErrCode.PARAM))

    @classmethod
    def check_path_owner_consistent(cls, path: str):
        """
        Function Description:
            check whether the path belong to process owner
        Parameter:
            path: the path to check
        Exception Description:
            when invalid path, prompt the user
        """

        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise RuntimeError(msg + pta_error(ErrCode.NOT_FOUND))
        if os.stat(path).st_uid != os.getuid():
            warnings.warn(f"Warning: The {path} owner does not match the current user.")

    @classmethod
    def check_directory_path_writeable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg + pta_error(ErrCode.UNAVAIL))
        if not os.access(path, os.W_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg + pta_error(ErrCode.PERMISSION))

    @classmethod
    def check_directory_path_readable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg + pta_error(ErrCode.UNAVAIL))
        if not os.access(path, os.R_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg + pta_error(ErrCode.PERMISSION))

    @classmethod
    def remove_path_safety(cls, path: str):
        msg = f"Failed to remove path: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg + pta_error(ErrCode.UNAVAIL))
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except Exception as err:
            raise RuntimeError(msg + pta_error(ErrCode.SYSCALL)) from err

    @classmethod
    def remove_file_safety(cls, file: str):
        msg = f"Failed to remove file: {file}"
        if os.path.islink(file):
            raise RuntimeError(msg)
        if not os.path.exists(file):
            return
        try:
            os.remove(file)
        except FileExistsError:
            return
        except Exception as err:
            raise RuntimeError(msg + pta_error(ErrCode.SYSCALL)) from err

    @classmethod
    def make_dir_safety(cls, path: str):
        msg = f"Failed to make directory: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg + pta_error(ErrCode.UNAVAIL))
        if os.path.exists(path):
            return
        try:
            os.makedirs(path, mode=cls.DATA_DIR_AUTHORITY, exist_ok=True)
        except Exception as err:
            raise RuntimeError(msg + pta_error(ErrCode.SYSCALL)) from err

    @classmethod
    def create_file_safety(cls, path: str):
        msg = f"Failed to create file: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg + pta_error(ErrCode.UNAVAIL))
        if os.path.exists(path):
            return
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, cls.DATA_FILE_AUTHORITY))
        except Exception as err:
            raise RuntimeError(msg + pta_error(ErrCode.SYSCALL)) from err

    @classmethod
    def _input_path_common_check(cls, path: str):
        if len(path) > cls.MAX_PATH_LENGTH:
            raise RuntimeError("Length of input path exceeds the limit." + pta_error(ErrCode.PARAM))

        if os.path.islink(path):
            msg = f"Invalid input path is a soft chain: {path}"
            raise RuntimeError(msg + pta_error(ErrCode.UNAVAIL))

        pattern = r'(\.|/|_|-|\s|[~0-9a-zA-Z]|[\u4e00-\u9fa5])+'
        if not re.fullmatch(pattern, path):
            msg = f"Invalid input path: {path}"
            raise RuntimeError(msg + pta_error(ErrCode.PARAM))

        path_split_list = path.split("/")
        for name in path_split_list:
            if len(name) > cls.MAX_FILE_NAME_LENGTH:
                raise RuntimeError("Length of input path exceeds the limit." + pta_error(ErrCode.PARAM))
