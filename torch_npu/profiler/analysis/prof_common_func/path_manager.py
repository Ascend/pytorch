import os
import re

from ....utils.path_manager import PathManager
from ..prof_common_func.constant import Constant


class ProfilerPathManager:
    @classmethod
    def get_fwk_path(cls, profiler_path: str) -> str:
        fwk_path = os.path.join(profiler_path, Constant.FRAMEWORK_DIR)
        if os.path.isdir(fwk_path):
            return fwk_path
        return ""

    @classmethod
    def get_cann_path(cls, profiler_path: str) -> str:
        sub_dirs = os.listdir(os.path.realpath(profiler_path))
        for sub_dir in sub_dirs:
            sub_path = os.path.join(profiler_path, sub_dir)
            if os.path.isdir(sub_path) and re.match(r"^PROF_\d+_\d+_[0-9a-zA-Z]+", sub_dir):
                return sub_path
        return ""

    @classmethod
    def get_info_file_path(cls, profiler_path: str) -> str:
        sub_files = os.listdir(profiler_path)
        for sub_file in sub_files:
            sub_path = os.path.join(profiler_path, sub_file)
            if sub_file == "profiler_info.json":
                return sub_path
            if re.match(r"^profiler_info_\d+\.json", sub_file):
                return sub_path
        return ""

    @classmethod
    def get_info_path(cls, profiler_path: str) -> str:
        info_path = os.path.join(cls.get_cann_path(profiler_path), 'host/info.json')
        if os.path.exists(info_path):
            return info_path
        else:
            return ""

    @classmethod
    def get_host_start_log_path(cls, profiler_path: str) -> str:
        info_path = os.path.join(cls.get_cann_path(profiler_path), 'host/host_start.log')
        if os.path.exists(info_path):
            return info_path
        else:
            return ""
    
    @classmethod
    def get_host_path(cls, cann_path: str) -> str:
        host_path = os.path.join(cann_path, 'host')
        if os.path.exists(host_path):
            return host_path
        else:
            return ""

    @classmethod
    def get_device_path(cls, cann_path: str) -> str:
        sub_dirs = os.listdir(os.path.realpath(cann_path))
        for sub_dir in sub_dirs:
            sub_path = os.path.join(cann_path, sub_dir)
            if os.path.isdir(sub_path) and re.match(r"^device_\d", sub_dir):
                return sub_path
        return ""

    @classmethod
    def get_start_info_path(cls, cann_path: str) -> str:
        start_info_path = os.path.join(cann_path, "host", "start_info")
        if os.path.exists(start_info_path):
            return start_info_path
        device_path = cls.get_device_path(cann_path)
        if not device_path:
            return ""
        device_path_split = os.path.basename(device_path).split("_")
        if len(device_path_split) != 2:
            return ""
        start_info_file = f"start_info.{device_path_split[1]}"
        start_info_path = os.path.join(device_path, start_info_file)
        if os.path.exists(start_info_path):
            return start_info_path
        return ""

    @classmethod
    def get_profiler_path_list(cls, input_path: str) -> list:
        if not os.path.isdir(input_path):
            return []
        if cls.get_fwk_path(input_path) or cls.get_cann_path(input_path):
            return [input_path]
        sub_dirs = os.listdir(os.path.realpath(input_path))
        profiler_path_list = []
        for sub_dir in sub_dirs:
            sub_path = os.path.join(input_path, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            if cls.get_fwk_path(sub_path) or cls.get_cann_path(sub_path):
                profiler_path_list.append(sub_path)
        return profiler_path_list

    @classmethod
    def get_device_all_file_list_by_type(cls, profiler_path: str, summary_or_timeline: str) -> list:
        file_list = []
        _path = os.path.join(cls.get_device_path(profiler_path), summary_or_timeline)
        if not os.path.isdir(_path):
            return file_list
        sub_files = os.listdir(os.path.realpath(_path))
        if not sub_files:
            return file_list
        for sub_file in sub_files:
            file_list.append(os.path.join(_path, sub_file))
        return file_list

    @classmethod
    def get_analyze_all_file(cls, profiler_path: str, analyze: str) -> list:
        file_list = []
        _path = os.path.join(profiler_path, analyze)
        if not os.path.isdir(_path):
            return file_list
        sub_files = os.listdir(os.path.realpath(_path))
        if not sub_files:
            return file_list
        for sub_file in sub_files:
            file_list.append(os.path.join(_path, sub_file))
        return file_list

    @classmethod
    def get_realpath(cls, path: str) -> str:
        if os.path.islink(path):
            msg = f"Invalid input path is a soft chain: {path}"
            raise RuntimeError(msg)
        return os.path.realpath(path)
    
    @classmethod
    def simplify_data(cls, profiler_path: str, simplify_flag: bool):
        cann_path = cls.get_cann_path(profiler_path)
        device_path = cls.get_device_path(cann_path)
        host_path = cls.get_host_path(cann_path)
        rm_dirs = ['log', 'sqlite', 'summary', 'timeline'] if simplify_flag else ['log', 'sqlite']
        for rm_dir in rm_dirs:
            if device_path:
                target_path = os.path.join(device_path, rm_dir)
                PathManager.remove_path_safety(target_path)
            if host_path:
                target_path = os.path.join(host_path, rm_dir)
                PathManager.remove_path_safety(target_path)
        if simplify_flag:
            fwk_path = cls.get_fwk_path(profiler_path)
            PathManager.remove_path_safety(fwk_path)
            if cann_path:
                cann_rm_dirs = ['analyze', 'mindstudio_profiler_log', 'mindstudio_profiler_output']
                for cann_rm_dir in cann_rm_dirs:
                    target_path = os.path.join(cann_path, cann_rm_dir)
                    PathManager.remove_path_safety(target_path)
