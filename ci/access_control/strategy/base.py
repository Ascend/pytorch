import os
import subprocess
from abc import ABCMeta, abstractmethod
from ..constants import BASE_DIR


class AccurateTest(metaclass=ABCMeta):
    base_dir = BASE_DIR
    test_dir = base_dir / 'test'

    @abstractmethod
    def identify(self, modify_file):
        """
        该接口提供代码对应的UT的路径信息
        """
        raise Exception("abstract method. Subclasses should implement it.")

    @classmethod
    def find_ut_by_regex(cls, regex):
        ut_files = []
        cmd = "find {} -name {}".format(str(cls.test_dir), regex)
        status, output = subprocess.getstatusoutput(cmd)
        # 对于找不到的暂不作处理
        if not(status or not output):
            files = output.split('\n')
            for ut_file in files:
                ut_file_basename = os.path.basename(ut_file)
                if ut_file_basename.startswith("test") and ut_file.endswith(".py"):
                    ut_files.append(ut_file)
        return ut_files
