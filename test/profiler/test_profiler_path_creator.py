import unittest
import os
import json

import torch

import torch_npu
from torch_npu.profiler._profiler_path_creator import ProfPathCreator
from torch_npu.utils._path_manager import PathManager
from torch_npu.testing.testcase import TestCase, run_tests


class TestPathCreator(TestCase):

    RESULT_DIR = "./result_dir"
    ASCEND_ENV_SET = "./profiling_data"
    PROFILING_TAIL = "ascend_pt"

    STE_PARAMS = [
        [None, None, None], [RESULT_DIR, None, None],
        [None, "work_name_1", None], [RESULT_DIR, "work_name_2", None],
        [None, "work_name_3", ASCEND_ENV_SET]
    ]

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TestPathCreator.RESULT_DIR):
            PathManager.remove_path_safety(TestPathCreator.RESULT_DIR)
        if os.path.exists(TestPathCreator.ASCEND_ENV_SET):
            PathManager.remove_path_safety(TestPathCreator.ASCEND_ENV_SET)
        sub_dirs = os.listdir(os.getcwd())
        for sub_dir in sub_dirs:
            if sub_dir.endswith(TestPathCreator.PROFILING_TAIL):
                PathManager.remove_path_safety(os.path.join(os.getcwd(), sub_dir))

    def test_create_prof_dir(self):
        for params in self.STE_PARAMS:
            dir_name, work_name, ascend_env = params
            if ascend_env:
                os.environ["ASCEND_WORK_PATH"] = ascend_env
            ProfPathCreator().init(dir_name=dir_name, worker_name=work_name)
            ProfPathCreator().create_prof_dir()
            exists = self._check_dir_exists(dir_name, ascend_env)
            self.assertEqual(True, exists)
            os.environ["ASCEND_WORK_PATH"] = ""

    def _check_dir_exists(self, dir_name, ascend_env):
        if not dir_name and not ascend_env:
            prof_dir = os.getcwd()
        elif not dir_name:
            prof_dir = ascend_env
        else:
            prof_dir = dir_name
        return os.path.exists(prof_dir)


if __name__ == "__main__":
    run_tests()