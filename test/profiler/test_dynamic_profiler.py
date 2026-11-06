# Copyright (c) 2026 Huawei Technologies Co., Ltd
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
# Owner(s): ["oncall: profiler"]
import copy
import json
import os
import stat
import struct
import subprocess
import time
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import torch_npu.profiler.dynamic_profile as dp
from torch_npu.profiler._dynamic_profiler._dynamic_profiler_config_context import (
    ConfigContext,
)
from torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor_shm import (
    DynamicProfilerShareMemory,
)
from torch_npu.profiler._dynamic_profiler._dynamic_profiler_utils import (
    DynamicProfilerUtils,
)
from torch_npu.profiler.dynamic_profile import _DynamicProfile
from torch_npu.profiler.profiler import profile, tensorboard_trace_handler
from torch_npu.profiler.scheduler import Schedule as schedule
from torch_npu.testing.testcase import run_tests, TestCase
from torch_npu.utils._path_manager import PathManager

import torch


class SmallModel(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=12):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, 3, padding=1)

    def forward(self, input_1):
        input_1 = self.conv1(input_1)
        input_1 = self.relu1(input_1)
        input_1 = self.conv2(input_1)
        return input_1.reshape(input_1.shape[0], -1)


class TrainModel:
    def __init__(self):
        self.input_shape = (4, 3, 24, 24)
        self.out_shape = (4, 12, 24, 24)
        self.device = torch.device("npu:0")
        self.model = SmallModel(self.input_shape[1], self.out_shape[1]).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)

    def train_one_step(self):
        inputs = torch.rand(self.input_shape).to(self.device)
        target = (
            torch.rand(self.out_shape).reshape(self.out_shape[0], -1).to(self.device)
        )
        output = self.model(inputs)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TestDynamicProfiler(TestCase):
    TRACE_FILE_NAME = "trace_view.json"
    KERNEL_FILE_NAME = "kernel_details.csv"
    model_train = TrainModel()
    small_steps = 1
    large_steps = 5
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    mode = stat.S_IRUSR | stat.S_IWUSR
    start_step = 0

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.json_sample = DynamicProfilerShareMemory.JSON_DATA
        cls.results_path = f"./dynamic_profiler_results_{str(os.getpid())}"
        cls.default_prof_dir = os.path.join(cls.results_path, "default_prof_dir")
        cls.rank_prof_dir = os.path.join(cls.results_path, "rank_prof_dir")
        cls.invalid_rank_prof_dir = os.path.join(
            cls.results_path, "invalid_rank_prof_dir"
        )
        cls.active_rank_prof_dir = os.path.join(
            cls.results_path, "active_rank_prof_dir"
        )
        cls.cfg_prof_dir = os.path.join(cls.results_path, "cfg_prof_dir")
        cls.cfg_path = os.path.join(cls.results_path, "profiler_config.json")
        os.environ["RANK"] = "0"
        dp.init(cls.results_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.results_path):
            PathManager.remove_path_safety(cls.results_path)

    def test_missing_optional_experimental_config_params_use_defaults(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["experimental_config"].pop("host_sys")
        cfg_json["experimental_config"].pop("sys_io")
        cfg_json["experimental_config"].pop("sys_interconnection")

        cfg_ctx = ConfigContext(cfg_json)

        self.assertEqual([], cfg_ctx.experimental_config._host_sys)
        self.assertFalse(cfg_ctx.experimental_config._sys_io)
        self.assertFalse(cfg_ctx.experimental_config._sys_interconnection)

    def test_modify_cfg_prof_dir_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["prof_dir"] = 1
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_analyse_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["analyse"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_record_shapes_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["record_shapes"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_profile_memory_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["profile_memory"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_with_stack_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["with_stack"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_with_flops_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["with_flops"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_with_modules_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["with_modules"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_rank_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["is_rank"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_rank_list_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["is_rank"] = True
        cfg_json["rank_list"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_profiler_level_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if "experimental_config" not in cfg_json:
            self.assertTrue("experimental_config" in cfg_json)
        cfg_json["experimental_config"]["profiler_level"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_aic_metrics_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if "experimental_config" not in cfg_json:
            self.assertTrue("experimental_config" in cfg_json)
        cfg_json["experimental_config"]["aic_metrics"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_l2_cache_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if "experimental_config" not in cfg_json:
            self.assertTrue("experimental_config" in cfg_json)
        cfg_json["experimental_config"]["l2_cache"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_data_simplification_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if "experimental_config" not in cfg_json:
            self.assertTrue("experimental_config" in cfg_json)
        cfg_json["experimental_config"]["data_simplification"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_record_op_args_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if "experimental_config" not in cfg_json:
            self.assertTrue("experimental_config" in cfg_json)
        cfg_json["experimental_config"]["record_op_args"] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_export_type_invalid(self) -> None:
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["is_rank"] = False
        if "experimental_config" not in cfg_json:
            self.assertTrue("experimental_config" in cfg_json)
        cfg_json["experimental_config"]["export_type"] = 1
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(
                wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0
            ),
            on_trace_ready=tensorboard_trace_handler(
                self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()
            ),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config,
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def ps_eT_grep(self, thread_name):
        proc = subprocess.Popen(["ps", "-eT"], stdout=subprocess.PIPE, text=True)
        grep = subprocess.Popen(
            ["grep", thread_name],
            stdin=proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output, _ = grep.communicate()
        return output.strip() != ""

    def test_dynamic_profiler_default_start(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["prof_dir"] = self.default_prof_dir
        cfg_json["start_step"] = TestDynamicProfiler.start_step + 1
        # 验证删除配置文件不会影响profiler功能
        os.remove(self.cfg_path)
        with os.fdopen(os.open(self.cfg_path, self.flags, self.mode), "w") as f:
            time.sleep(1)
            json.dump(cfg_json, f, indent=4)
        time.sleep(3)
        dp.step()
        # 检查监控进程DynMonitorProc名称是否正确
        self.assertTrue(self.ps_eT_grep("DynMonitorProc"))
        TestDynamicProfiler.start_step += 1
        self.model_train.train_one_step()
        dp.step()
        TestDynamicProfiler.start_step += 1
        has_prof = False
        if self.has_prof_dir(self.default_prof_dir):
            has_prof = True
        if os.path.exists(self.default_prof_dir):
            PathManager.remove_path_safety(self.default_prof_dir)
        self.assertTrue(has_prof)

    def test_dynamic_profiler_default_start_next_step(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["prof_dir"] = self.default_prof_dir
        cfg_json["start_step"] = -1
        with os.fdopen(os.open(self.cfg_path, self.flags, self.mode), "w") as f:
            time.sleep(1)
            json.dump(cfg_json, f, indent=4)
        time.sleep(3)
        dp.step()
        TestDynamicProfiler.start_step += 1
        self.model_train.train_one_step()
        dp.step()
        TestDynamicProfiler.start_step += 1
        has_prof = self.has_prof_dir(self.default_prof_dir)
        if os.path.exists(self.default_prof_dir):
            PathManager.remove_path_safety(self.default_prof_dir)
        self.assertTrue(has_prof)

    def test_dynamic_profiler_rank(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["prof_dir"] = self.rank_prof_dir
        cfg_json["is_rank"] = True
        cfg_json["rank_list"] = [0]
        cfg_json["start_step"] = TestDynamicProfiler.start_step + 1

        with os.fdopen(os.open(self.cfg_path, self.flags, self.mode), "w") as f:
            time.sleep(1)
            json.dump(cfg_json, f, indent=4)
        time.sleep(3)
        dp.step()
        TestDynamicProfiler.start_step += 1
        self.model_train.train_one_step()
        dp.step()
        TestDynamicProfiler.start_step += 1
        has_prof = False
        if self.has_prof_dir(self.rank_prof_dir):
            has_prof = True
        if os.path.exists(self.rank_prof_dir):
            PathManager.remove_path_safety(self.rank_prof_dir)
        self.assertTrue(has_prof)

    def test_dynamic_profiler_rank_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["prof_dir"] = self.invalid_rank_prof_dir
        cfg_json["is_rank"] = True
        cfg_json["rank_list"] = [1]
        cfg_json["start_step"] = TestDynamicProfiler.start_step + 1

        with os.fdopen(os.open(self.cfg_path, self.flags, self.mode), "w") as f:
            time.sleep(1)
            json.dump(cfg_json, f, indent=4)
        time.sleep(3)
        dp.step()
        TestDynamicProfiler.start_step += 1
        self.model_train.train_one_step()
        dp.step()
        TestDynamicProfiler.start_step += 1
        has_prof = False
        if self.has_prof_dir(self.invalid_rank_prof_dir):
            has_prof = True
        if os.path.exists(self.invalid_rank_prof_dir):
            PathManager.remove_path_safety(self.invalid_rank_prof_dir)
        self.assertFalse(has_prof)

    @staticmethod
    def has_prof_dir(path: str) -> bool:
        path = os.path.realpath(path)
        if not os.path.exists(path):
            return False
        for sub_dir in os.listdir(path):
            if sub_dir.endswith("_pt"):
                sub_dir = os.path.join(path, sub_dir)
                for p in os.listdir(sub_dir):
                    if p.startswith("PROF"):
                        return True
        return False

    @staticmethod
    def has_analyse_dir(path: str) -> bool:
        path = os.path.realpath(path)
        if not os.path.exists(path):
            return False
        for sub_dir in os.listdir(path):
            if sub_dir.endswith("_pt"):
                sub_dir = os.path.join(path, sub_dir)
                for p in os.listdir(sub_dir):
                    if p.startswith("ASCEND"):
                        return True
        return False

    def test_out_log_dyno_model(self):
        original_model = DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL
        DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = (
            DynamicProfilerUtils.DynamicProfilerConfigModel.DYNO_CONFIG
        )
        original_stdout_log = DynamicProfilerUtils.stdout_log
        log_calls = []

        def log_function(infos, level):
            log_calls.append((infos, level))

        DynamicProfilerUtils.stdout_log = log_function

        try:
            DynamicProfilerUtils.out_log(
                "test information", DynamicProfilerUtils.LoggerLevelEnum.INFO
            )
            DynamicProfilerUtils.out_log(
                "test warning", DynamicProfilerUtils.LoggerLevelEnum.WARNING
            )
            DynamicProfilerUtils.out_log(
                "test error", DynamicProfilerUtils.LoggerLevelEnum.ERROR
            )

            self.assertEqual(len(log_calls), 3)
            self.assertEqual(
                log_calls[0],
                ("test information", DynamicProfilerUtils.LoggerLevelEnum.INFO),
            )
            self.assertEqual(
                log_calls[1],
                ("test warning", DynamicProfilerUtils.LoggerLevelEnum.WARNING),
            )
            self.assertEqual(
                log_calls[2], ("test error", DynamicProfilerUtils.LoggerLevelEnum.ERROR)
            )
        finally:
            DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = original_model
            DynamicProfilerUtils.stdout_log = original_stdout_log

    def test_clean_shm_for_killed_pid_time_none(self):
        shm_instance = DynamicProfilerShareMemory(self.results_path, self.cfg_path, 0)
        with mock.patch(
            "torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor_shm.DynamicProfilerShareMemory._get_pid_st_ctime",
            return_value=None,
        ):
            try:
                shm_instance._clean_shm_for_killed()
            except Exception:
                self.fail(
                    "_clean_shm_for_killed should not raise exception when pid_time is None"
                )

    def test_create_shm_over_py38_retry_failure(self):
        from multiprocessing import shared_memory

        with mock.patch(
            "multiprocessing.resource_tracker.register", lambda *args, **kwargs: None
        ):
            with mock.patch.object(
                shared_memory.SharedMemory,
                "__init__",
                side_effect=FileNotFoundError("Test error"),
            ):
                with self.assertRaises(RuntimeError):
                    shm_instance = DynamicProfilerShareMemory(
                        self.results_path, self.cfg_path, 0
                    )
                    shm_instance._create_shm_over_py38()

    def test_get_pid_st_ctime_exception(self):
        shm_instance = DynamicProfilerShareMemory(self.results_path, self.cfg_path, 0)
        with mock.patch("os.open", side_effect=Exception("Test exception")):
            result = shm_instance._get_pid_st_ctime(12345)
            self.assertIsNone(result)

    def test_call_dyno_monitor_with_proxy(self):
        from torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor import (
            DynamicProfilerMonitor,
        )

        monitor = DynamicProfilerMonitor()
        mock_proxy = MagicMock()
        with patch(
            "torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor.PyDynamic"
            "MonitorProxySingleton"
        ) as mock_singleton:
            mock_singleton_instance = MagicMock()
            mock_singleton_instance.get_proxy.return_value = mock_proxy
            mock_singleton.return_value = mock_singleton_instance
            test_data = {"key": "value", "number": 123}
            monitor._call_dyno_monitor(test_data)
            mock_proxy.enable_dyno_npu_monitor.assert_called_once()

    def test_clean_resource_with_process(self):
        from torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor import (
            DynamicProfilerMonitor,
        )

        monitor = DynamicProfilerMonitor()
        monitor._process = MagicMock()
        with patch.object(monitor, "_shared_loop_flag") as mock_flag:
            monitor.clean_resource()
            mock_flag.value = False
            monitor._process.join.assert_called_once()

    def test_shm_to_prof_conf_context_read_time_exception(self):
        from torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor import (
            DynamicProfilerMonitor,
        )

        monitor = DynamicProfilerMonitor()
        with patch.object(
            monitor._shm_obj, "read_bytes", side_effect=Exception("Read error")
        ):
            result = monitor.shm_to_prof_conf_context()
            self.assertEqual(result, [])

    def test_shm_to_prof_conf_context_none_shm(self):
        from torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor import (
            DynamicProfilerMonitor,
        )

        monitor = DynamicProfilerMonitor()
        monitor._shm_obj = None
        result = monitor.shm_to_prof_conf_context()
        self.assertEqual(result, [])

    def test_start_while_profiler_active(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json["prof_dir"] = self.active_rank_prof_dir
        with os.fdopen(os.open(self.cfg_path, self.flags, self.mode), "w") as f:
            json.dump(cfg_json, f, indent=4)

        dp.start(self.cfg_path)
        dp.start(self.cfg_path)
        self.assertIsNotNone(_DynamicProfile().prof)
        _DynamicProfile().prof.stop()
        _DynamicProfile().prof = None
        if os.path.exists(self.active_rank_prof_dir):
            PathManager.remove_path_safety(self.active_rank_prof_dir)

    def test_init_repeated_warning(self):
        dp.init(self.results_path)
        dp.init(self.results_path)
        self.assertTrue(_DynamicProfile().repeat_init)

    def test_step_time_calculation(self):
        dynamic_prof = _DynamicProfile()
        dynamic_prof.RECORD_TIME_STEP = 2
        dynamic_prof.cur_step = 1
        dynamic_prof._step_record_time = time.time()
        dynamic_prof.step()
        self.assertIsNotNone(dynamic_prof._step_time)

    def test_set_state(self):
        dynamic_prof = _DynamicProfile()
        dynamic_prof.cur_step = 7

        dynamic_prof.cur_step = 0
        dynamic_prof.set_state({"cur_step": 7})
        self.assertEqual(dynamic_prof.cur_step, 7)

        dynamic_prof.set_state({"cur_step": -1})
        self.assertEqual(dynamic_prof.cur_step, 7)

    def test_set_state_continue_step(self):
        dynamic_prof = _DynamicProfile()
        dynamic_prof.cur_step = 0
        dynamic_prof._dynamic_monitor = MagicMock()
        dynamic_prof._dynamic_monitor.shm_to_prof_conf_context.return_value = []
        dynamic_prof._step_mstx_range_id = 0
        dynamic_prof.set_state({"cur_step": 10})
        with patch("torch_npu.profiler.dynamic_profile.mstx.range_start", return_value=1):
            dynamic_prof.step()
            self.assertEqual(dynamic_prof.cur_step, 11)


def _dyno_frame(configs):
    sep = DynamicProfilerUtils.RECORD_SEPARATOR
    payload = sep.join(configs)
    payload_bytes = payload.encode("utf-8")
    head = configs[-1]
    head += "\nDYNO_BATCH_COUNT=" + str(len(configs))
    head += "\nDYNO_BATCH_LEN=" + str(len(payload_bytes))
    head += "\nDYNO_BATCH_SUM=" + "{:016x}".format(
        DynamicProfilerUtils._fnv1a64(payload_bytes))
    return head + sep + payload


class TestParseDynoConfigs(TestCase):
    SEP = "\x1e"

    def _make_cfg(self, iteration, trace_id):
        return (
            "\n"
            "            PROFILE_ACTIVITIES=CPU,NPU\n"
            f"            ACTIVITIES_ITERATIONS={iteration}\n"
            "            WARMUP_ITERATIONS=1\n"
            f"            REQUEST_TRACE_ID={trace_id}"
        )

    def _cfg_pair(self):
        return self._make_cfg(2, "111"), self._make_cfg(3, "222")

    def test_batch(self):
        cfg1, cfg2 = self._cfg_pair()
        configs = DynamicProfilerUtils.parse_dyno_configs(_dyno_frame([cfg1, cfg2]))
        self.assertEqual(2, len(configs))
        self.assertEqual("2", configs[0]["ACTIVITIES_ITERATIONS"])
        self.assertEqual("111", configs[0]["REQUEST_TRACE_ID"])
        self.assertEqual("3", configs[1]["ACTIVITIES_ITERATIONS"])
        self.assertEqual("222", configs[1]["REQUEST_TRACE_ID"])

    def test_corrupted_frame_drops_batch(self):
        cfg1, cfg2 = self._cfg_pair()
        record = _dyno_frame([cfg1, cfg2])
        corrupted = [
            record.replace("DYNO_BATCH_COUNT=2", "DYNO_BATCH_COUNT=3"),
            record.replace("222", "999"),
            record[:record.rfind(self.SEP)],  # truncated payload
        ]
        for rec in corrupted:
            self.assertEqual([], DynamicProfilerUtils.parse_dyno_configs(rec))

    def test_empty_record(self):
        self.assertEqual([], DynamicProfilerUtils.parse_dyno_configs(""))
        self.assertEqual([], DynamicProfilerUtils.parse_dyno_configs(None))

    def test_legacy_single_record_falls_back(self):
        cfg1, cfg2 = self._cfg_pair()
        for cfg, it, tid in ((cfg1, "2", "111"), (cfg2, "3", "222")):
            configs = DynamicProfilerUtils.parse_dyno_configs(cfg)
            self.assertEqual(1, len(configs))
            self.assertEqual(it, configs[0]["ACTIVITIES_ITERATIONS"])
            self.assertEqual(tid, configs[0]["REQUEST_TRACE_ID"])
        self.assertEqual([], DynamicProfilerUtils.parse_dyno_configs("\n\nnot-a-pair\n"))


class _SharedInt:
    def __init__(self, value=0):
        self.value = value


class _BudgetFlag:
    def __init__(self, budget):
        self._budget = budget

    @property
    def value(self):
        if self._budget <= 0:
            return False
        self._budget -= 1
        return True


class _ScriptedInt:
    def __init__(self, base=0, hooks=None):
        self._base = base
        self._read_count = 0
        self._hooks = hooks or {}

    @property
    def value(self):
        idx = self._read_count
        self._read_count += 1
        hook = self._hooks.get(idx)
        if hook:
            hook()
        return self._base


class _FakeShm:
    def __init__(self, fail_first=False):
        self.writes = []
        self.attempts = 0
        self._fail_first = fail_first

    def write_bytes_over_py38(self, data):
        self.attempts += 1
        if self._fail_first:
            self._fail_first = False
            raise RuntimeError("injected shm write failure")
        self.writes.append(data)


def _make_dyno_poll(script):
    script_iter = iter(script)
    return lambda: next(script_iter, "")


def _dyno_raw_record(trace_id, pad=0):
    record = ("\nPROFILE_ACTIVITIES=CPU,NPU\nACTIVITIES_ITERATIONS=1\n"
              "WARMUP_ITERATIONS=1\nREQUEST_TRACE_ID=" + trace_id)
    return record + "\nPADDING=" + "x" * pad if pad else record


def _dyno_record(trace_id, pad=0):
    return _dyno_frame([_dyno_raw_record(trace_id, pad)])


def _log_texts(log_mock):
    return [str(call[0][0]) for call in log_mock.call_args_list]


class TestShmToProfConfContext(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ.setdefault("RANK", "0")

    @staticmethod
    def _cfg_dict(trace_id, **extra):
        cfg = {"is_valid": True, "activities": ["cpu"], "prof_dir": "./",
               "REQUEST_TRACE_ID": trace_id}
        cfg.update(extra)
        return cfg

    @staticmethod
    def _shm_payload(payload, broken=False):
        from torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor import (
            DynamicProfilerMonitor)
        t_bytes = struct.pack("<I", int(time.time()) + 5)
        body = b"{broken json" if broken else \
            ConfigContext.profiler_cfg_json_to_bytes(payload)
        shm = MagicMock()
        shm.read_bytes.side_effect = [t_bytes, body]
        monitor = DynamicProfilerMonitor.__new__(DynamicProfilerMonitor)
        monitor._shm_obj = shm
        monitor._rank_id = 0
        monitor._cur_time = int(time.time())
        monitor._consume_seq = _SharedInt(0)
        monitor.prof_cfg_context = None
        monitor.t_bytes = t_bytes
        return monitor

    def test_payload_wrapped_in_list(self):
        monitor = self._shm_payload([self._cfg_dict("t1"), self._cfg_dict("t2")])
        with patch.object(DynamicProfilerUtils, "out_log"):
            result = monitor.shm_to_prof_conf_context()
        self.assertEqual(2, len(result))
        self.assertEqual(1, monitor._consume_seq.value)
        monitor = self._shm_payload(self._cfg_dict("t1"))
        with patch.object(DynamicProfilerUtils, "out_log"):
            result = monitor.shm_to_prof_conf_context()
        self.assertEqual(1, len(result))
        self.assertEqual("t1", result[0].trace_id())
        self.assertEqual(1, monitor._consume_seq.value)
        monitor._shm_obj.read_bytes.side_effect = [monitor.t_bytes]
        with patch.object(DynamicProfilerUtils, "out_log"):
            result = monitor.shm_to_prof_conf_context()
        self.assertEqual([], result)
        self.assertEqual(1, monitor._consume_seq.value)

    def test_mixed_batch_dispatch_and_skip(self):
        from torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor import (
            DynamicProfilerMonitor)
        monitor_cfg = {"is_valid": True, "NPU_MONITOR_START": True}
        monitor = self._shm_payload([monitor_cfg, 123, self._cfg_dict("valid1")])
        with patch.object(DynamicProfilerMonitor, "_call_dyno_monitor") as call_mock, \
                patch.object(DynamicProfilerUtils, "out_log") as log_mock:
            result = monitor.shm_to_prof_conf_context()
        self.assertEqual(1, len(result))
        self.assertEqual("valid1", result[0].trace_id())
        call_mock.assert_called_once()
        self.assertTrue(call_mock.call_args[0][0].get("NPU_MONITOR_START"))
        self.assertTrue(any("expect dict" in t for t in _log_texts(log_mock)))

    def test_decode_failure_retries_same_timestamp(self):
        monitor = self._shm_payload([self._cfg_dict("t1")], broken=True)
        init_time = monitor._cur_time
        with patch.object(DynamicProfilerUtils, "out_log"):
            result = monitor.shm_to_prof_conf_context()
        self.assertEqual([], result)
        self.assertEqual(0, monitor._consume_seq.value)
        self.assertEqual(init_time, monitor._cur_time)
        monitor._shm_obj.read_bytes.side_effect = [
            monitor.t_bytes,
            ConfigContext.profiler_cfg_json_to_bytes([self._cfg_dict("t1")]),
        ]
        with patch.object(DynamicProfilerUtils, "out_log"):
            result = monitor.shm_to_prof_conf_context()
        self.assertEqual(1, len(result))
        self.assertEqual(1, monitor._consume_seq.value)
        self.assertEqual(init_time + 5, monitor._cur_time)

    def test_invalid_config_does_not_drop_whole_batch(self):
        original_model = DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL
        DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = (
            DynamicProfilerUtils.DynamicProfilerConfigModel.DYNO_CONFIG)
        no_activities = {"is_valid": True, "REQUEST_TRACE_ID": "t1"}
        bad_async_mode = {"is_valid": True, "REQUEST_TRACE_ID": "t2",
                          "PROFILE_ACTIVITIES": "CPU", "PROFILE_ASYNC_MODE": 123}
        good = {"is_valid": True, "REQUEST_TRACE_ID": "t3",
                "PROFILE_ACTIVITIES": "CPU"}
        try:
            monitor = self._shm_payload([no_activities, bad_async_mode, good])
            with patch.object(DynamicProfilerUtils, "out_log") as log_mock:
                result = monitor.shm_to_prof_conf_context()
        finally:
            DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = original_model
        self.assertEqual(["t1", "t3"], [ctx.trace_id() for ctx in result])
        self.assertEqual(2, len(result[0].activities()))
        self.assertEqual(1, monitor._consume_seq.value)
        texts = _log_texts(log_mock)
        self.assertTrue(any("PROFILE_ACTIVITIES is missing" in t for t in texts))
        self.assertTrue(any("Skip invalid config context" in t and "t2" in t
                            for t in texts))


class TestWorkerDynoFunc(TestCase):
    @staticmethod
    def _single_cfg_size(record):
        cfg = DynamicProfilerUtils.parse_dyno_configs(record)[0]
        cfg["is_valid"] = True
        return len(ConfigContext.profiler_cfg_json_to_bytes([cfg])) + 4

    @staticmethod
    def _consume_bump():
        consume = _SharedInt(0)
        return consume, lambda: setattr(consume, "value", consume.value + 1)

    @staticmethod
    def _big_frame():
        pad = 600_000
        return _dyno_frame([_dyno_raw_record("tA", pad=pad),
                            _dyno_raw_record("tB", pad=pad)])

    def _run_dyno_worker(self, poll_script, budget, max_size, consume,
                         sleep_hooks=None, shm=None):
        from torch_npu.profiler._dynamic_profiler import (
            _dynamic_profiler_monitor as monitor_mod)
        proxy = MagicMock()
        proxy.init_dyno.return_value = True
        proxy.poll_dyno.side_effect = _make_dyno_poll(poll_script)
        singleton_instance = MagicMock()
        singleton_instance.get_proxy.return_value = proxy
        if shm is None:
            shm = _FakeShm()
        params = {
            "loop_flag": _BudgetFlag(budget),
            "poll_interval": _ScriptedInt(base=0, hooks=sleep_hooks),
            "shm": shm,
            "rank_id": 0,
            "max_size": max_size,
            "consume_seq": consume,
            "dynamic_profiler_utils": DynamicProfilerUtils,
        }
        with mock.patch.object(
            monitor_mod, "PyDynamicMonitorProxySingleton",
            return_value=singleton_instance), \
                mock.patch.object(DynamicProfilerUtils, "out_log") as log_mock, \
                mock.patch("time.sleep"):
            monitor_mod.worker_dyno_func(params)
        return shm, log_mock

    @staticmethod
    def _trace_batches(shm):
        return [[c["REQUEST_TRACE_ID"] for c in json.loads(w[4:])]
                for w in shm.writes]

    def test_two_writes_before_read_self_heals(self):
        consume, bump = self._consume_bump()
        rec_a, rec_b, rec_c = (_dyno_record("tA"), _dyno_record("tB"),
                               _dyno_record("tC"))
        shm, log_mock = self._run_dyno_worker(
            poll_script=[rec_a, rec_b, "", rec_c], budget=4,
            max_size=DynamicProfilerUtils.CFG_BUFFER_SIZE,
            consume=consume, sleep_hooks={3: bump})
        self.assertEqual([["tA"], ["tA", "tB"], ["tC"]], self._trace_batches(shm))
        self.assertTrue(any("clear 2 cached config(s)" in t
                            for t in _log_texts(log_mock)))

    def test_overflow_sends_prefix_and_resumes(self):
        record = self._big_frame()
        shm, log_mock = self._run_dyno_worker(
            poll_script=[record, "", "", ""], budget=4,
            max_size=self._single_cfg_size(record) + 4096,
            consume=_SharedInt(0))
        batches = self._trace_batches(shm)
        self.assertEqual([["tA"], ["tA"], ["tA"], ["tA"]], batches)
        self.assertTrue(any("keep 1 config(s) for retry" in t
                            for t in _log_texts(log_mock)))
        consume, bump = self._consume_bump()
        shm, _ = self._run_dyno_worker(
            poll_script=[record, "", "", ""], budget=4,
            max_size=self._single_cfg_size(record) + 4096,
            consume=consume, sleep_hooks={1: bump})
        self.assertEqual([["tA"], ["tB"]], self._trace_batches(shm))

    def test_write_failure_retries_next_poll(self):
        rec_a = _dyno_record("tA")
        shm = _FakeShm(fail_first=True)
        shm, log_mock = self._run_dyno_worker(
            poll_script=[rec_a, ""], budget=2,
            max_size=DynamicProfilerUtils.CFG_BUFFER_SIZE,
            consume=_SharedInt(0), shm=shm)
        self.assertEqual(2, shm.attempts)
        self.assertEqual([["tA"]], self._trace_batches(shm))
        self.assertTrue(any("cfg bytes write failed" in t
                            for t in _log_texts(log_mock)))

    def test_bytes_poll_result_is_decoded(self):
        rec = _dyno_record("tA").encode("utf-8")
        shm, _ = self._run_dyno_worker(
            poll_script=[rec], budget=1,
            max_size=DynamicProfilerUtils.CFG_BUFFER_SIZE,
            consume=_SharedInt(0))
        self.assertEqual([["tA"]], self._trace_batches(shm))

    def test_parse_failure_keeps_worker_alive(self):
        shm, log_mock = self._run_dyno_worker(
            poll_script=[{"not": "a record"}, _dyno_record("tB")], budget=2,
            max_size=DynamicProfilerUtils.CFG_BUFFER_SIZE,
            consume=_SharedInt(0))
        self.assertEqual([["tB"]], self._trace_batches(shm))
        self.assertTrue(any("Parse dynolog config failed" in t
                            for t in _log_texts(log_mock)))

    def test_oversized_config_dropped_others_sent(self):
        record = _dyno_frame([_dyno_raw_record("big1", pad=6000),
                              _dyno_raw_record("small1")])
        shm, log_mock = self._run_dyno_worker(
            poll_script=[record, ""], budget=2, max_size=4096,
            consume=_SharedInt(0))
        texts = _log_texts(log_mock)
        self.assertTrue(any("keep 1 config(s) to send." in t for t in texts))
        self.assertFalse(any("for retry" in t for t in texts))
        self.assertEqual(["small1"], self._trace_batches(shm)[-1])

    def test_pending_buffer_capped(self):
        from torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor import (
            MAX_PENDING_CONFIGS)
        records = [_dyno_record(f"t{i:03d}") for i in range(MAX_PENDING_CONFIGS + 5)]
        shm, log_mock = self._run_dyno_worker(
            poll_script=records, budget=MAX_PENDING_CONFIGS + 5,
            max_size=1024 * 1024, consume=_SharedInt(0))
        texts = _log_texts(log_mock)
        self.assertTrue(any(t.startswith("Pending dynolog config buffer is full")
                            for t in texts))
        last_traces = self._trace_batches(shm)[-1]
        self.assertEqual(MAX_PENDING_CONFIGS + 1, len(last_traces))
        self.assertIn("t068", last_traces)
        self.assertNotIn("t000", last_traces)


class TestDynamicProfileConfigQueue(TestCase):
    DEFAULTS = {
        "_cfg_ctx_queue": [], "_seen_trace_ids": {}, "cfg_ctx": None, "prof": None,
        "cur_step": 0, "step_num": 0, "_dynamic_monitor": None,
        "_step_mstx_range_id": 0, "_step_record_time": None, "_step_time": 0,
        "_running_trace_id": "", "_rejected_cfg_count": 0,
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ.setdefault("RANK", "0")

    def setUp(self):
        super().setUp()
        self.dp_inst = _DynamicProfile()
        self._saved = {
            attr: getattr(self.dp_inst, attr) for attr in self.DEFAULTS}
        for attr, value in self.DEFAULTS.items():
            setattr(self.dp_inst, attr, copy.deepcopy(value))

    def tearDown(self):
        for attr, value in self._saved.items():
            setattr(self.dp_inst, attr, value)
        super().tearDown()

    def _mock_monitor(self):
        self.dp_inst._dynamic_monitor = MagicMock()
        self.dp_inst._dynamic_monitor.shm_to_prof_conf_context.return_value = []

    @staticmethod
    def _make_ctx(trace_id, start_step=0):
        cfg = {"is_valid": True, "activities": ["cpu"], "prof_dir": "./",
               "start_step": start_step}
        if trace_id:
            cfg["REQUEST_TRACE_ID"] = trace_id
        return ConfigContext(cfg)

    def _queue_trace_ids(self):
        return [ctx.trace_id() for ctx in self.dp_inst._cfg_ctx_queue]

    def _step(self):
        with patch("torch_npu.profiler.dynamic_profile.mstx.range_start",
                   return_value=1):
            self.dp_inst.step()

    def test_duplicate_batch_skipped_with_single_log(self):
        first = [self._make_ctx("t1"), self._make_ctx("t2")]
        second = [self._make_ctx("t1"), self._make_ctx("t2")]
        with patch.object(DynamicProfilerUtils, "out_log") as log_mock:
            self.dp_inst._append_cfg_ctxs(first)
            self.dp_inst._append_cfg_ctxs(second)
        self.assertEqual(["t1", "t2"], self._queue_trace_ids())
        skip_logs = [t for t in _log_texts(log_mock) if t.startswith("Skip ")]
        self.assertEqual(1, len(skip_logs))
        self.assertTrue(skip_logs[0].startswith("Skip 2 duplicate config context(s)"))
        self.dp_inst._append_cfg_ctxs([self._make_ctx(""), self._make_ctx("")])
        self.assertEqual(["t1", "t2", "", ""], self._queue_trace_ids())
        self.assertEqual({"t1": None, "t2": None}, self.dp_inst._seen_trace_ids)

    def test_queue_full_rejects_newest_and_can_retry(self):
        with patch.object(DynamicProfilerUtils, "out_log") as log_mock:
            for i in range(self.dp_inst.MAX_CONFIG_QUEUE_SIZE):
                self.dp_inst._append_cfg_ctxs([self._make_ctx(f"t{i}")])
            self.dp_inst._append_cfg_ctxs([self._make_ctx("t5")])
            self.assertEqual(
                [f"t{i}" for i in range(self.dp_inst.MAX_CONFIG_QUEUE_SIZE)],
                self._queue_trace_ids(),
            )
            self.assertEqual(1, self.dp_inst._rejected_cfg_count)
            self.assertNotIn("t5", self.dp_inst._seen_trace_ids)
            self.dp_inst._cfg_ctx_queue.pop(0)
            self.dp_inst._append_cfg_ctxs([self._make_ctx("t5")])
        self.assertEqual(
            [f"t{i}" for i in range(1, self.dp_inst.MAX_CONFIG_QUEUE_SIZE)] + ["t5"],
            self._queue_trace_ids(),
        )
        self.assertTrue(
            any("reject 1 config(s)" in t and "t5" in t
                for t in _log_texts(log_mock))
        )

    def test_rejected_configs_logged_as_one_aggregate_warning(self):
        with patch.object(DynamicProfilerUtils, "out_log") as log_mock:
            for i in range(self.dp_inst.MAX_CONFIG_QUEUE_SIZE):
                self.dp_inst._append_cfg_ctxs([self._make_ctx(f"ok{i}")])
            rejected = [self._make_ctx(f"drop{i}") for i in range(3)]
            self.dp_inst._append_cfg_ctxs(rejected)
        self.assertEqual(3, self.dp_inst._rejected_cfg_count)
        reject_logs = [t for t in _log_texts(log_mock) if "Config queue is full" in t]
        self.assertEqual(1, len(reject_logs))
        self.assertIn("reject 3 config(s)", reject_logs[0])
        for i in range(3):
            self.assertIn(f"drop{i}", reject_logs[0])

    def test_status_carries_trace_id_and_queue_depth(self):
        ready = self._make_ctx("ready1", start_step=100)
        queued = self._make_ctx("queued1", start_step=200)
        self.dp_inst._cfg_ctx_queue = [ready, queued]
        self.dp_inst.cur_step = 1
        self._mock_monitor()
        self._step()
        self.dp_inst._dynamic_monitor.update_cfg_status.assert_called_once_with(
            "ready1", 2, 0
        )

    def test_step_moves_queue_head_to_pending_slot(self):
        ready = self._make_ctx("ready1", start_step=100)
        future = self._make_ctx("future1", start_step=200)
        self.dp_inst._cfg_ctx_queue = [ready, future]
        self.dp_inst._seen_trace_ids = {"ready1": None, "future1": None}
        self.dp_inst.cur_step = 1
        self._mock_monitor()
        self._step()
        self.assertEqual(2, self.dp_inst.cur_step)
        self.assertIs(ready, self.dp_inst.cfg_ctx)
        self.assertEqual(["future1"], self._queue_trace_ids())
        self.dp_inst._dynamic_monitor.update_profiler_status.assert_any_call(
            DynamicProfilerUtils.ProfilerStatus.READY
        )

    def test_queued_config_expired_while_waiting_starts_immediately(self):
        ctx = self._make_ctx("delayed1", start_step=3)
        self.dp_inst.cur_step = 2
        self.dp_inst._append_cfg_ctxs([ctx])
        self.assertEqual(2, ctx.enqueued_step)
        self.dp_inst.cur_step = 5
        self._mock_monitor()
        with patch("torch_npu.profiler.dynamic_profile.print_warn_msg") as warn_mock, \
                patch.object(self.dp_inst, "enable_prof") as enable_mock:
            self._step()
        enable_mock.assert_called_once_with()
        self.assertIsNone(self.dp_inst.cfg_ctx)
        self.assertEqual(6, self.dp_inst.cur_step)
        self.assertTrue(
            any("delayed by the config queue" in str(c[0][0])
                for c in warn_mock.call_args_list)
        )
        self.assertFalse(
            any("not effective" in str(c[0][0]) for c in warn_mock.call_args_list)
        )
        self.dp_inst._dynamic_monitor.update_profiler_status.assert_any_call(
            DynamicProfilerUtils.ProfilerStatus.RUNNING
        )

    def test_config_stale_on_arrival_dropped_on_enqueue(self):
        ctx = self._make_ctx("stale1", start_step=1)
        self.dp_inst.cur_step = 5
        with patch.object(DynamicProfilerUtils, "out_log") as log_mock:
            self.dp_inst._append_cfg_ctxs([ctx])
        self.assertEqual([], self.dp_inst._cfg_ctx_queue)
        self.assertNotIn("stale1", self.dp_inst._seen_trace_ids)
        self.assertEqual(0, self.dp_inst._rejected_cfg_count)
        self.assertTrue(
            any("expired config(s) on arrival" in t and "stale1" in t
                for t in _log_texts(log_mock))
        )

    def test_expired_arrivals_do_not_consume_queue_slots(self):
        self.dp_inst.cur_step = 100
        self.dp_inst.cfg_ctx = self._make_ctx("ready1", start_step=1000)
        with patch.object(DynamicProfilerUtils, "out_log"):
            for i in range(self.dp_inst.MAX_CONFIG_QUEUE_SIZE):
                self.dp_inst._append_cfg_ctxs(
                    [self._make_ctx(f"stale{i}", start_step=1)])
            self.dp_inst._append_cfg_ctxs(
                [self._make_ctx("valid1", start_step=2000),
                 self._make_ctx("instant1", start_step=-1)])
        self.assertEqual(["valid1", "instant1"], self._queue_trace_ids())
        self.assertEqual(0, self.dp_inst._rejected_cfg_count)

    def test_file_mode_latest_config_replaces_pending_head(self):
        original_model = DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL
        DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = (
            DynamicProfilerUtils.DynamicProfilerConfigModel.CFG_CONFIG)
        try:
            self.dp_inst.cur_step = 0
            self.dp_inst.cfg_ctx = self._make_ctx("A", start_step=1000)
            self.dp_inst._dynamic_monitor = MagicMock()
            self.dp_inst._dynamic_monitor.shm_to_prof_conf_context.return_value = [
                self._make_ctx("B", start_step=500)]
            with patch.object(DynamicProfilerUtils, "out_log") as log_mock, \
                    patch.object(self.dp_inst, "enable_prof") as enable_mock:
                self._step()
            self.assertIsNotNone(self.dp_inst.cfg_ctx)
            self.assertEqual("B", self.dp_inst.cfg_ctx.trace_id())
            self.assertEqual([], self.dp_inst._cfg_ctx_queue)
            enable_mock.assert_not_called()
            self.assertTrue(
                any("Replace pending config" in t for t in _log_texts(log_mock))
            )
        finally:
            DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = original_model

    @staticmethod
    def _make_dyno_ctx(trace_id, start_step):
        return ConfigContext({
            "is_valid": True,
            "PROFILE_ACTIVITIES": "CPU,NPU",
            "PROFILE_START_STEP": str(start_step),
            "REQUEST_TRACE_ID": trace_id,
        })

    def test_dyno_mode_new_config_queues_behind_pending_head(self):
        original_model = DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL
        DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = (
            DynamicProfilerUtils.DynamicProfilerConfigModel.DYNO_CONFIG)
        try:
            self.dp_inst.cur_step = 0
            self.dp_inst.cfg_ctx = self._make_dyno_ctx("A", start_step=1000)
            self.dp_inst._dynamic_monitor = MagicMock()
            self.dp_inst._dynamic_monitor.shm_to_prof_conf_context.return_value = [
                self._make_dyno_ctx("B", start_step=500)]
            with patch.object(DynamicProfilerUtils, "out_log"), \
                    patch.object(self.dp_inst, "enable_prof") as enable_mock:
                self._step()
            self.assertEqual("A", self.dp_inst.cfg_ctx.trace_id())
            self.assertEqual(["B"], self._queue_trace_ids())
            enable_mock.assert_not_called()
        finally:
            DynamicProfilerUtils.DYNAMIC_PROFILER_MODEL = original_model


if __name__ == "__main__":
    run_tests()
