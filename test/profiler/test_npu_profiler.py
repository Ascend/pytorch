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

import unittest
import os
import json

import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils._path_manager import PathManager

worker_id = 1


class SmallModel(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=12):
        super(SmallModel, self).__init__()
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
        inputs = torch.rand(self.input_shape, requires_grad=True).to(self.device)
        inputs.register_hook(lambda grad: print("tersor backward hook"))
        target = torch.rand(self.out_shape).reshape(self.out_shape[0], -1).to(self.device)
        output = self.model(inputs)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TestNpuProfiler(TestCase):
    TRACE_FILE_NAME = "trace_view.json"
    KERNEL_FILE_NAME = "kernel_details.csv"
    OPERATOR_FILE_NAME = "operator_details.csv"
    OPERATOR_MEMORY = "operator_memory.csv"
    MEMORY_RECORD = "memory_record.csv"
    STACK_FILE_NAME = "profiler_stacks.log"
    METADATA_FILE_NAME = "profiler_metadata.json"
    results_path = "./results"
    results_work_path = "./work_result_path"
    model_train = TrainModel()
    small_steps = 1
    large_steps = 5

    @property
    def worker_name(self):
        global worker_id
        worker_name = f"npu_profiler_test{worker_id}"
        worker_id += 1
        return worker_name

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TestNpuProfiler.results_path):
            PathManager.remove_path_safety(TestNpuProfiler.results_path)
        if os.path.exists(TestNpuProfiler.results_work_path):
            PathManager.remove_path_safety(TestNpuProfiler.results_work_path)

    def test_default_profiler(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.TRACE_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.KERNEL_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.OPERATOR_FILE_NAME))
        # self.assertEqual(True, self._check_trace_view_keywords(self.results_path, worker_name, ["async_npu"]))

    def test_start_stop_profiler(self):
        worker_name = self.worker_name
        prof = torch_npu.profiler.profile(
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=2),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name))
        prof.start()
        for step in range(self.large_steps):
            self.model_train.train_one_step()
            prof.step()
        prof.stop()
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.TRACE_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.KERNEL_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.OPERATOR_FILE_NAME))
        # self.assertEqual(True, self._check_trace_view_keywords(self.results_path, worker_name, ["async_npu"]))

    def test_activities_cpu(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.TRACE_FILE_NAME))
        self.assertEqual(False, self._has_view_result(self.results_path, worker_name, self.KERNEL_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.OPERATOR_FILE_NAME))
        self.assertEqual(False, self._check_trace_view_keywords(self.results_path, worker_name, ["async_npu"]))

    def test_activities_npu(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        # self.assertEqual(True, self._has_view_result(worker_name, self.TRACE_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.KERNEL_FILE_NAME))
        self.assertEqual(False, self._has_view_result(self.results_path, worker_name, self.OPERATOR_FILE_NAME))
        # self.assertEqual(False, self._check_trace_view_keywords(worker_name, ["async_npu"]))

    def test_record_shapes(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                record_shapes=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.TRACE_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.KERNEL_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.OPERATOR_FILE_NAME))
        self.assertEqual(True, self._check_trace_view_keywords(self.results_path, worker_name, ["Input Dims", "Input type"]))

    def test_with_stack(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                with_stack=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.TRACE_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.KERNEL_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.OPERATOR_FILE_NAME))
        self.assertEqual(True, self._check_trace_view_keywords(self.results_path, worker_name, ["python_function", "built-in function print"]))

    def test_schedule(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=2),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.large_steps):
                self.model_train.train_one_step()
                prof.step()
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.TRACE_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.KERNEL_FILE_NAME))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.OPERATOR_FILE_NAME))

    def test_export_chrome_trace(self):
        PathManager.remove_path_safety(self.results_work_path)
        os.environ["ASCEND_WORK_PATH"] = self.results_work_path
        trace_path = f"{self.results_work_path}/chrome_trace.json"
        with torch_npu.profiler.profile() as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        prof.export_chrome_trace(trace_path)
        os.environ["ASCEND_WORK_PATH"] = ""
        self.assertEqual(True, os.path.isfile(trace_path))

    def test_memory_view(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                profile_memory=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.OPERATOR_MEMORY))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.MEMORY_RECORD))

    def test_memory_when_workspace(self):
        original_value = os.environ.get("TASK_QUEUE_ENABLE")
        os.environ["TASK_QUEUE_ENABLE"] = "2"
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                profile_memory=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for _ in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.OPERATOR_MEMORY))
        self.assertEqual(True, self._has_view_result(self.results_path, worker_name, self.MEMORY_RECORD))
        if original_value is None:
            del os.environ["TASK_QUEUE_ENABLE"]
        else:
            os.environ["TASK_QUEUE_ENABLE"] = original_value

    def test_ascend_work_path(self):
        PathManager.remove_path_safety(self.results_work_path)
        os.environ["ASCEND_WORK_PATH"] = self.results_work_path
        with torch_npu.profiler.profile(
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler()
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()

        os.environ["ASCEND_WORK_PATH"] = ""
        self.assertEqual(True, os.path.exists(os.path.join(self.results_work_path, "profiling_data")))

    def test_add_metadata(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
                prof.add_metadata("test_key1", "test_val1")
                prof.add_metadata_json("test_key2", "[1,2, 3]")
        works = os.listdir(self.results_path)
        work_name = None
        for work in works:
            if worker_name in work:
                work_name = work
                break
        self.assertEqual(False, work_name is None)
        fw_path = os.path.join(self.results_path, work_name)
        fname = os.path.join(fw_path, self.METADATA_FILE_NAME)
        with open(fname) as fp:
            data = json.load(fp)
        has_key1 = "test_key1" in data
        has_key2 = "test_key2" in data
        match_val1 = data["test_key1"] == "test_val1"
        match_val2 = data["test_key2"] == [1, 2, 3]
        self.assertEqual(True, has_key1)
        self.assertEqual(True, has_key2)
        self.assertEqual(True, match_val1)
        self.assertEqual(True, match_val2)

    def test_export_stacks(self):
        PathManager.remove_path_safety(self.results_work_path)
        os.environ["ASCEND_WORK_PATH"] = self.results_work_path
        with torch_npu.profiler.profile(
            with_stack=True
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        stack_path = os.path.join(self.results_work_path, self.STACK_FILE_NAME)
        PathManager.remove_path_safety(stack_path)
        prof.export_stacks(stack_path)
        os.environ["ASCEND_WORK_PATH"] = ""
        with open(stack_path) as fp:
            lines = fp.readlines()
        not_empty = len(lines) > 0
        self.assertEqual(True, not_empty)
        for line in lines:
            is_int = False
            try:
                metrics = int(line.split(" ")[-1])
                is_int = True
            except ValueError:
                pass
            self.assertEqual(True, is_int)
            self.assertEqual(True, metrics > 0)

    def test_set_step_num_offset_for_dynamic_profiler(self):
        worker_name = self.worker_name
        prof = torch_npu.profiler.profile(
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=2),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name))
        prof._set_step_num_offset_for_dynamic_prof(10)
        self.assertEqual(10, prof._step_num_offset)

    def test_kineto_start_stop(self):
        PathManager.remove_path_safety(self.results_work_path)
        os.environ["ASCEND_WORK_PATH"] = self.results_work_path
        prof = torch_npu.profiler._KinetoProfile(with_stack=True)
        prof.start()
        self.model_train.train_one_step()
        prof.stop()
        stack_path = os.path.join(self.results_work_path, self.STACK_FILE_NAME)
        trace_path = os.path.join(self.results_work_path, self.TRACE_FILE_NAME)
        PathManager.remove_path_safety(stack_path)
        PathManager.remove_path_safety(trace_path)
        prof.export_stacks(stack_path)
        prof.export_chrome_trace(trace_path)
        os.environ["ASCEND_WORK_PATH"] = ""
        self.assertEqual(True, os.path.isfile(stack_path))
        self.assertEqual(True, os.path.isfile(trace_path))

    def test_offline_analyse(self):
        PathManager.remove_path_safety(self.results_work_path)
        os.environ["ASCEND_WORK_PATH"] = self.results_work_path
        prof = torch_npu.profiler._KinetoProfile()
        prof.start()
        self.model_train.train_one_step()
        prof.stop()
        result_dir = os.path.join(self.results_work_path, "profiling_data")
        torch_npu.profiler.profiler.analyse(result_dir, export_type="text")
        work_names = [p for p in os.listdir(result_dir) if p.endswith("ascend_pt")]
        os.environ["ASCEND_WORK_PATH"] = ""
        # only one device
        valid_work_name = len(work_names) == 1 and work_names[0].endswith("ascend_pt")
        self.assertEqual(True, valid_work_name)
        self.assertEqual(True, self._has_view_result(result_dir, work_names[0], self.TRACE_FILE_NAME))
        self.assertEqual(True, self._has_view_result(result_dir, work_names[0], self.KERNEL_FILE_NAME))
        self.assertEqual(True, self._has_view_result(result_dir, work_names[0], self.OPERATOR_FILE_NAME))

    def _get_tensorboard_output(self, dir_name: str, worker_name: str) -> str:
        sub_dirs = os.listdir(os.path.realpath(dir_name))
        for sub_dir in sub_dirs:
            if sub_dir.find(worker_name) != -1:
                return os.path.join(dir_name, sub_dir, "ASCEND_PROFILER_OUTPUT")
        return ""

    def _has_view_result(self, dir_name: str, worker_name: str, view_name: str) -> bool:
        output_path = self._get_tensorboard_output(dir_name, worker_name)
        if os.path.isdir(output_path):
            return os.path.isfile(os.path.join(output_path, view_name))
        return False

    def _check_trace_view_keywords(self, dir_name: str, worker_name: str, keywords: list) -> bool:
        if not self._has_view_result(dir_name, worker_name, self.TRACE_FILE_NAME):
            return False
        trace_path = os.path.realpath(os.path.join(self._get_tensorboard_output(dir_name, worker_name), self.TRACE_FILE_NAME))
        file_size = os.path.getsize(trace_path)
        if file_size <= 0:
            return False
        PathManager.check_directory_path_readable(trace_path)
        with open(trace_path, "rt") as file:
            all_data = file.read()
            return all(all_data.find(keyword) != -1 for keyword in keywords)
        return False


if __name__ == "__main__":
    run_tests()
