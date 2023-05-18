import os
import shutil

import torch

import torch_npu
from torch_npu.testing.testcase import TestCase

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
        inputs = torch.rand(self.input_shape).to(self.device)
        target = torch.rand(self.out_shape).reshape(self.out_shape[0], -1).to(self.device)
        output = self.model(inputs)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TestNpuProfiler(TestCase):
    TRACE_FILE_NAME = "trace_view.json"
    KERNEL_FILE_NAME = "kernel_details.csv"
    results_path = "./results"
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
            shutil.rmtree(TestNpuProfiler.results_path)

    def test_default_profiler(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_trace_view(worker_name))
        self.assertEqual(True, self._has_kernel_details(worker_name))
        self.assertEqual(True,
                         self._check_trace_view_keywords(worker_name, ["torch_to_acl", "torch_to_npu", "acl_to_npu"]))

    def test_activities_cpu(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_trace_view(worker_name))
        self.assertEqual(False, self._has_kernel_details(worker_name))
        self.assertEqual(False,
                         self._check_trace_view_keywords(worker_name, ["torch_to_acl", "torch_to_npu", "acl_to_npu"]))

    def test_activities_npu(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_trace_view(worker_name))
        self.assertEqual(True, self._has_kernel_details(worker_name))
        self.assertEqual(True, self._check_trace_view_keywords(worker_name, ["acl_to_npu"]))
        self.assertEqual(False, self._check_trace_view_keywords(worker_name, ["torch_to_acl", "torch_to_npu"]))

    def test_record_shapes(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                record_shapes=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_trace_view(worker_name))
        self.assertEqual(True, self._has_kernel_details(worker_name))
        self.assertEqual(True, self._check_trace_view_keywords(worker_name, ["Input Dims", "Input type"]))

    def test_with_stack(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                with_stack=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        self.assertEqual(True, self._has_trace_view(worker_name))
        self.assertEqual(True, self._has_kernel_details(worker_name))
        self.assertEqual(True, self._check_trace_view_keywords(worker_name, ["Call stack"]))

    def test_schedule(self):
        worker_name = self.worker_name
        with torch_npu.profiler.profile(
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=2),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.results_path, worker_name=worker_name)
        ) as prof:
            for step in range(self.large_steps):
                self.model_train.train_one_step()
                prof.step()
        self.assertEqual(True, self._has_trace_view(worker_name))
        self.assertEqual(True, self._has_kernel_details(worker_name))

    def test_export_chrome_trace(self):
        trace_path = f"{self.results_path}/chrome_trace.json"
        with torch_npu.profiler.profile() as prof:
            for step in range(self.small_steps):
                self.model_train.train_one_step()
        prof.export_chrome_trace(trace_path)
        self.assertEqual(True, os.path.isfile(trace_path))

    def _get_tensorboard_output(self, worker_name: str) -> str:
        sub_dirs = os.listdir(os.path.realpath(self.results_path))
        for sub_dir in sub_dirs:
            if sub_dir.find(worker_name) != -1:
                return os.path.join(self.results_path, sub_dir, "ASCEND_PROFILER_OUTPUT")
        return ""

    def _has_trace_view(self, worker_name: str) -> bool:
        output_path = self._get_tensorboard_output(worker_name)
        if os.path.isdir(output_path):
            return os.path.isfile(os.path.join(output_path, self.TRACE_FILE_NAME))
        return False

    def _has_kernel_details(self, worker_name: str) -> bool:
        output_path = self._get_tensorboard_output(worker_name)
        if os.path.isdir(output_path):
            return os.path.isfile(os.path.join(output_path, self.KERNEL_FILE_NAME))
        return False

    def _check_trace_view_keywords(self, worker_name: str, keywords: list) -> bool:
        if not self._has_trace_view(worker_name):
            return False
        trace_path = os.path.join(self._get_tensorboard_output(worker_name), self.TRACE_FILE_NAME)
        file_size = os.path.getsize(trace_path)
        if file_size <= 0:
            return False
        with open(trace_path, "rt") as file:
            all_data = file.read()
            return all(all_data.find(keyword) != -1 for keyword in keywords)
