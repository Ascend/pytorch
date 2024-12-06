import os
import stat

import torch
import torch_npu
from torch_npu.utils._path_manager import PathManager
from torch_npu.profiler._profiler_path_creator import ProfPathCreator
from torch_npu.testing.testcase import TestCase, run_tests


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(8 * 2 * 2, 10)
        self.fc2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 8 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class TrainModel:
    def __init__(self):
        self.device = "npu:0"
        self.model = SimpleCNN().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        self.inputs = torch.randn(1, 3, 8, 8, device=self.device)
        self.labels = torch.rand_like(self.model(self.inputs))

    def train_one_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(self.inputs)
        self.criterion(pred, self.labels).backward()
        self.optimizer.step()


class TestExportMemoryTimeline(TestCase):
    model_train = TrainModel()
    train_steps = 3

    def test_export_memory_timeline_on_npu(self):
        def trace_handler(prof: torch_npu.profiler.profile):
            prof.export_memory_timeline(output_path="./mem_tl.json", device="npu:0")
            prof.export_memory_timeline(output_path="./mem_tl.raw.json.gz", device="npu:0")
        
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=self.train_steps, repeat=1, skip_first=0),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as prof:
            for _ in range(self.train_steps):
                self.model_train.train_one_step()
                prof.step()

        has_prof = False
        prof_dir = ProfPathCreator().get_prof_dir()
        if self.has_prof_dir(prof_dir):
            has_prof = True
        if os.path.exists(prof_dir):
            PathManager.remove_path_safety(prof_dir)
        self.assertTrue(has_prof)

        has_result = False
        json_file = "./mem_tl.json"
        json_gz_file = "./mem_tl.raw.json.gz"
        if os.path.isfile(json_file) and os.path.isfile(json_gz_file):
            has_result = True
            PathManager.remove_file_safety(json_file)
            PathManager.remove_file_safety(json_gz_file)
        self.assertTrue(has_result)
    
    def test_export_memory_timeline_on_cpu(self):
        def trace_handler(prof: torch_npu.profiler.profile):
            prof.export_memory_timeline(output_path="./mem_tl.json", device="cpu")
            prof.export_memory_timeline(output_path="./mem_tl.raw.json.gz", device="cpu")
        
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=self.train_steps, repeat=1, skip_first=0),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as prof:
            for _ in range(self.train_steps):
                self.model_train.train_one_step()
                prof.step()

        has_prof = False
        prof_dir = ProfPathCreator().get_prof_dir()
        if self.has_prof_dir(prof_dir):
            has_prof = True
        if os.path.exists(prof_dir):
            PathManager.remove_path_safety(prof_dir)
        self.assertTrue(has_prof)

        has_result = False
        json_file = "./mem_tl.json"
        json_gz_file = "./mem_tl.raw.json.gz"
        if os.path.isfile(json_file) and os.path.isfile(json_gz_file):
            has_result = True
            PathManager.remove_file_safety(json_file)
            PathManager.remove_file_safety(json_gz_file)
        self.assertTrue(has_result)

    @staticmethod
    def has_prof_dir(path: str) -> bool:
        if path is None:
            return False
        path = os.path.realpath(path)
        if not os.path.exists(path):
            return False
        if path.endswith("_pt"):
            return True
        return False


if __name__ == "__main__":
    run_tests()