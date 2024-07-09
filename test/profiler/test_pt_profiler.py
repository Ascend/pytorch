import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class SmallModel(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SmallModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel, 1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, input_1):
        input_1 = self.conv1(input_1)
        input_1 = self.relu1(input_1)
        input_1 = self.conv2(input_1)
        return input_1.reshape(input_1.shape[0], -1)


class TestProfiler(TestCase):

    def mm_op(self, device="cpu"):
        a = torch.rand(5, 5).to(device)
        b = torch.randn(5, 5).to(device)
        c = torch.mm(a, b)

    def test_cpu_op_profiler(self):
        with torch.autograd.profiler.profile(use_device=None) as prof:
            self.mm_op()
        found_mm = False

        for e in prof.function_events:
            if "mm" in e.name:
                found_mm = True
        self.assertTrue(found_mm)

    def test_npu_op_profiler(self):
        # test basic function for npu op
        if torch.npu.is_available():
            device = "npu:0"
        else:
            return
        with torch.autograd.profiler.profile(use_device='npu') as prof:
            self.mm_op(device)
        found_mm = False

        for e in prof.function_events:
            if "mm" in e.name:
                found_mm = True
        self.assertTrue(found_mm)

    def train(self, steps):
        input_shape = (4, 3, 24, 24)
        out_shape = (4, 12, 24, 24)
        device = "npu:0" if torch.npu.is_available() else "cpu"
        model = SmallModel(input_shape[1], out_shape[1]).to(device)
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        for index in range(steps):
            x = torch.rand(input_shape).to(device)
            y = torch.rand(out_shape).reshape(out_shape[0], -1).to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_model_profiler(self):
        """Checks that model forward and backward.
        """
        steps = 5
        try:
            self.train(steps)
        except Exception:
            self.assertTrue(False, "Expected no exception without profiling.")

        def judge(expected_event_count, prof):
            actual_event_count = {}
            for e in prof.function_events:
                if "#" in e.name:
                    key = e.name
                    if key in expected_event_count.keys():
                        actual_event_count[key] = actual_event_count.setdefault(key, 0) + 1
            for key, count in expected_event_count.items():
                self.assertTrue((key in actual_event_count.keys()) and (count == actual_event_count.get(key, 0)))

        with torch.autograd.profiler.profile(use_device='npu') as prof:
            self.train(steps)
        expected_event_count = {
            "Optimizer.step#SGD.step": steps,
            "Optimizer.zero_grad#SGD.zero_grad": steps
        }
        judge(expected_event_count, prof)


if __name__ == '__main__':
    run_tests()
