import threading
import torch
from torch.utils.data import Dataset

from torch_npu.testing.testcase import TestCase, run_tests


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data[index].clone()


def thread_worker(data, device, results, index):
    for _ in range(50):
        data = data.to(device, non_blocking=True)
        results.append(data)


class TestModel:

    def run(self):
        torch.npu.set_device(0)
        device = torch.device(f'npu:{0}')

        dataset = RandomDataset(1000, 1000)

        results = [None] * len(dataset)

        threads = []
        batch_size = 64
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            thread = threading.Thread(target=thread_worker, args=(batch, device, results, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


class AllocatorMultiThreadProf(TestCase):
    test_model = TestModel()

    def test_model_run_succ(self):
        res = True
        try:
            self.test_model.run()
        except Exception as e:
            res = False
        self.assertEqual(res, True)


if __name__ == "__main__":
    run_tests()