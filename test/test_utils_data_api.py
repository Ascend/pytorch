"""
Add validation cases for torch.utils.data worker/control APIs:
1. PyTorch community tests cover several worker/control APIs through DataLoader call chains, so this file adds direct API validations.
2. This file validates _MultiProcessingDataLoaderIter, _InfiniteConstantSampler, ManagerWatchdog, _IterableDatasetStopIteration, _ResumeIteration, _set_worker_signal_handlers, and _remove_worker_pids (extendable).
"""

import os

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.data import DataLoader, Dataset, dataloader
from torch.utils.data._utils import signal_handling, worker


class _TinyMapDataset(Dataset):

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return f"sample-{index}"


class TestUtilsDataWorkerControlAPIs(TestCase):

    def test_infinite_constant_sampler_yields_none(self):
        sampler_iter = iter(dataloader._InfiniteConstantSampler())

        self.assertEqual([next(sampler_iter) for _ in range(3)], [None, None, None])

    def test_worker_control_message_fields(self):
        stop_message = worker._IterableDatasetStopIteration(worker_id=2)
        resume_message = worker._ResumeIteration(seed=123)

        self.assertEqual(stop_message.worker_id, 2)
        self.assertEqual(resume_message.seed, 123)
        self.assertIn("worker_id=2", repr(stop_message))
        self.assertIn("seed=123", repr(resume_message))

    def test_manager_watchdog_reports_parent_alive(self):
        watchdog = worker.ManagerWatchdog()

        self.assertTrue(watchdog.is_alive())

    def test_worker_signal_handlers_and_pid_cleanup(self):
        loader_id = id(self)

        self.assertIsNone(signal_handling._set_worker_signal_handlers())
        self.assertIsNone(signal_handling._set_worker_pids(loader_id, (os.getpid(),)))
        self.assertIsNone(signal_handling._remove_worker_pids(loader_id))

    def test_multiprocessing_dataloader_iter_type_and_shutdown(self):
        loader = DataLoader(_TinyMapDataset(), batch_size=2, num_workers=1)
        iterator = iter(loader)
        try:
            self.assertIsInstance(iterator, dataloader._MultiProcessingDataLoaderIter)
            self.assertEqual(next(iterator), ["sample-0", "sample-1"])
        finally:
            iterator._shutdown_workers()


if __name__ == "__main__":
    run_tests()
