# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

import os
import threading
import itertools
import queue
import functools
import torch
import torch.distributed as dist
from torch.utils.data import _utils, IterDataPipe, MapDataPipe
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter as SrcSingleProcessDataLoaderIter
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as SrcMultiProcessingDataLoaderIter
from torch.utils.data.dataloader import DataLoader as SrcDataLoader
from torch.utils.data.dataloader import (_DatasetKind, _share_dist_seed, _sharding_worker_init_fn,
                                         _get_distributed_settings)
from torch._utils import ExceptionWrapper
import torch.multiprocessing as multiprocessing
import torch_npu

MP_STATUS_CHECK_INTERVAL = 5.0


def _pin_memory_loop(in_queue, out_queue, device_id, done_event):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.

    torch.set_num_threads(1)
    if torch_npu.npu.is_available():
        torch_npu.npu.set_device(device_id)
    else:
        torch.cuda.set_device(device_id)

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = _utils.pin_memory.pin_memory(data)
            except Exception:
                data = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(device_id))
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        del r  # save memory


def npu_worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                  auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                  num_workers, persistent_workers):
    # valid only in DVPP acceleration scenario
    os.environ["TE_PARALLEL_COMPILER"] = "1" # set the number of operator compilation processes to 1
    torch_npu.npu.set_device(dataset.device)
    torch_npu.npu.current_stream().set_data_preprocess_stream(True)
    torch.npu.set_compile_mode(jit_compile=False)
    _utils.worker._worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                               auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                               num_workers, persistent_workers)


class DataLoader(SrcDataLoader):
    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)


class _SingleProcessDataLoaderIter(SrcSingleProcessDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        self._pin_memory = loader.pin_memory and torch_npu.npu.is_available()
        if self._timeout != 0:
            raise ValueError("self._timeout != 0")
        if self._num_workers != 0:
            raise ValueError("self._num_workers != 0")

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)


class _MultiProcessingDataLoaderIter(SrcMultiProcessingDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

    def __init__(self, loader):
        torch_npu.npu.synchronize()
        self._dataset = loader.dataset
        self._shared_seed = None
        self._pg = None
        if isinstance(self._dataset, IterDataPipe):
            if dist.is_available() and dist.is_initialized():
                self._pg = dist.new_group(backend="gloo")
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._pin_memory = loader.pin_memory and torch_npu.npu.is_available()
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(self.__class__.__name__)
        ws, rank = _get_distributed_settings()
        self._world_size = ws
        self._rank = rank

        if self._num_workers <= 0:
            raise ValueError("self._num_workers <= 0")
        if self._prefetch_factor <= 0:
            raise ValueError("self._prefetch_factor <= 0")

        worker_loop = _utils.worker._worker_loop
        daemon = True
        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
            # if enable dvpp, worker process start method should be spawn and cannot be daemonic
            if hasattr(self._dataset, 'accelerate_enable') and self._dataset.accelerate_enable:
                multiprocessing_context = multiprocessing.get_context('spawn')
                worker_loop = npu_worker_loop # set device and priority
                daemon = False
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Additional worker init function will take care of sharding in MP and Distributed
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            self._worker_init_fn = functools.partial(
                _sharding_worker_init_fn, self._worker_init_fn, self._world_size, self._rank)
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()
        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()  # type: ignore
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._collate_fn, self._drop_last,
                      self._base_seed, self._worker_init_fn, i, self._num_workers,
                      self._persistent_workers, self._shared_seed))
            w.daemon = daemon
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            train_device_id = 0
            if torch_npu.npu.is_available():
                train_device_id = torch_npu.npu.current_device()
            else:
                train_device_id = torch.cuda.current_device()
            self._pin_memory_thread_done_event = threading.Event()

            self._data_queue = queue.Queue()  # type: ignore
            pin_memory_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      train_device_id,
                      self._pin_memory_thread_done_event))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit
            for w in self._workers:
                atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _shutdown_workers(self):
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        if _utils is None or _utils.python_exit_status is True or _utils.python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Normal exit when last reference is gone / iterator is depleted.
                # See (1) and the second half of the note.

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, '_pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    # We should be able to join here, but in case anything went
                    # wrong, we set a timeout and if the workers fail to join,
                    # they are killed in the `finally` block.
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False
                for w in self._workers:
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()

    # staticmethod is used to remove reference to `_MultiProcessingDataLoaderIter`
    @staticmethod
    def _clean_up_worker(w):
        try:
            w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
        finally:
            if w.is_alive():
                w.terminate()

    def __del__(self):
        self._shutdown_workers()


def add_dataloader_method():
    torch.utils.data.DataLoader = DataLoader
    torch.utils.data.dataloader.DataLoader = DataLoader
