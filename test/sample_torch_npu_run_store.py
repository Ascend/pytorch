import os
import time
from datetime import datetime
from datetime import timedelta

import torch
from torch.distributed import PrefixStore
from torch.distributed.rendezvous import rendezvous
import torch_npu


class TorchNpuRunStoreSample:
    def __init__(self, ip: str, port: int):
        self._current_rank = int(os.getenv('RANK', 0))
        self._world_size = int(os.getenv('WORLD_SIZE', 0))
        self._init_method = f'parallel://{ip}:{port}'
        self._timeout = timedelta(minutes=1)
        self._key = 'sample_torch_npu_run_store:test_case_001'
        
        rendezvous_iterator = rendezvous(
            self._init_method, self._current_rank, self._world_size, timeout=self._timeout
        )

        self._store, self._current_rank, self._world_size = next(rendezvous_iterator)
        self._store.set_timeout(self._timeout)
        self._store = PrefixStore("default_pg", self._store)

    def store_based_barrier(self):
        self._store.add(self._key, 1)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        print(f'[{timestamp}]rank: {self._current_rank} add key: {self._key} first time')

        start_time = time.time()
        alive_count = self._store.add(self._key, 0)
        while alive_count != self._world_size:
            time.sleep(0.01)
            alive_count = self._store.add(self._key, 0)
            if alive_count == self._world_size:
                break
            if timedelta(seconds=(time.time() - start_time)) > self._timeout:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                raise RuntimeError(f'[{timestamp}]rank: {self._current_rank} wait all workers ready timeout')
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        print(f'[{timestamp}] Rank: {self._current_rank} complete store-based barrier for worker count: {alive_count}')


if __name__ == "__main__":
    sample = TorchNpuRunStoreSample(ip='127.0.0.1', port=29513)
    sample.store_based_barrier()
