import os
import queue
import threading

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29529'


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


def train(rank, world_size, error_queue):
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)

    # Create the model and move it to the appropriate device
    model = SimpleModel().npu()
    ddp_model = DDP(model, device_ids=[rank])
    data = torch.randn(100, 10)
    targets = torch.randn(100, 1)
    # Create a simple dataset and dataloader
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=10)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch in range(2):
        for i, (inputs, labels) in enumerate(dataloader):
            try:
                if rank == 1 and epoch == 0 and i == 2:
                    raise RuntimeError("UCE ERROR")
                inputs, labels = inputs.to('npu'), labels.to('npu')
                optimizer.zero_grad()
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                torch.npu.synchronize()

                print(f"Rank {rank}, Epoch [{epoch+1}/2], Iter [{i+1}/10], Loss: {loss.item()}")

            except RuntimeError as e:
                print(f"Rank {rank}: Caught RuntimeError during training")
                if "UCE ERROR" in str(e):
                    print(f"Rank {rank}: Detected UCE ERROR")
                    try:
                        error_queue.put((rank, str(e)))
                        continue
                    except Exception as queue_error:
                        raise queue_error
                if "FORCE STOP" in str(e):
                    try:
                        error_queue.put((rank, str(e)))
                    except Exception as queue_error:
                        raise queue_error
                else:
                    raise e  # Reraise other uncaught exceptions

    # Destroy the process group
    dist.destroy_process_group()


def monitor(error_queue, stop_event):
    while not stop_event.is_set():
        try:
            rank, error_msg = error_queue.get(timeout=1)
            if "UCE ERROR" in error_msg:
                torch_npu.npu.stop_device(0)
                torch_npu.npu.stop_device(1)
            if "FORCE STOP" in error_msg:
                if not torch_npu.npu.check_uce_in_memory(0):
                    torch_npu.npu.restart_device(0)
                if not torch_npu.npu.check_uce_in_memory(1):
                    torch_npu.npu.restart_device(1)
        except queue.Empty:
            continue
        except Exception as e:
            raise e


def run(rank, world_size, error_queue):
    # Create the monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor, args=(error_queue, stop_event))
    monitor_thread.start()

    # Start training
    train(rank, world_size, error_queue)

    # After training ends, set the stop event and wait for the monitor thread to finish
    stop_event.set()
    monitor_thread.join()


def main():
    world_size = 2
    ctx = mp.get_context('spawn')
    error_queue = ctx.Queue(4)
    mp.spawn(run, args=(world_size, error_queue), nprocs=world_size, join=True)


class TestDistributedTraining(TestCase):

    def test_distributed_training(self):
        main()
