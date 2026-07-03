# Memory Sharing (IPC)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T07:50:12.877Z pushedAt=2026-06-15T12:00:44.082Z -->

## Introduction

Inter-process communication (IPC) enables direct access to device memory between processes without explicit data copying. This feature is developed based on the [community IPC feature](https://docs.pytorch.org/docs/2.8/multiprocessing.html) and supports the usage patterns of the community IPC feature. The overall workflow is as follows:

1. The sender encapsulates the memory information of the tensor or storage into a handle, packages it together with necessary information such as storage_size and storage_offset, as well as a reconstruction function, and returns the package for the receiver to reconstruct;
2. After obtaining the above information, the cross-process receiver uses the reconstruction function and parameters such as the handle to restore the original tensor or storage object.

## Use Scenario

In reinforcement learning scenarios involving both training and inference, this feature can be used to reduce frequent copying of weights and lower device memory usage.

## Usage Guide

Choose one of the following methods based on your actual situation:

- Supports direct IPC sharing of NPU tensors through `multiprocessing.Queue`.
- Supports reducing NPU tensors through `torch.multiprocessing.reductions.reduce_tensor` first, and then performing IPC sharing. This method allows the receiver to modify the device card number for cross-card access.

## Usage Example

- Share an NPU Tensor via IPC using multiprocessing.Queue. The sample script is as follows:

    ```python
    import torch
    import torch_npu
    import torch.multiprocessing as mp
    from multiprocessing import Queue
     
    def worker(tensor_queue):
        tensor = tensor_queue.get()
        print(f"Receiver received tensor = {tensor}")
     
     
    if __name__ == '__main__':
        # Must use the spawn method
        mp.set_start_method('spawn')
     
        tensor_queue = Queue()
        p = mp.Process(target=worker, args=(tensor_queue,))
        p.start()
     
        tensor = torch.full((5,), float(1.0), device='npu:0')
     
        tensor_queue.put(tensor)
        print(f"Sender sent tensor = {tensor}")
     
        p.join()
    ```

    The execution result is as follows:

    ```python
    Sender send tensor = tensor([1., 1., 1., 1., 1.], device='npu:0')
    Receiver receive tensor = tensor([1., 1., 1., 1., 1.], device='npu:0')
    ```

- First reduce the NPU tensors through `torch.multiprocessing.reductions.reduce_tensor`, then perform IPC sharing. The sample script is as follows:

    ```py
    import torch
    import torch_npu
    import torch.multiprocessing as mp
    from torch.multiprocessing.reductions import reduce_tensor
    from multiprocessing import Queue
     
     
    def worker(tensor_queue):
        shared_handles = tensor_queue.get()
        func, args = shared_handles
        print(func)
    list_args = list(args)
    # Modify the target device to card 1 to enable cross-card access
    list_args[6] = 1
    tensor = func(*list_args)
    print(f"Receiver received tensor = {tensor}")
     
     
    if __name__ == '__main__':
        # Must use the spawn method
        mp.set_start_method('spawn')
        tensor_queue = Queue()
        p = mp.Process(target=worker, args=(tensor_queue,))
        p.start()
     
        tensor = torch.full((5, ), float(3.14), device=f"npu:0")
        shared_handle = reduce_tensor(tensor)
        tensor_queue.put(shared_handle)
        print(f"Sender sent tensor = {tensor}")
     
        p.join()
    ```

    The execution result is as follows:

    ```python
    Sender send tensor = tensor([3.1400, 3.1400, 3.1400, 3.1400, 3.1400], device='npu:0')
    Receiver receive tensor = tensor([3.1400, 3.1400, 3.1400, 3.1400, 3.1400], device='npu:1')
    ```

## Constraints

This feature is only supported on Ascend HDK 25.3.RC1 or later and CANN 8.3.RC1 or later.
