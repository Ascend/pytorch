# 内存共享（IPC）

## 简介

IPC（Inter-Process Communication），表示进程间通信，进程间可以直接访问设备内存，而不需要做显式的数据拷贝。该特性基于社区IPC特性（[LINK](https://docs.pytorch.org/docs/2.8/multiprocessing.html)）开发，支持社区IPC特性的使用方式，整体使用流程如下：

1.  发送方将tensor、storage的内存信息封装为handle，结合storage\_size、storage\_offset等必要信息以及重建函数打包返回，用于发送给接收方重构；
2.  跨进程的接收方获得以上信息后，利用重构函数和handle等参数，恢复回原来的tensor、storage对象。

## 使用场景

在强化学习既有训练又有推理的场景可以使用该特性以减少对权重的频繁拷贝，并降低设备内存使用。

## 使用指导

根据实际情况选择如下任一操作方法：

-   支持直接通过multiprocessing.Queue对NPU Tensor进行IPC共享。
-   支持先通过torch.multiprocessing.reductions.reduce\_tensor对NPU Tensor进行reduce化，然后再进行IPC共享。该方式支持在接收方修改设备卡号来跨卡访问。

## 使用样例

-   通过multiprocessing.Queue对NPU Tensor进行IPC共享，样例脚本如下：

    ```py
    import torch
    import torch_npu
    import torch.multiprocessing as mp
    from multiprocessing import Queue
     
    def worker(tensor_queue):
        tensor = tensor_queue.get()
        print(f"接收方收到 tensor = {tensor}")
     
     
    if __name__ == '__main__':
        # 必须使用spawn方法
        mp.set_start_method('spawn')
     
        tensor_queue = Queue()
        p = mp.Process(target=worker, args=(tensor_queue,))
        p.start()
     
        tensor = torch.full((5,), float(1.0), device='npu:0')
     
        tensor_queue.put(tensor)
        print(f"发送方发送 tensor = {tensor}")
     
        p.join()
    ```

    运行结果如下：

    ```
    发送方发送 tensor = tensor([1., 1., 1., 1., 1.], device='npu:0')
    接收方收到 tensor = tensor([1., 1., 1., 1., 1.], device='npu:0')
    ```

-   先通过torch.multiprocessing.reductions.reduce\_tensor对NPU Tensor进行reduce化，然后再进行IPC共享，样例脚本如下：

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
    # 修改目标设备为卡1，从而实现跨卡访问
    list_args[6] = 1
    tensor = func(*list_args)
    print(f"接收方收到 tensor = {tensor}")
     
     
    if __name__ == '__main__':
        # 必须使用spawn方法
        mp.set_start_method('spawn')
        tensor_queue = Queue()
        p = mp.Process(target=worker, args=(tensor_queue,))
        p.start()
     
        tensor = torch.full((5, ), float(3.14), device=f"npu:0")
        shared_handle = reduce_tensor(tensor)
        tensor_queue.put(shared_handle)
        print(f"发送方发送 tensor = {tensor}")
     
        p.join()
    ```

    运行结果如下：

    ```
    发送方发送 tensor = tensor([3.1400, 3.1400, 3.1400, 3.1400, 3.1400], device='npu:0')
    接收方收到 tensor = tensor([3.1400, 3.1400, 3.1400, 3.1400, 3.1400], device='npu:1')
    ```

## 约束说明

该特性仅支持在Ascend HDK 25.3.RC1及以上且CANN 8.3.RC1及以上版本使用。

