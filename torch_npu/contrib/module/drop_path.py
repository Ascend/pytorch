import torch
import torch.nn as nn
import torch_npu


def npu_drop_path(x, random_tensor, keep_prob: float = 0.):
    """Less ops than timm version.
    Async generating and applying of random tensor for accelerating.
    """
    random_tensor += keep_prob
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPathTask:
    def __init__(self, shape, device, dtype, ndim, drop_prob):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.ndim = ndim
        self.drop_prob = drop_prob
        self.request_count = 0
        self.rand_queue = []


class NpuDropPath(nn.Module):
    """Using NPU affinity writing method to replace the native Drop paths in swin_transformer.py.
    
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks.)

    .. note::
        Dynamic shapes are not supported.

    Args:
        drop_prob (float): the dropout probabilities.
        x (Tensor): The input tensor to apply dropout.

    Examples::
        >>> input1 = torch.randn(68, 5).npu()
        >>> input1.requires_grad_(True)
        >>> input2 = torch.randn(68, 5).npu()
        >>> input2.requires_grad_(True)
        >>> fast_drop_path = NpuDropPath(0).npu()
        >>> output = input1 + fast_drop_path(input2)
        >>> output.sum().backward()
    """
    task_dict = {}
    droppath_stream = None

    def __init__(self, drop_prob=None):
        super(NpuDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        if isinstance(x, torch.Tensor):
            shape = x.shape
            dtype = x.dtype
            device = x.device
            ndim = x.ndim
        else:
            raise RuntimeError("input type error!")

        key = (shape, device, dtype, ndim)
        if key not in NpuDropPath.task_dict:
            droppath_task = DropPathTask(shape, device, dtype, ndim, self.drop_prob)
            droppath_task.request_count += 1
            NpuDropPath.task_dict[key] = droppath_task
        elif not NpuDropPath.task_dict[key].rand_queue:
            NpuDropPath.task_dict[key].request_count += 1
        else:
            random_tensor = NpuDropPath.task_dict[key].rand_queue.pop(0)
            return npu_drop_path(x, random_tensor, self.keep_prob)

        return x

    @classmethod
    def enable_droppath_ensemble(cls, model):
        if cls.droppath_stream is None:
            cls.droppath_stream = torch.npu.Stream()

        def wait_stream_hook():
            def hook_function(module, inputs):
                torch.npu.current_stream().wait_stream(cls.droppath_stream)
            return hook_function
        model.register_forward_pre_hook(wait_stream_hook())

        def random_tensor_gen_hook():
            def hook_function(module, inputs):
                with torch.npu.stream(cls.droppath_stream):
                    with torch.no_grad():
                        for _, task in cls.task_dict.items():
                            if len(task.rand_queue) >= task.request_count:
                                continue
                            for i in range(task.request_count - len(task.rand_queue)):
                                shape = (task.shape[0],) + (1,) * (task.ndim - 1)  
                                random_tensor = torch.rand(shape, dtype=task.dtype, device=task.device)
                                task.rand_queue.append(random_tensor)
            return hook_function
        model.register_forward_pre_hook(random_tensor_gen_hook())
