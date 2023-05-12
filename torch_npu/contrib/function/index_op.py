import torch
import torch_npu

def npu_fast_condition_index_put(x, condition, value):
    """Using NPU affinity writing method to replace the native writing method in bool type index_put function.

    Examples::
    >>> x = torch.randn(128, 8192)
    >>> condition = x < 0.5
    >>> value = 0.
    >>> x1 = copy.deepcopy(x)[condition] = value
    >>> x1_opt = npu_fast_condition_index_put(x, condition, value)

    .. note::
        Because the index operator has been optimized all the time, the native implementation 
        performance of some scenarios is better.

    Args:
        x (torch.Tensor): Normal tensor.
        condition (torch.BoolTensor): Judgment condition, bool dtype.
        value (int, float): Stride of bboxes. Only IntTensor is supported.

    Returns:
        torch.Tensor: Box transformation deltas
    """

    assert condition.dtype in [torch.bool]

    if value == 0:
        mask = torch.zeros_like(x)
    elif value == 1:
        mask = torch.ones_like(x)
    else:
        mask = torch.zeros_like(x) + value

    x = torch.where(condition, mask, x)
    return x


def _npu_fast_condition_index_put_test():
    x = torch.randn(128, 8192).npu()
    condition = x < 0.5
    value = 0.
    repeat_time = 100
    x1 = copy.deepcopy(x)

    x1[condition] = value
    torch.npu.synchronize()
    t1 = time.time()
    for _ in range(repeat_time):
        x1[condition] = value
    torch.npu.synchronize()
    print('x1[condition] = value time: %.4fms' % ((time.time() - t1) / repeat_time * 1000))

    x1_opt = npu_fast_condition_index_put(x, condition, value)
    torch.npu.synchronize()
    t2 = time.time()
    for _ in range(repeat_time):
        x1_opt = npu_fast_condition_index_put(x, condition, value)
    torch.npu.synchronize()
    print('x1_opt = npu_fast_condition_index_put(x, condition, value) time: %.4fms' % (
            (time.time() - t2) / repeat_time * 1000))

    print('DIFF: ', (x1 - x1_opt).sum())

if __name__ == "__main__":
    import copy
    import time

    torch.npu.set_device(0)

    _npu_fast_condition_index_put_test()
