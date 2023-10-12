import torch


class Prefetcher(object):
    """Prefetcher using on npu device.


    Args:
        loder (torch.utils.data.DataLoader or DataLoader like iterator):
            Using to generate inputs after preprocessing.
        stream (torch.npu.Stream): Default None.
            Because of the limitation of NPU's memory mechanism,
            if prefetcher is initialized repeatedly during training,
            a defined stream should be introduced to prevent memory leakage;
            if prefetcher is initialized only once during training,
            a defined stream is not necessary.

    Returns:
        float: tensors of shape (k, 5) and (k, 1). Labels are 0-based.
    """

    def __init__(self, loader, stream=None):
        self.loader = iter(loader)
        self.stream = stream if stream is not None else torch.npu.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.npu.stream(self.stream):
            self.next_input = self.next_input.npu(non_blocking=True)
            self.next_target = self.next_target.npu(non_blocking=True)

    def next(self):
        torch.npu.current_stream().wait_stream(self.stream)
        next_input = self.next_input
        next_target = self.next_target
        if next_target is not None:
            self.preload()
        return next_input, next_target
