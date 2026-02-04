import torch


def npu_create_tagged_event(tag: str):
    from torch_npu.dynamo.torchair import ops
    return ops.npu_create_tagged_event(tag=tag)


def npu_tagged_event_record(event):
    from torch_npu.dynamo.torchair import ops
    return ops.npu_tagged_event_record(event=event)


def npu_tagged_event_wait(event):
    from torch_npu.dynamo.torchair import ops
    return ops.npu_tagged_event_wait(event=event)


def npu_record_tagged_stream(input: torch.Tensor, tagged_stream: str):
    from torch_npu.dynamo.torchair import ops
    return ops.npu_record_tagged_stream(input=input, tagged_stream=tagged_stream)


def record():
    from torch_npu.dynamo.torchair import ops
    return ops.record()


def wait(tensors: list):
    from torch_npu.dynamo.torchair import ops
    return ops.wait(tensors=tensors)