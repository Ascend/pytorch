def register_dynamo_backends():
    from torch_npu.dynamo import _register_backends

    _register_backends()


def register_dynamo_trace_rules():
    """
    # Support stream into Dynamo charts. Enable Dynamo to recognize NPU
    stream/device/memory/random APIs and related torch_npu._C bindings during graph capture.
    """
    from torch_npu.dynamo.trace_rule import _patch_npu_trace_rules

    _patch_npu_trace_rules()
