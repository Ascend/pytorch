from torch._inductor.codegen.common import DeviceOpOverrides, register_device_op_overrides


class NPUDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from torch._C import _npu_getCurrentRawStream as {name}"

    def set_device(self, device_idx):
        return f"torch_npu.npu.set_device({device_idx})"

    def synchronize(self):
        return "torch_npu.npu.synchronize()"

    def device_guard(self, device_idx):
        return f"torch_npu.npu._DeviceGuard({device_idx})"


def _inductor_register_device_op_overrides():
    register_device_op_overrides('npu', NPUDeviceOpOverrides())
