class TorchAclBean:
    def __init__(self, data: dict):
        self._origin_data = data

    @property
    def acl_start_time(self):
        return self._origin_data.get("acl_start_time")

    @property
    def op_name(self):
        return self._origin_data.get("op_name")

    @property
    def torch_op_start_time(self):
        return self._origin_data.get("torch_op_start_time")

    @property
    def torch_op_tid(self):
        return self._origin_data.get("torch_op_tid")

    @property
    def torch_op_pid(self):
        return self._origin_data.get("torch_op_pid")

    @property
    def npu_kernel_list(self):
        return [data.replace("-", "_") for data in self._origin_data.get("npu_kernel_list")]
