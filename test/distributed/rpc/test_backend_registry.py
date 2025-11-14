import torch
import torch.distributed.rpc as rpc
from torch._C import _get_privateuse1_backend_name

from torch.distributed.rpc import api
from torch.distributed.rpc import constants as rpc_constants

import torch_npu._C
from torch_npu.utils._error_code import ErrCode, dist_error
from torch_npu.distributed.rpc.backend_registry import (
    _get_device_count_info, _init_device_state, _tensorpipe_validate_devices, _validate_device_maps,
    _get_device_infos, _tensorpipe_exchange_and_check_all_device_maps, _set_devices_and_reverse_device_map,
    _backend_type_repr, _construct_rpc_backend_options, _init_backend,
    _npu_tensorpipe_construct_rpc_backend_options_handler,
    _npu_tensorpipe_init_backend_handler, _rpc_backend_registry)
import torch_npu.distributed.rpc.backend_registry as backend_registry
from torch_npu.testing.testcase import TestCase, run_tests


class TestBackendRegistry(TestCase):
    def test_validate_device_maps_invalid_target_nodes(self):
        all_names = ['node1', 'node2']
        all_device_counts = {'node1': {'npu': 2}, 'node2': {'npu': 2}}
        all_device_maps = {'node1': {'node3': {torch.device('npu:0'): torch.device('npu:1')}}}
        all_devices = {'node1': [torch.device('npu:0')], 'node2': [torch.device('npu:1')]}
        with self.assertRaises(ValueError) as context:
            _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices)
        self.assertIn("invalid target node names", str(context.exception))

    def test_validate_device_maps_duplicated_devices(self):
        all_names = ['node1']
        all_device_counts = {'node1': {'cpu': 1}}
        all_device_maps = {'node1': {}}
        all_devices = {'node1': [torch.device('cpu:0'), torch.device('cpu:0')]}
        with self.assertRaises(ValueError) as context:
            _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices)
        self.assertIn("duplicated devices", str(context.exception))

    def test_tensorpipe_validate_devices_valid(self):
        devices = [torch.device('cpu'), torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')]
        device_count = {'cuda': 1} if torch.cuda.is_available() else {'cpu': 1}
        result = _tensorpipe_validate_devices(devices, device_count)
        self.assertTrue(result)


if __name__ == "__main__":
    run_tests()
