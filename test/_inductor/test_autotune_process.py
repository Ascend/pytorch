import os
from unittest import mock

import torch
from torch._inductor import (
    autotune_process as inductor_autotune_process,
    config as inductor_config,
    utils as inductor_utils,
)
from torch._inductor.autotune_process import TuningProcess, TuningProcessPool
from torch.testing._internal.common_utils import TestCase, run_tests


COMMUNITY_GET_DEVICE_LIST = TuningProcessPool.get_device_list

from torch_npu._inductor import utils as npu_utils
from torch_npu._inductor.autotune_process import (
    ASCEND_VISIBLE_DEVICES,
    patch_tuning_process,
)


class TestAutotuneProcessAdapter(TestCase):
    def setUp(self):
        super().setUp()
        self.visible_devices_key = inductor_autotune_process.CUDA_VISIBLE_DEVICES
        self.pool = object.__new__(TuningProcessPool)

    def tearDown(self):
        inductor_autotune_process.CUDA_VISIBLE_DEVICES = self.visible_devices_key
        super().tearDown()

    def test_pool_get_device_list_is_community_method(self):
        self.assertEqual(
            COMMUNITY_GET_DEVICE_LIST.__module__,
            "torch._inductor.autotune_process",
        )
        self.assertIs(TuningProcessPool.get_device_list, COMMUNITY_GET_DEVICE_LIST)

    def test_single_device_mode(self):
        with inductor_config.patch("autotune_multi_device", False):
            self.assertEqual(self.pool.get_device_list(), [None])

    def test_multi_device_uses_npu_interface_and_visible_key(self):
        interface = mock.Mock()
        interface.device_count.return_value = 4
        patch_tuning_process()
        with (
            inductor_config.patch("autotune_multi_device", True),
            mock.patch.object(
                inductor_autotune_process, "get_gpu_type", return_value="npu"
            ),
            mock.patch.object(
                inductor_autotune_process,
                "get_interface_for_device",
                return_value=interface,
            ),
            mock.patch.dict(
                os.environ, {ASCEND_VISIBLE_DEVICES: "3,1"}, clear=True
            ),
        ):
            self.assertEqual(self.pool.get_device_list(), [3, 1])

        with (
            inductor_config.patch("autotune_multi_device", True),
            mock.patch.object(
                inductor_autotune_process, "get_gpu_type", return_value="npu"
            ),
            mock.patch.object(
                inductor_autotune_process,
                "get_interface_for_device",
                return_value=interface,
            ),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(self.pool.get_device_list(), [0, 1, 2, 3])

    def test_tuning_process_scopes_visible_device_to_child(self):
        patch_tuning_process()
        with (
            mock.patch.dict(
                os.environ, {ASCEND_VISIBLE_DEVICES: "2,3"}, clear=True
            ),
            mock.patch.object(
                inductor_autotune_process.subprocess, "Popen"
            ) as popen_mock,
        ):
            process = TuningProcess(1)
            self.addCleanup(process.write_pipe.close)
            self.addCleanup(process.read_pipe.close)
            self.addCleanup(process.selector.close)

            child_env = popen_mock.call_args.kwargs["env"]
            self.assertEqual(child_env[ASCEND_VISIBLE_DEVICES], "1")
            self.assertEqual(os.environ[ASCEND_VISIBLE_DEVICES], "2,3")


class TestPatchIsGpu(TestCase):
    def setUp(self):
        super().setUp()
        self.gpu_types = list(inductor_utils.GPU_TYPES)
        inductor_utils.get_gpu_type.cache_clear()

    def tearDown(self):
        inductor_utils.GPU_TYPES[:] = self.gpu_types
        inductor_utils.get_gpu_type.cache_clear()
        super().tearDown()

    def test_patch_is_gpu_is_idempotent_and_clears_cached_device(self):
        inductor_utils.GPU_TYPES[:] = ["cuda"]
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(inductor_utils.get_gpu_type(), "cuda")

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.npu, "is_available", return_value=True),
        ):
            npu_utils.patch_is_gpu()
            self.assertEqual(inductor_utils.get_gpu_type(), "npu")
            self.assertEqual(inductor_utils.get_gpu_type.cache_info().currsize, 1)
            npu_utils.patch_is_gpu()
            self.assertEqual(inductor_utils.get_gpu_type.cache_info().currsize, 0)

        self.assertEqual(inductor_utils.GPU_TYPES.count("npu"), 1)


if __name__ == "__main__":
    run_tests()
