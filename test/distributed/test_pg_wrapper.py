# Owner(s): ["oncall: distributed"]

import os
import sys
from datetime import timedelta

import torch
import torch_npu
import torch.distributed as c10d
from torch._C._distributed_c10d import _ProcessGroupWrapper
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    with_dist_debug_levels,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = c10d.get_default_backend_for_device(device_type)


class AbstractProcessGroupWrapperTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _validate_error(self, exception, op_type, rank, tensor, verify_diff=True):
        err = str(exception)
        self.assertTrue(
            op_type in err, f"Got {err} but expected {op_type} to be in error."
        )
        # User doesn't call barrier with tensor.
        if op_type != "BARRIER":
            self.assertTrue(
                f"{list(tensor.shape)}" in err,
                f"Did not find shapes {list(tensor.shape)} in error {err}",
            )
            # For NPU, only assert on device type, not index
            if device_type in str(tensor.device):
                self.assertTrue(
                    device_type in err,
                    f"Did not find {device_type} device in error {err}",
                )
            else:
                self.assertTrue(
                    str(tensor.device) in err,
                    f"Did not find tensor device {str(tensor.device)} in error {err}",
                )
            # C++ and python type strings are not exactly the same.
            if "float" in str(tensor.dtype):
                self.assertTrue("Float" in err, "Expected Float type")
            elif "int" in str(tensor.dtype):
                self.assertTrue("Long" in err, "Expected Long type")
            else:
                self.fail(f"Unexpected dtype {str(tensor.dtype)} for error {err}")

            # Ensure sequence number is logged in error
            self.assertTrue("SequenceNumber" in err)
            # Ensure info about how collectives diff is in the error.
            if verify_diff:
                self.assertTrue(
                    "Collectives differ in the following" in err, f"Got error {err}"
                )


# ASAN is not safe since we are spawning processes.
if not TEST_WITH_DEV_DBG_ASAN:
    class ProcessGroupHCCLWrapperTest(AbstractProcessGroupWrapperTest):
        def setUp(self):
            super(AbstractProcessGroupWrapperTest, self).setUp()
            self._spawn_processes()
            # TORCH_HCCL_BLOCKING_WAIT overrides TORCH_HCCL_ASYNC_ERROR_HANDLING hence tests
            # that use TORCH_HCCL_BLOCKING_WAIT will test it as expected.
            os.environ["TORCH_HCCL_ASYNC_ERROR_HANDLING"] = "1"

        @property
        def world_size(self) -> int:
            return 2

        def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size,
                store=store,
                timeout=timedelta(seconds=timeout),
            )
            if with_new_group:
                pg = c10d.new_group(backend=backend, timeout=timedelta(seconds=timeout))
            else:
                if device_type == "xpu":
                    _pg = c10d.ProcessGroupXCCL(
                        store,
                        self.rank,
                        self.world_size,
                    )
                else:
                    _pg = torch_npu._C._distributed_c10d.ProcessGroupHCCL(
                        store,
                        self.rank,
                        self.world_size,
                        timeout=timedelta(seconds=timeout),
                    )
                pg = c10d._create_process_group_wrapper(
                    _pg,
                    "unused",
                    store,
                    self.rank,
                    self.world_size,
                    timeout=timeout,
                )
            return pg

        @skipIfUnsupportMultiNPU(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_wrapper_forwards_hccl_methods(self):
            """
            Tests that ProcessGroupWrapper correctly forwards HCCL-specific
            utility methods to the wrapped backend. See issue #173538.
            """
            torch.npu.set_device(self.rank)
            device = torch.device(f"npu:{self.rank}")
            wrapper = self._create_wrapper_pg(with_new_group=False)

            # Verify we're testing the wrapper
            self.assertIsInstance(wrapper, _ProcessGroupWrapper)
            unwrapped = wrapper.wrapped_pg

            # Verify wrapper forwards property/method calls to wrapped backend
            self.assertEqual(wrapper.supports_splitting, unwrapped.supports_splitting)
            self.assertEqual(wrapper.supports_coalescing, unwrapped.supports_coalescing)
            self.assertEqual(
                wrapper.supports_time_estimate, unwrapped.supports_time_estimate
            )
            self.assertEqual(
                wrapper.supports_tensor_alloc(device),
                unwrapped.supports_tensor_alloc(device),
            )
            if hasattr(unwrapped, 'get_error'):
                self.assertEqual(wrapper.get_error(), unwrapped.get_error())
            try:
                wrapper_options = wrapper.options
            except RuntimeError as e:
                self.assertIn("does not implement getBackendOptions", str(e))
                wrapper_options = None

            if wrapper_options is not None:
                self.assertIs(wrapper_options, unwrapped.options)

            # Test eager_connect_single_device forwarding (should not raise)
            wrapper.eager_connect_single_device(device)

            # HCCL does not support mem_allocator and allocate_tensor
            if wrapper.supports_tensor_alloc(device):
                self.assertIs(wrapper.mem_allocator, unwrapped.mem_allocator)
                tensor = wrapper.allocate_tensor(
                    1024, dtype=torch.float32, device=device
                )
                self.assertEqual(tensor.shape, torch.Size([1024]))


if __name__ == "__main__":
    assert not (torch.npu.is_initialized() or torch.xpu.is_initialized()), (
        "test_pg_wrapper must not have initialized NPU/XPU context on main process"
    )

    run_tests()
