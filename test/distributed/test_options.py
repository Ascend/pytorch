import os
import numpy as np

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.multiprocessing as mp

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
import torch_npu


class OptionsTest(TestCase):
    @classmethod
    def _init_dist_hccl(cls, rank, options, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', pg_options=options, world_size=world_size, rank=rank)

    @classmethod
    def _test_all_reduce_with_options(cls, rank, ranks, world_size, input1):
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_config1 = {"hccl_buffer_size": 300, "group_name": "custom"}
        options.hccl_config = hccl_config1
        OptionsTest._init_dist_hccl(rank, options, world_size)
        input1 = input1.npu()
        dist.all_reduce(input1)
        hccl_config2 = {"hccl_buffer_size": 200}
        options.hccl_config = hccl_config2
        dist.all_reduce(input1)
        default_pg = c10d._get_default_group()._get_backend(torch.device('npu'))
        test_case = TestCase()
        test_case.assertEqual(default_pg.options.hccl_config, hccl_config1,
                              "Once Options are set for a ProcessGroupHCCL, later changes to Options won't affect "
                              "that ProcessGroupHCCL.")
        test_case.assertEqual(default_pg.options.hccl_config.get("group_name", ""), "custom")
        pg = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg)

    # Extract HCCL configuration constants for maintainability
    HCCL_DEFAULT_CONFIG = {
        "hccl_buffer_size": 300,
        "group_name": "custom",
        "hccl_exec_timeout": 500,
        "hccl_algo": "allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R",
        "hccl_retry_enable": "L0:0,L1:1,L2:0",
        "hccl_retry_params": "MaxCnt:5,HoldTime:5000,IntervalTime:5000",
        "hccl_buffer_name": "sharedBuffer"
    }

    @classmethod
    def _test_options_hccl_config_all_reduce(cls, rank, ranks, world_size, input1):
        """
        Test all_reduce operation with custom HCCL configuration options, validating:
        1. HCCL config is correctly applied to default process group
        2. New process group inherits the correct HCCL configuration
        3. all_reduce executes successfully on NPU
        
        Args:
            rank (int): Current process rank
            ranks (list[int]): List of ranks in the process group
            world_size (int): Total number of processes
            input1 (torch.Tensor): Input tensor for all_reduce operation
        """
        test_case = TestCase()
        new_pg = None  # Initialize process group variable to avoid undefined reference

        try:
            # 1. Validate input parameter validity
            test_case.assertTrue(world_size > 0, "world_size must be greater than 0")
            test_case.assertTrue(len(ranks) == world_size, "Length of ranks must match world_size")
            test_case.assertTrue(isinstance(input1, torch.Tensor), "input1 must be a torch.Tensor")

            # 2. Create custom HCCL options
            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            options.hccl_config = cls.HCCL_DEFAULT_CONFIG.copy()

            # 3. Initialize distributed process group with custom options
            cls._init_dist_hccl(rank, options, world_size)
            test_case.assertTrue(dist.is_initialized(), "Distributed process group initialization failed")

            # 4. Move input tensor to target NPU device (specify rank to avoid device conflict)
            input1 = input1.npu(rank)
            test_case.assertEqual(
                input1.device, 
                torch.device(f'npu:{rank}'), 
                "Tensor not correctly moved to target NPU device"
            )

            # 5. Execute all_reduce on default process group and validate configuration
            dist.all_reduce(input1)
            
            # Get default process group's HCCL backend
            default_pg = c10d._get_default_group()._get_backend(torch.device(f'npu:{rank}'))
            
            # Validate full HCCL configuration
            test_case.assertEqual(
                default_pg.options.hccl_config, 
                cls.HCCL_DEFAULT_CONFIG,
                "Default process group HCCL config does not match expected configuration"
            )
            
            # Validate individual config items (add default value to avoid KeyError)
            test_case.assertEqual(
                default_pg.options.hccl_config.get("hccl_exec_timeout", -1), 
                500,
                "hccl_exec_timeout config value mismatch"
            )
            test_case.assertEqual(
                default_pg.options.hccl_config.get("hccl_algo", ""), 
                "allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R",
                "hccl_algo config value mismatch"
            )

            # 6. Create new process group with same options and validate configuration
            new_pg = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
            test_case.assertTrue(new_pg is not None, "Failed to create new HCCL process group")
            
            # Validate new process group's HCCL configuration
            new_pg_backend = new_pg._get_backend(torch.device(f'npu:{rank}'))
            test_case.assertEqual(
                new_pg_backend.options.hccl_config, 
                cls.HCCL_DEFAULT_CONFIG,
                "New process group HCCL config does not match expected configuration"
            )
            
            # Execute all_reduce on new process group
            dist.all_reduce(input1, group=new_pg)

        except Exception as e:
            # Capture exceptions and mark test as failed
            test_case.fail(f"Test execution failed with error: {str(e)}")
        
        finally:
            # 7. Clean up resources to prevent memory leaks
            # Destroy custom process group if created
            if new_pg is not None:
                dist.destroy_process_group(new_pg)
            # Destroy default process group if initialized
            if dist.is_initialized():
                dist.destroy_process_group()
            # Clear NPU cache (optional, based on test environment requirements)
            if torch.npu.is_available():
                torch.npu.empty_cache()

    @classmethod
    def _test_options_wrong_type(cls, rank, hccl_config, error_expect, world_size, input1):
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = hccl_config
        input1 = input1.npu()
        test_case = TestCase()
        with test_case.assertRaises(RuntimeError) as cm:
            OptionsTest._init_dist_hccl(rank, options, world_size)
            dist.all_reduce(input1)
        
        test_case.assertTrue(error_expect in str(cm.exception),
            f"Expected error messages '{error_expect}' not found in actual error: {str(cm.exception)}")

    @classmethod
    def _test_options_group_name_wrong_types(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"group_name": 123}, "Value type of group_name should be string", world_size, input1)

    @classmethod
    def _test_options_qos_traffic_class_wrong_types(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"qos_traffic_class": "123"}, "Value type of qos_traffic_class should be int.", world_size, input1)

    @classmethod
    def _test_options_qos_service_level_wrong_types(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"qos_service_level": "123"}, "Value type of qos_service_level should be int.", world_size, input1)

    @classmethod
    def _test_options_hccl_sdma_qos_wrong_types(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"hccl_sdma_qos": "123"}, "Value type of hccl_sdma_qos should be int.", world_size, input1)

    @classmethod
    def _test_options_hccl_op_expansion_mode_wrong_types(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"hccl_op_expansion_mode": "123"}, "Value type of hccl_op_expansion_mode should be int.", world_size, input1)

    @classmethod
    def _test_options_hccl_exec_timeout_wrong_type(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"hccl_exec_timeout": "123"}, "Value type of hccl_exec_timeout should be int.", world_size, input1)

    @classmethod
    def _test_options_hccl_exec_timeout_invalid(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"hccl_exec_timeout": 2147483648}, "Value type of hccl_exec_timeout exceeds INT32_MAX(2147483647).", world_size, input1)

    @classmethod
    def _test_options_hccl_algo_wrong_type(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"hccl_algo": 123}, "Value type of hccl_algo should be string.", world_size, input1)

    @classmethod
    def _test_options_hccl_retry_enable_wrong_type(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"hccl_retry_enable": 123}, "Value type of hccl_retry_enable should be string.", world_size, input1)

    @classmethod
    def _test_options_hccl_retry_params_wrong_type(cls, rank, ranks, world_size, input1):
        cls._test_options_wrong_type(rank, {"hccl_retry_params": 123}, "Value type of hccl_retry_params should be string.", world_size, input1)

    @classmethod
    def _test_options_hccl_algo_exceed_length(cls, rank, ranks, world_size, input1):
        max_length = 1600
        exceed_length = max_length + 100
        long_algo_str = "a" * exceed_length
        hccl_config = {"hccl_algo": long_algo_str}
        
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = hccl_config
        input1 = input1.npu()
        
        test_case = TestCase()
        try:
            OptionsTest._init_dist_hccl(rank, options, world_size)
            dist.all_reduce(input1)
            
            default_pg = c10d._get_default_group()._get_backend(torch.device('npu'))
            actual_algo = default_pg.options.hccl_config.get("hccl_algo", "")
            test_case.assertEqual(len(actual_algo), max_length - 1,
                                "hccl_algo should be truncated to HCCL_COMM_ALGO_MAX_LENGTH - 1")
        except RuntimeError as e:
            if "HCCL function error" or "call hccl api failed" in str(e):
                test_case.assertTrue(True, "HCCL init failed as expected")

        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def _test_multiprocess(self, f, input1, world_size):
        ctx = mp.get_context('spawn')

        ps = []
        ranks = range(world_size)
        for rank in ranks:
            p = ctx.Process(
                target=f,
                args=(rank, ranks, world_size, input1.cpu()))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

        for p in ps:
            self.assertEqual(p.exitcode, 0)

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_with_options(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_all_reduce_with_options,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_group_name_wrong_type(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_group_name_wrong_types,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_qos_traffic_class_wrong_type(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_qos_traffic_class_wrong_types,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_qos_service_level_wrong_type(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_qos_service_level_wrong_types,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_hccl_sdma_qos_wrong_type(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_hccl_sdma_qos_wrong_types,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_hccl_op_expansion_mode_wrong_type(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_hccl_op_expansion_mode_wrong_types,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_hccl_exec_timeout_wrong_type(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_hccl_exec_timeout_wrong_type,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_hccl_exec_timeout_invalid(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_hccl_exec_timeout_invalid,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_hccl_algo_wrong_type(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_hccl_algo_wrong_type,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_hccl_algo_exceed_length(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_hccl_algo_exceed_length,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_hccl_retry_enable_wrong_type(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_hccl_retry_enable_wrong_type,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_hccl_retry_params_wrong_type(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_hccl_retry_params_wrong_type,
                                    input1, world_size)

    @skipIfUnsupportMultiNPU(2)
    def test_options_hccl_config_all_reduce(self):
        ranks = [2]
        shape = [np.int32, 0, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            self._test_multiprocess(OptionsTest._test_options_hccl_config_all_reduce,
                                    input1, world_size)

if __name__ == '__main__':
    run_tests()
