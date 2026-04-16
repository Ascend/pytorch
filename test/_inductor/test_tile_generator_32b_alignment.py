import torch
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils

from torch_npu._inductor.codegen.tile_generator import TileGenerator, aligned_numel_32byte
from torch_npu._inductor.codegen.triton_utils import NPUKernelType


class TestTileGenerator32ByteAlignment(TestUtils):
    """
    Test class for TileGenerator 32Byte alignment functionality
    """

    def test_32byte_alignment(self):
        """
        Test that block_size and sub_block_size are 32Byte aligned
        """
        # Test with different data types
        dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int32]
        test_cases = [
            # (numels, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims)
            ((1024, 512), ["h", "w"], [0, 1], [], [0, 1], []),
            ((2048, 2048), ["h", "w"], [0, 1], [], [0, 1], []),
            ((100, 200), ["h", "w"], [0, 1], [], [0, 1], []),  # Non-power-of-2 sizes
        ]

        for dtype in dtypes:
            for numels, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims in test_cases:
                # Create TileGenerator instance
                tile_gen = TileGenerator(
                    numels=numels,
                    axis_names=axis_names,
                    tiling_axis=tiling_axis,
                    no_loop_axis=no_loop_axis,
                    split_axis=split_axis,
                    low_dims=low_dims,
                    persistent_reduction=False,
                    dtype=dtype,
                    npu_kernel_type=NPUKernelType.SIMD
                )

                # Generate configs
                configs = tile_gen.descend_split_tiling()

                # Check each config for 32Byte alignment
                for cfg in configs:
                    kwargs = cfg.kwargs
                    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
                    min_numel = 32 // dtype_bytes

                    # Check split axis block sizes
                    for axis in split_axis:
                        block_name = f"{axis_names[axis].upper()}BLOCK"
                        if block_name in kwargs:
                            block_size = kwargs[block_name]
                            if block_size > min_numel:
                                self.assertEqual(block_size % min_numel, 0, 
                                                 msg=f"Block size {block_size} for {block_name} "
                                                 f"with dtype {dtype} is not 32Byte aligned")

                    # Check tiling axis sub-block sizes
                    for axis in tiling_axis:
                        sub_block_name = f"{axis_names[axis].upper()}BLOCK_SUB"
                        if sub_block_name in kwargs:
                            sub_block_size = kwargs[sub_block_name]
                            if sub_block_size > min_numel:
                                self.assertEqual(sub_block_size % min_numel, 0, 
                                                 msg=f"Sub-block size {sub_block_size} for {sub_block_name} "
                                                 f"with dtype {dtype} is not 32Byte aligned")

    def test_aligned_numel_method(self):
        """
        Test the aligned_numel_32byte method directly
        """
        # Test with different data types
        test_cases = [
            (torch.float32, 32, 32),      # 32 elements = 128 bytes, already aligned
            (torch.float32, 33, 40),      # 33 elements = 132 bytes, should align to 40 elements (160 bytes)
            (torch.float16, 32, 32),      # 32 elements = 64 bytes, already aligned
            (torch.float16, 33, 48),      # 33 elements = 66 bytes, should align to 48 elements (96 bytes)
            (torch.bfloat16, 32, 32),     # 32 elements = 64 bytes, already aligned
            (torch.bfloat16, 33, 48),     # 33 elements = 66 bytes, should align to 48 elements (96 bytes)
            (torch.int32, 32, 32),        # 32 elements = 128 bytes, already aligned
            (torch.int32, 33, 40),        # 33 elements = 132 bytes, should align to 40 elements (160 bytes)
        ]

        for dtype, input_numel, expected_numel in test_cases:
            # Get dtype_bytes for the test
            dtype_bytes = torch.tensor([], dtype=dtype).element_size()
            result = aligned_numel_32byte(input_numel, dtype_bytes)
            self.assertEqual(result, expected_numel, 
                             msg=f"aligned_numel_32byte({input_numel}) for {dtype} "
                             f"returned {result}, expected {expected_numel}")
            
            
if __name__ == "__main__":
    run_tests()