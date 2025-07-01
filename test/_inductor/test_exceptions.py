from testutils import OperatorType, TestUtils
import torch
from torch._inductor.codecache import TritonCodeCache
from torch.testing._internal.common_utils import run_tests
import torch_npu


src_code_1 = '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[16384, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'y0_numel': 'i32', 'x1_numel': 'i32'},
    'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910B3', 
                                  major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32),
                                  'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_0', 'mutated_arg_names': [], 
                   'backend_hash': 'bc71dba4086164e7ac2b0779fa861dbf7467f0265d4a57b8f48cf6dda02b150f', 'split_axis': [0], 
                   'tiling_axis': [0, 1], 'axis_names': ['y0', 'x1'], 'low_dims': {1}, 'numof_reduction_axis': 0, 
                   'split_axis_dtype': torch.float16, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 
                   'traced_graph_dir': 'TRACED_GRAPH_DIR'},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_0(in_ptr0, in_ptr1, out_ptr0, y0_numel, x1_numel, Y0BLOCK: tl.constexpr, Y0BLOCK_SUB: tl.constexpr, X1BLOCK_SUB: tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < x1_numel
            tmp0 = tl.load(in_ptr0 + (x1 + 128*y0), x1_mask & y0_mask)
            # Not define tmp1 and make error manually for triton: 'tmp1 is not defined'
            tmp2 = tmp0 + tmp1
            tl.store(out_ptr0 + (x1 + 32*y0), tmp2, x1_mask & y0_mask)
'''


class TestExceptions(TestUtils):
    def test_triton_kernel_failed(self):
        with self.assertRaisesRegex(Exception, "tmp1 is not defined"):
            kernel = TritonCodeCache.load("triton_unk_fused_add_0", src_code_1)
            kernel.precompile()


if __name__ == "__main__":
    run_tests()
