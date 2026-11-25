#ifndef __TORCH_NPU_LIBASCEND_HAL__
#define __TORCH_NPU_LIBASCEND_HAL__

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace at_npu {
namespace native {
C10_NPU_API int64_t getFreq();
C10_NPU_API bool isSyscntEnable();
} // namespace native
} // namespace torchat_npu_npu
#endif
