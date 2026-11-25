#pragma once


#include <cstdint>
#include <string>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
namespace opapi {
typedef int32_t aclnnStatus;

/**
 * This API is used to check whether aclnnSilentCheck exist.
*/
C10_NPU_API bool IsExistAclnnSilentCheck();

/**
  This Api is used to reselect static kernel, it need to be called once at process.
 */
C10_NPU_API aclnnStatus ReselectStaticKernel();

/**
  This Api is used to reselect static kernel with a specified path.
 */
C10_NPU_API aclnnStatus ReselectStaticKernelWithPath(const std::string &path);

} // namespace opapi
} // namespace c10_npu
