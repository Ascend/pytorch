#pragma once

#include <cstdint>
#include <string>

namespace c10_npu {
namespace opapi {
typedef int32_t aclnnStatus;

/**
 * This API is used to check whether aclnnSilentCheck exist.
*/
bool IsExistAclnnSilentCheck();

/**
  This Api is used to reselect static kernel, it need to be called once at process.
 */
aclnnStatus ReselectStaticKernel();

/**
  This Api is used to reselect static kernel with a specified path.
 */
aclnnStatus ReselectStaticKernelWithPath(const std::string &path);

} // namespace opapi
} // namespace c10_npu
