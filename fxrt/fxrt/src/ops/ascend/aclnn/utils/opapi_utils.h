#ifndef __OPS_ASCEND_ACLNN_UTILS_OPAPI_UTILS_H__
#define __OPS_ASCEND_ACLNN_UTILS_OPAPI_UTILS_H__

#include <cstdint>
#include <string>
#include <unordered_map>

#include "common/visible.h"
#include "common/common.h"
#include "ir/tensor/tensor.h"

namespace fxrt {
namespace ops {
FXRT_EXPORT int8_t GetCubeMathType();

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_UTILS_OPAPI_UTILS_H__
