#ifndef __PLUGIN_NATIVE_UTILS_NPU_CONFIG__
#define __PLUGIN_NATIVE_UTILS_NPU_CONFIG__


#include <c10/util/SmallVector.h>
#include <third_party/acl/inc/graph/operator.h>

#include <functional>
#include <vector>

namespace at_npu {
namespace native {

// in npu device, the max shape size is 8
constexpr int MAX_FORMAT_SHAPE_SIZE = 8;
using FormatShape = c10::SmallVector<int64_t, MAX_FORMAT_SHAPE_SIZE>;

using DyNumAndIndex = std::vector<std::pair<uint32_t, uint32_t>>;
using DynamicInputRegFunc = std::function<ge::OperatorPtr(DyNumAndIndex, std::string)>;

} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_UTILS_NPU_CONFIG__