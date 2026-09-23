#ifndef __OPS_UTILS_OP_SUPPORT_H__
#define __OPS_UTILS_OP_SUPPORT_H__

#include <cstdint>
#include <string>
#include <vector>

#include "hardware/device.h"
#include "common/visible.h"
#include "ir/common/dtype.h"
#include "ir/value/value.h"

namespace fxrt {
namespace runtime {

enum class OpSupportStatus : int32_t {
  kOk = 0,
  kUnsupportedDevice = 1,
  kUnsupportedInputType = 2,
};

struct OpSupportResult {
  OpSupportStatus status{OpSupportStatus::kOk};
  std::string message;
};

FXRT_EXPORT hardware::Device GetDeviceFromOutputAndInputs(
    const ir::ValuePtr& output,
    const std::vector<ir::ValuePtr>& inputs);

FXRT_EXPORT OpSupportResult CheckOpSupport(
    const std::string& op_name,
    const ir::ValuePtr& output_value,
    const std::vector<ir::ValuePtr>& input_values);

} // namespace runtime
} // namespace fxrt

#endif // __OPS_UTILS_OP_SUPPORT_H__
