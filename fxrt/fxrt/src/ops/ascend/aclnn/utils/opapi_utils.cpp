#include "common/logger.h"
#include "config/device/ascend/op_precision_conf.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"

namespace fxrt {
namespace ops {
namespace {
using config::ascend::AclCubeMathType;
using config::ascend::OpPrecisionConf;
constexpr auto AclCubeMathTypeArraySize = 4;

constexpr AclCubeMathType AclCubeMathTypeArray[AclCubeMathTypeArraySize] = {
    AclCubeMathType::KEEP_DTYPE,
    AclCubeMathType::USE_FP16,
    AclCubeMathType::USE_HF32,
    AclCubeMathType::ALLOW_FP32_DOWN_PRECISION,
};
} // namespace

int8_t GetCubeMathType() {
  auto& opPrecisionConf = OpPrecisionConf::Instance();
  uint8_t cubeMathTypeIndex = (static_cast<uint8_t>(opPrecisionConf.IsAllowMatmulHF32()) << 1) +
      static_cast<uint8_t>(opPrecisionConf.IsAllowFP32ToFP16());
  if (cubeMathTypeIndex >= AclCubeMathTypeArraySize) {
    RT_VLOG(VL_OPS) << "Invalid cubeMathType index: " << cubeMathTypeIndex
                    << ", set AclCubeMathType to ALLOW_FP32_DOWN_PRECISION";
    return static_cast<int8_t>(AclCubeMathType::ALLOW_FP32_DOWN_PRECISION);
  }
  return static_cast<int8_t>(AclCubeMathTypeArray[cubeMathTypeIndex]);
}

} // namespace ops
} // namespace fxrt
