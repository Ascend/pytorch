#include "config/device/ascend/op_precision_conf.h"

namespace fxrt {
namespace config {
namespace ascend {
constexpr auto kAclMustKeepOriginDtype = "must_keep_origin_dtype";
constexpr auto kAclAllowFP32ToFP16 = "allow_fp32_to_fp16";

OpPrecisionConf& OpPrecisionConf::Instance() {
  static OpPrecisionConf instance;
  return instance;
}

bool OpPrecisionConf::IsAllowFP32ToFP16() {
  bool ret = socVersion_ < SocVersion::k910B1;
  if (!aclPrecisionMode_.empty()) {
    if (aclPrecisionMode_ == kAclMustKeepOriginDtype) {
      ret = false;
    } else if (aclPrecisionMode_ == kAclAllowFP32ToFP16) {
      ret = true;
    } else {
      RT_VLOG(VL_CONFIG) << "Unsupported precision mode: " << aclPrecisionMode_;
    }
  }
  return ret;
}

} // namespace ascend
} // namespace config
} // namespace fxrt
