#ifndef __CONFIG_DEVICE_ASCEND_OP_PRECISION_CONF_H__
#define __CONFIG_DEVICE_ASCEND_OP_PRECISION_CONF_H__

#include <string>

#include "common/common.h"
#include "common/visible.h"
#include "config/device/ascend/common.h"

namespace fxrt {
namespace config {
namespace ascend {
class FXRT_EXPORT OpPrecisionConf {
 public:
  static OpPrecisionConf& Instance();

  void SetAclPrecisionMode(const std::string& aclPrecisionMode) {
    aclPrecisionMode_ = aclPrecisionMode;
  }

  const std::string& AclPrecisionMode() const {
    return aclPrecisionMode_;
  }

  void SetIsAllowMatmulHF32(bool isAllowMatmulHF32) {
    isAllowMatmulHF32_ = isAllowMatmulHF32;
  }

  bool IsAllowMatmulHF32() const {
    return isAllowMatmulHF32_;
  }

  void SetSocVersion(const SocVersion& socVersion) {
    socVersion_ = socVersion;
  }

  bool IsAllowFP32ToFP16();

 private:
  OpPrecisionConf() = default;
  DISABLE_COPY_AND_ASSIGN(OpPrecisionConf);

  std::string aclPrecisionMode_;
  bool isAllowMatmulHF32_;
  SocVersion socVersion_{SocVersion::UnsupportedSocVersion};
};

} // namespace ascend
} // namespace config
} // namespace fxrt
#endif // __CONFIG_DEVICE_ASCEND_OP_PRECISION_CONF_H__
