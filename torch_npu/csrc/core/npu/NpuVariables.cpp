#include <iostream>
#include <map>
#include <string>
#include <algorithm>

#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace c10_npu {
static SocVersion g_curSocVersion = SocVersion::UnsupportedSocVersion;

static std::map<std::string, SocVersion> socVersionMap = {
    {"Ascend910PremiumA", SocVersion::Ascend910PremiumA},
    {"Ascend910ProA", SocVersion::Ascend910ProA},
    {"Ascend910A", SocVersion::Ascend910A},
    {"Ascend910ProB", SocVersion::Ascend910ProB},
    {"Ascend910B", SocVersion::Ascend910B},
    {"Ascend310P1", SocVersion::Ascend310P1},
    {"Ascend310P2", SocVersion::Ascend310P2},
    {"Ascend310P3", SocVersion::Ascend310P3},
    {"Ascend310P4", SocVersion::Ascend310P4},
    {"Ascend310P5", SocVersion::Ascend310P5},
    {"Ascend310P7", SocVersion::Ascend310P7},
    {"Ascend910B1", SocVersion::Ascend910B1},
    {"Ascend910B2", SocVersion::Ascend910B2},
    {"Ascend910B2C", SocVersion::Ascend910B2C},
    {"Ascend910B3", SocVersion::Ascend910B3},
    {"Ascend910B4", SocVersion::Ascend910B4},
    {"Ascend910B4-1", SocVersion::Ascend910B4_1},
    {"Ascend310B1", SocVersion::Ascend310B1},
    {"Ascend310B2", SocVersion::Ascend310B2},
    {"Ascend310B3", SocVersion::Ascend310B3},
    {"Ascend310B4", SocVersion::Ascend310B4},
    {"Ascend910_9391", SocVersion::Ascend910_9391},
    {"Ascend910_9392", SocVersion::Ascend910_9392},
    {"Ascend910_9381", SocVersion::Ascend910_9381},
    {"Ascend910_9382", SocVersion::Ascend910_9382},
    {"Ascend910_9372", SocVersion::Ascend910_9372},
    {"Ascend910_9362", SocVersion::Ascend910_9362}};

void SetSocVersion(const char* const socVersion)
{
  if (socVersion == nullptr ||
      g_curSocVersion != SocVersion::UnsupportedSocVersion) {
    return;
  }

  SocVersion curSocVersion = SocVersion::UnsupportedSocVersion;

  auto const& iter = socVersionMap.find(socVersion);
  if (iter != socVersionMap.end()) {
    curSocVersion = iter->second;
  } else {
    std::string unsupported_soc(socVersion);
    std::replace(std::begin(unsupported_soc), std::end(unsupported_soc), '_', ' ');
    AT_ERROR("Unsupported soc version: ", unsupported_soc);
  }

  g_curSocVersion = curSocVersion;
}

const SocVersion& GetSocVersion()
{
    return g_curSocVersion;
}

bool IsSupportInfNan()
{
    static bool default_support_inf_nan = ((GetSocVersion() >= SocVersion::Ascend910B1) &&
        (GetSocVersion() < SocVersion::Ascend310B1)) ||
        (GetSocVersion() >= SocVersion::Ascend910_9391);
    if (!c10_npu::option::OptionsManager::CheckInfNanModeEnable()) {
        if (default_support_inf_nan && !c10_npu::option::OptionsManager::CheckInfNanModeForceDisable()) {
            AT_ERROR("INF_NAN_MODE_ENABLE shouldn't be set to 0 on the current device. If you want to disable ",
                "inf-nan mode, please export INF_NAN_MODE_FORCE_DISABLE=1");
        }
        return false;
    }
    if (c10_npu::option::OptionsManager::CheckInfNanModeForceDisable()) {
        return false;
    }
    if (c10_npu::acl::IsExistGetCannAttribute()) {
        const static bool supportInfNan = []() -> bool {
            int enable = 0;
            NPU_CHECK_ERROR(c10_npu::acl::AclGetCannAttribute(ACL_CANN_ATTR_INF_NAN, &enable));
            return enable != 0;
        }();
        return supportInfNan;
    }
    return default_support_inf_nan;
}

bool IsBF16Supported()
{
    return GetSocVersion() >= SocVersion::Ascend910B1;
}
}  // namespace c10_npu

