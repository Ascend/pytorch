// Copyright (c) 2022 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_npu/csrc/core/npu/NpuVariables.h"

#include <c10/util/Exception.h>

#include <iostream>
#include <map>
#include <string>

#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
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
    {"Ascend910B1", SocVersion::Ascend910B1},
    {"Ascend910B2", SocVersion::Ascend910B2},
    {"Ascend910B2C", SocVersion::Ascend910B2C},
    {"Ascend910B3", SocVersion::Ascend910B3},
    {"Ascend910B4", SocVersion::Ascend910B4},
    {"Ascend310B1", SocVersion::Ascend310B1},
    {"Ascend310B2", SocVersion::Ascend310B2},
    {"Ascend310B3", SocVersion::Ascend310B3},
    {"Ascend310B4", SocVersion::Ascend310B4},
    {"Ascend910C1", SocVersion::Ascend910C1},
    {"Ascend910C2", SocVersion::Ascend910C2},
    {"Ascend910C3", SocVersion::Ascend910C3},
    {"Ascend910C4", SocVersion::Ascend910C4},
    {"Ascend910D1", SocVersion::Ascend910D1}};

bool SetSocVersion(const char* const socVersion) {
  if (socVersion == nullptr ||
      g_curSocVersion != SocVersion::UnsupportedSocVersion) {
    return true;
  }

  SocVersion curSocVersion = SocVersion::UnsupportedSocVersion;

  auto const& iter = socVersionMap.find(socVersion);
  if (iter != socVersionMap.end()) {
    curSocVersion = iter->second;
  } else {
    return false;
  }

  g_curSocVersion = curSocVersion;
  return true;
}

const SocVersion& GetSocVersion() { return g_curSocVersion; }

bool IsSupportInfNan() {
    if (!c10_npu::option::OptionsManager::CheckInfNanModeEnable()) {
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
    return ((GetSocVersion() >= SocVersion::Ascend910B1) && (GetSocVersion() < SocVersion::Ascend310B1)) ||
        (GetSocVersion() >= SocVersion::Ascend910C1);
}

bool IsBF16Supported()
{
    return GetSocVersion() >= SocVersion::Ascend910B1;
}
}  // namespace c10_npu
