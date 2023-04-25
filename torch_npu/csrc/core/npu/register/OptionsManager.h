// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
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

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <c10/util/Exception.h>

namespace c10_npu {
namespace option {

class OptionsManager {
public:
  static bool CheckQueueEnable();
  static bool CheckCombinedOptimizerEnable();
  static bool CheckTriCombinedOptimizerEnable();
  static bool CheckAclDumpDateEnable();
  static bool CheckDisableAclopComAndExe();
  static bool CheckSwitchMMOutputEnable();
  static uint32_t GetHCCLExecTimeout();
  static std::string CheckDisableDynamicPath();
private:
  static int GetBoolTypeOption(const char* env_str, int defaultVal = 0);
};

} // namespace option
} // namespace c10_npu