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

#include <string>

#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace c10_npu {
namespace option {

using namespace std;

bool OptionsManager::CheckQueueEnable() {
  static int32_t queue_enable = -1;
  if (queue_enable == -1) {
    queue_enable = GetBoolTypeOption("TASK_QUEUE_ENABLE");
  }
  return (queue_enable == 1);
}

bool OptionsManager::CheckCombinedOptimizerEnable() {
  static int32_t combined_optimize = -1;
  if (combined_optimize == -1) {
    combined_optimize = GetBoolTypeOption("COMBINED_ENABLE");
  }
  return (combined_optimize == 1);
}

bool OptionsManager::CheckAclDumpDateEnable() {
  static int aclDumpDataEnable = -1;
  if (aclDumpDataEnable == -1) {
    aclDumpDataEnable = GetBoolTypeOption("ACL_DUMP_DATA");
  }
  return (aclDumpDataEnable == 1);
}

bool OptionsManager::CheckSwitchMMOutputEnable() {
  static int switchMMOutputEnable = -1;
  if (switchMMOutputEnable == -1) {
    switchMMOutputEnable = GetBoolTypeOption("SWITCH_MM_OUTPUT_ENABLE");
  }
  return (switchMMOutputEnable == 1);
}

int OptionsManager::GetBoolTypeOption(const char* env_str) {
  char* env_val = std::getenv(env_str);
  int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 0;
  return (envFlag != 0) ? 1 : 0;
}

bool OptionsManager::CheckUseNpuLogEnable() {
  static int useNpuLog = -1;
  if (useNpuLog == -1) {
    useNpuLog = GetBoolTypeOption("NPU_LOG_ENABLE");
  }

  return (useNpuLog == 1);
}
} // namespace option
} // namespace c10_npu