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

#ifndef __TORCH_NPU_OPTIONSMANAGER_H__
#define __TORCH_NPU_OPTIONSMANAGER_H__

#include <ATen/npu/Exceptions.h>
#include <map>
#include <string>
#include <unordered_map>

namespace torch_npu {
namespace option {

class OptionsManager {
public:
  static bool CheckQueueEnable();
  static bool CheckCombinedOptimizerEnable();
  static bool CheckTriCombinedOptimizerEnable();
  static bool CheckAclDumpDateEnable();
  static bool CheckSwitchMMOutputEnable();
  static bool CheckDynamicOptimizer(const char* op);
  static bool CheckUseNpuLogEnable();
  static std::string CheckDisableDynamicPath();
private:
  static int GetBoolTypeOption(const char* env_str);
};

} // namespace option
} // namespace torch_npu

#endif // __TORCH_NPU_OPTIONSMANAGER_H__