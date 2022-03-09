// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#ifndef __NATIVE_NPU_TOOLS_AOEUTILS__
#define __NATIVE_NPU_TOOLS_AOEUTILS__

#include <unordered_set>
#include <third_party/acl/inc/acl/acl_op_compiler.h>
#include <c10/npu/NPUException.h>

namespace at_npu {
namespace native {
namespace aoe {

class AoeDumpGraphManager  {
public:
  void SetDumpGraphPath(const std::string& dump_path);
  std::string GetDumpGraphPath() const;

  aclGraphDumpOption* CreateGraphDumpOption();
  void DestropyGraphDumpOption();

  void EnableAoe();
  bool IsAoeEnabled() const;
  bool IsInBlacklist(const std::string &opName) const;
  
  bool aoe_enable=false;
  // to save graph for autotune, default path is ./
  std::string autotune_graphdumppath="./";
  aclGraphDumpOption* AclGraphDumpOption=NULL;
  std::unordered_set<std::string> black_list_ = {"TransData"};
   
};

AoeDumpGraphManager& aoe_manager();

} // namespace aoe
} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_TOOLS_AOEUTILS__