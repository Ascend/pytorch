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

#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/aoe/AoeUtils.h"

namespace at_npu {
namespace native {
namespace aoe {

void AoeDumpGraphManager::SetDumpGraphPath(const std::string& dump_path) {
    autotune_graphdumppath = dump_path;
}

std::string AoeDumpGraphManager::GetDumpGraphPath() const {
    return autotune_graphdumppath;
}

aclGraphDumpOption* AoeDumpGraphManager::CreateGraphDumpOption() {
    AclGraphDumpOption = AclCreateGraphDumpOpt();
    return AclGraphDumpOption;
}

void AoeDumpGraphManager::DestropyGraphDumpOption() {
    AclDestroyGraphDumpOpt(AclGraphDumpOption);
    AclGraphDumpOption = NULL;
}

void AoeDumpGraphManager::EnableAoe() {
    aoe_enable = true;
}

bool AoeDumpGraphManager::IsAoeEnabled() const {
    return aoe_enable;
}

bool AoeDumpGraphManager::IsInWhitelist(const std::string &opName) const {
    if (white_list_.find(opName) != white_list_.end())
    {
        return true;
    }
    return false;
}

AoeDumpGraphManager& aoe_manager() {
    static AoeDumpGraphManager instance;
    return instance;
}

} // namespace aoe
} // namespace native
} // namespace at_npu