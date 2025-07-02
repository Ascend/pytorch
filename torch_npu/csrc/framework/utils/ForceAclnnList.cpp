// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at_npu
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_npu/csrc/framework/utils/ForceAclnnList.h"
#include <iostream>

namespace at_npu {
namespace native {
void ForceAclnn::RegisterOp(const std::string &list)
{
    if (list.empty()) {
        return;
    }

    auto value = list;
    std::string delimiter = ",";
    auto start = 0U;
    auto end = value.find(delimiter);
    std::string token;
    while (end != std::string::npos) {
        token = value.substr(start, end - start);
        if (!token.empty()) {
            force_aclnn_op_list_.insert(token);
        }
        start = end + delimiter.size();
        end = value.find(delimiter, start);
    }
    token = value.substr(start, end - start);
    if (!token.empty()) {
        force_aclnn_op_list_.insert(token);
    }
    return;
}

bool ForceAclnn::IsForceAclnnOp(const std::string &op_name) const
{
    bool ret = (force_aclnn_op_list_.find(op_name) != force_aclnn_op_list_.end());
    return ret;
}
} // namespace native
} // namespace at_npu