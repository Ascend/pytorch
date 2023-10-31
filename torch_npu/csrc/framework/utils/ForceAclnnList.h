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

#ifndef TORCHNPU_TORCH_NPU_CSRC_FRAMEWORK_UTILS_FORCEACLNNLIST_H_
#define TORCHNPU_TORCH_NPU_CSRC_FRAMEWORK_UTILS_FORCEACLNNLIST_H_

#include <set>
#include <string>

namespace at_npu {
namespace native {

class ForceAclnn {
public:
  static ForceAclnn& GetInstance() {
    static ForceAclnn instance;
    return instance;
  }
  void RegisterOp(const std::string& list);
  bool IsForceAclnnOp(const std::string &op_name) const;
  ~ForceAclnn() = default;
private:
  ForceAclnn() = default;
  std::set<std::string> force_aclnn_op_list_;
};

} // namespace native
} // namespace at_npu
#endif TORCHNPU_TORCH_NPU_CSRC_FRAMEWORK_UTILS_FORCEACLNNLIST_H_