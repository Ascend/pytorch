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

#include <string>
#include <memory>
#include <tuple>

#include "torch_npu/csrc/framework/interface/AclTdtInterface.h"
namespace c10_npu {
class TdtDataSet {
public:
  TdtDataSet() {
    dataset_ = std::shared_ptr<acltdtDataset>(acl_tdt::AcltdtCreateDataset(),
                                              [](acltdtDataset* item) {
                                                acl_tdt::AcltdtDestroyDataset(item);
                                              });
  }
  std::shared_ptr<acltdtDataset> GetPtr() const {
    return dataset_;
  }
private:
  std::shared_ptr<acltdtDataset> dataset_ = nullptr;
};
} // namespace c10_npu

