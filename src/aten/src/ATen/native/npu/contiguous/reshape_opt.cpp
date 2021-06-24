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

#include <ATen/native/npu/contiguous/ReshapeOpt.h>

namespace at {
namespace native {
namespace npu {

class ReshapeContiguousOpt : public ContiguousOpt {
 public:
  bool Optimizer(const Tensor& src, Tensor& self) override {
    if (check_reshape_match(src, self)) {
      RECORD_FUNCTION("View_d2dCopyAsync", std::vector<c10::IValue>({src}));
      copy_d2d_by_memcpy(self, src, prod_intlist(self.storage().get_npu_desc().storage_sizes_));      
      return true;
    }
    return false;
  }

  bool CanOptimizer(const Tensor& src) override {
    return check_reshape_match(src);
  }
}; // class ReshapeContiguousOpt

REGISTER_COPY_OPT(reshape, ReshapeContiguousOpt)

} // namespace npu
} // namespace native
} // namespace at