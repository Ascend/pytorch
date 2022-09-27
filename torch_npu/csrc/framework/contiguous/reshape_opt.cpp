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

#include "torch_npu/csrc/framework/contiguous/ReshapeOpt.h"

namespace at_npu {
namespace native {

class ReshapeContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(at::Tensor &result, const at::Tensor &src,
                 const ContiguousTensorDesc &src_desc) override {
    ContiguousTensorDesc result_desc = TransContiguous::GetTensorDescInfo(result);
    if (check_reshape_match(result_desc, src_desc)) {
      RECORD_FUNCTION("View_d2dCopyAsync", std::vector<c10::IValue>({src}));
      NPUNativeFunctions::npu_reshape_out(src, src.sizes(), false, result);
      return true;
    }
    return false;
  }

  bool CanOptimizer(const ContiguousTensorDesc &src_desc) override {
    return check_reshape_match(src_desc);
  }
}; // class ReshapeContiguousOpt

REGISTER_COPY_OPT(reshape, ReshapeContiguousOpt)

} // namespace native
} // namespace at_npu