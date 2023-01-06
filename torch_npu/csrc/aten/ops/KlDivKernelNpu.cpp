// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::kl_div(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target) {
  at::Tensor result =
      reduction == at::Reduction::None ?
      OpPreparation::ApplyTensor(self) :
      OpPreparation::ApplyTensor({}, self.options(), self);
  string reductionStr;
  if (reduction == at::Reduction::Mean) {
    reductionStr = "batchmean";
  } else if (reduction == at::Reduction::Sum) {
    reductionStr = "sum";
  } else if (reduction == at::Reduction::None) {
    reductionStr = "none";
  }
  OpCommand cmd;
  cmd.Name("KLDiv")
      .Input(self)
      .Input(target)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Attr("log_target", log_target)
      .Run();
  if (reduction == at::Reduction::Mean) {
    auto inputShape = self.sizes();
    int batchSquareSize = c10::multiply_integers(inputShape) / inputShape[0];
    result.div_(batchSquareSize);
  }
  return result;
}
} // namespace native
} // namespace at_npu
