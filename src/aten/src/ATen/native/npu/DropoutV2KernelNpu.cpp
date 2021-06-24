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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor&, Tensor&, Tensor&> dropout_v2_out_npu(
    Tensor& result,
    Tensor& mask,
    Tensor& new_seed,
    const Tensor& self,
    Tensor& seed,
    double p) {

  OpCommand cmd;
  cmd.Name("DropoutV2")
      .Input(self)
      .Input(seed)
      .Output(result)
      .Output(mask)
      .Output(new_seed)
      .Attr("p", static_cast<float>(p))
      .Run();
  
  return tuple<Tensor&, Tensor&, Tensor&>(result, mask, new_seed);
}

tuple <Tensor, Tensor, Tensor> dropout_v2_npu(const Tensor& self, Tensor& seed, double p) {
  Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  Tensor formatCastOfSeed = OpPreparation::CastBackToOriFormat(seed);
  
  Tensor result = at::empty_with_format(
      formatCastOfSelf.sizes(), formatCastOfSelf.options(), CalcuOpUtil::get_tensor_npu_format(formatCastOfSelf));
  Tensor mask = at::empty_with_format(
      formatCastOfSelf.sizes(), formatCastOfSeed.options(), CalcuOpUtil::get_tensor_npu_format(formatCastOfSelf));

  dropout_v2_out_npu(result, mask, formatCastOfSeed, formatCastOfSelf, formatCastOfSeed, p);
  NpuUtils::format_fresh_view(seed, formatCastOfSeed);
  return std::tuple<Tensor, Tensor, Tensor>(result, mask, seed);
}


} // namespace native
} // namespace at

