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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor linalg_cross_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  return is_self_wrapped ? other : self;
}

at::Tensor& linalg_cross_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<int64_t> dim,
    at::Tensor& result) {
  int64_t real_dim = dim.has_value() ? dim.value() : -65530;
  OpCommand cmd;
  cmd.Name("Cross")
      .Input(self)
      .Input(other)
      .Output(result)
      .Attr("dim", real_dim)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::linalg_cross_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const int64_t dim,
    at::Tensor& result){
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::Tensor output_tensor = linalg_cross_dest_output(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(output_tensor),
      self.scalar_type(),
      output_size);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    linalg_cross_out_npu_nocheck(self, other, dim, contiguous_result);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    linalg_cross_out_npu_nocheck(self, other, dim, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::linalg_cross(
    const at::Tensor& self,
    const at::Tensor& other,
    const int64_t dim) {
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::Tensor output_tensor = linalg_cross_dest_output(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(output_size, self.options(), output_tensor);
  linalg_cross_out_npu_nocheck(self, other, dim, result);
  return result;
}

} // namespace native
} // namespace at_npu
