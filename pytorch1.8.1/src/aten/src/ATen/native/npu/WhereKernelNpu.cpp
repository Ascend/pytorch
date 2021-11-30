// Copyright (c) 2020, Huawei Technologies.
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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

std::tuple<Tensor, Tensor, Tensor> npu_expand_outplace(
    const Tensor &to_expand1,
    const Tensor &to_expand2,
    const Tensor &to_expand3,
    const char *api_name) {
  for (auto& t : {to_expand1, to_expand2, to_expand3}) {
    if (!t.defined()) {
      AT_ERROR(api_name, "(...) called with an undefined Tensor");
    }
  }

  if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(to_expand1, to_expand2, to_expand3);
  }

  auto expanded_size12 = broadcast_ops_npu_output_size(to_expand1, to_expand2);
  auto expanded_size = broadcast_ops_npu_output_size(expanded_size12, to_expand3.sizes());

  return std::make_tuple(
      to_expand1.expand(expanded_size, true),
      to_expand2.expand(expanded_size, true),
      to_expand3.expand(expanded_size, true));
}

Tensor _s_where_npu(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  Tensor result = OpPreparation::ApplyTensor(self);

  OpCommand cmd;
  cmd.Name("Select")
    .Input(condition)
    .Input(self)
    .Input(other)
    .Output(result)
    .Run();

  return result;
}

Tensor where_self_npu(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
              "expected condition, x and y to be on the same device, but condition is on ",
              condition.device(), " and x and y are on ", self.device(), " and ", other.device(),
              " respectively");
  if (condition.scalar_type() != ScalarType::Byte && condition.scalar_type() != ScalarType::Bool) {
    AT_ERROR("Expected condition to have ScalarType Byte, but got ScalarType ",
        toString(condition.scalar_type()));
  }
  Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = npu_expand_outplace(condition, self, other, "where_npu");
  return at::_s_where(b_condition, b_self, b_other);
}

SmallVector<int64_t, SIZE> where_npu_output_size(const Tensor& condition){
  int64_t dim = condition.dim();
  Tensor boolSelf = condition.npu_dtype_cast(ScalarType::Bool);
  Tensor intSelf  = boolSelf.npu_dtype_cast(ScalarType::Int);
  Tensor coutNonzeroSelf = at::sum(intSelf, ScalarType::Int);
  int64_t nonzeroNum = coutNonzeroSelf.item().toInt();
  SmallVector<int64_t, SIZE> outputSize = {nonzeroNum, dim};
  return outputSize;
}


vector<Tensor> where_npu(const Tensor& condition) {
  Tensor formatCastOfCondition = condition;
  if (condition.storage().unsafeGetStorageImpl()->npu_desc_.npu_format_ !=
    ACL_FORMAT_ND) {
    formatCastOfCondition = formatCastOfCondition.npu_format_cast(ACL_FORMAT_ND);
  }
  if (condition.scalar_type() == ScalarType::Half) {
    formatCastOfCondition = formatCastOfCondition.npu_dtype_cast(ScalarType::Float);
  }

  // calculate the output size
  auto outputSize = where_npu_output_size(formatCastOfCondition);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, formatCastOfCondition.options().dtype(kLong), ACL_FORMAT_ND);

  OpCommand cmd;
  cmd.Name("NonZero")
    .Input(formatCastOfCondition)
    .Output(result)
    .Run();
  result = result.transpose(1, 0);
  std::vector<Tensor> chunkResult = result.chunk(result.size(0), 0);
  std::vector<Tensor> squeezeResult;
  for(int64_t i = 0; i < chunkResult.size(); i++){
    squeezeResult.push_back(chunkResult[i].squeeze(0));
  }

  return squeezeResult;
}

// pytorch1.8 add interface where.ScalarSelf,where.ScalarOther, where.Scalar
TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("where.self", TORCH_FN(where_self_npu));
  m.impl("where", TORCH_FN(where_npu));
  m.impl("_s_where", TORCH_FN(_s_where_npu));
}

} // namespace native
} // namespace at