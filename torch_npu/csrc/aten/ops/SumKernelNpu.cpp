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

#include <ATen/WrapDimUtilsMulti.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu{
namespace native{

at::Tensor &sum_out_npu_nocheck(
    at::Tensor &result,
    const at::Tensor &self,
    at::IntArrayRef dim,
    bool keepdim) {
  at::dim_list_to_bitset(dim, self.dim());
  c10::SmallVector<int64_t, N> dim_list = dim.empty() ? CalcuOpUtil::GetDimlistForTensor(self) :
      c10::SmallVector<int64_t, N>(dim);
  OpCommand cmd;
  cmd.Name("ReduceSum")
      .Input(self)
      .Input(dim_list, at::kLong)
      .Output(result)
      .Attr("keep_dims", keepdim)
      .Run();
  return result;
}

at::Tensor check_dtype(
    const at::Tensor &self,
    c10::ScalarType out_type) {
  if (isIntegralType(out_type, true)) {
    out_type = at::kFloat;
  }
  at::Tensor self_cp = (self.scalar_type() == out_type) ? self :
      NPUNativeFunctions::npu_dtype_cast(self, out_type);
  return self_cp;
}

at::Tensor &NPUNativeFunctions::sum_out(
    const at::Tensor &self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor &result) {
  auto output_size = sum_npu_output_size(self, dim, keepdim);
  auto res_type = dtype.has_value() ? dtype.value() : result.scalar_type();

  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      res_type,
      output_size);

  if (self.numel() == 0) {
    at::Tensor result_cast = at::empty(output_size, self.options().dtype(res_type));
    result.copy_(result_cast);
    return result;
  }

  at::Tensor self_cp = check_dtype(self, res_type);
  at::Tensor result_cp = result.scalar_type() == self_cp.scalar_type() ? result :
      NPUNativeFunctions::npu_dtype_cast(result, self_cp.scalar_type());

  sum_out_npu_nocheck(result_cp, self_cp, dim, keepdim);
  if (result_cp.scalar_type() != res_type) {
    result_cp = NPUNativeFunctions::npu_dtype_cast(result_cp, res_type);
    result.copy_(result_cp);
  } else {
    result = result_cp;
  }
  return result;
}

at::Tensor &NPUNativeFunctions::sum_out(
    const at::Tensor &self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor &result) {
  return NPUNativeFunctions::sum_out(self, dimnames_to_positions(self, dim), keepdim, dtype, result);
}

at::Tensor NPUNativeFunctions::sum(
    const at::Tensor &self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  auto output_size = reduce_ops_npu_output_size(self, dim, keepdim);
  auto out_type = self.scalar_type();

  if (dtype.has_value()) {
    out_type = dtype.value();
  } else if (isIntegralType(out_type, true)) {
    out_type = at::kLong;
  }

  if (self.numel() == 0) {
    return at::zeros(output_size, self.options().dtype(out_type));
  }

  at::Tensor self_cp = check_dtype(self, out_type);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      output_size, self_cp.options(), ACL_FORMAT_ND);
  sum_out_npu_nocheck(result, self_cp, dim, keepdim);

  if (result.scalar_type() != out_type) {
    result = NPUNativeFunctions::npu_dtype_cast(result, out_type);
  }
  return result;
}

at::Tensor NPUNativeFunctions::sum(
    const at::Tensor &self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  return NPUNativeFunctions::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

at::Tensor NPUNativeFunctions::sum(const at::Tensor &self, c10::optional<c10::ScalarType> dtype) {
  return NPUNativeFunctions::sum(self, c10::SmallVector<int64_t, N>{}, false, dtype);
}

} // namespace native
} // namespace at_npu
