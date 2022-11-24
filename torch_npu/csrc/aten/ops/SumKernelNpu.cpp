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
  c10::SmallVector<int64_t, N> dimList = dim.empty() ? CalcuOpUtil::get_dimlist_for_tensor(self) : c10::SmallVector<int64_t, N>(dim);
  OpCommand cmd;
  cmd.Name("ReduceSum")
      .Input(self)
      .Input(dimList, at::kLong)
      .Output(result)
      .Attr("keep_dims", keepdim)
      .Run();
  return result;
}

at::Tensor &NPUNativeFunctions::sum_out(
    const at::Tensor &self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor &result) {
  at::Tensor self_cp = self;
  at::Tensor result_cp = result;

  auto outputSize = sum_npu_output_size(self_cp, dim, keepdim);
  auto res_type = dtype.has_value() ? dtype.value() : result.scalar_type();

  OpPreparation::CheckOut(
      {self_cp},
      result_cp,
      ACL_FORMAT_ND,
      res_type,
      outputSize);

  auto selfSize = self_cp.sizes();
  for (int64_t i = 0; i < selfSize.size(); i++) {
    if (selfSize[i] == 0) {
      at::Tensor result_cast = at::empty(outputSize);
      result_cp.copy_(result_cast);
      return result_cp;
    }
  }

  OpPipeWithDefinedOut pipe;
  pipe.CheckMemory({self_cp}, {result_cp});

  if (self.scalar_type() == at::kBool) {
    self_cp = NPUNativeFunctions::npu_dtype_cast(self_cp, at::kFloat);
    result_cp = NPUNativeFunctions::npu_dtype_cast(result_cp, at::kFloat);
  }
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
  at::Tensor self_cp = self.scalar_type() == at::kBool ? NPUNativeFunctions::npu_dtype_cast(self, at::kFloat) : self;
  auto outputSize = reduce_ops_npu_output_size(self_cp, dim, keepdim);
  auto selfSize = self_cp.sizes();
  auto out_type = self.scalar_type();

  if (dtype.has_value()) {
    out_type = dtype.value();
  } else if (isIntegralType(out_type, true)) {
    out_type = at::kLong;
  }

  for (int64_t i = 0; i < selfSize.size(); i++) {
    if (selfSize[i] == 0) {
      return at::zeros(outputSize, self_cp.options());
    }
  }

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self_cp.options(), ACL_FORMAT_ND);
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
