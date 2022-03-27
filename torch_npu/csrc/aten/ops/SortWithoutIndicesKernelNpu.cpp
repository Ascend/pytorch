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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

at::Tensor& sort_without_indices_no_transpose(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  OpCommand cmd;
  cmd.Name("SortV2")
      .Input(self)
      .Output(result)
      .Attr("axis", dim)
      .Attr("descending", descending)
      .Run();
  
  return result;
}

at::Tensor& NPUNativeFunctions::npu_sort_v2_out(
    const at::Tensor& self,
    int64_t dim,
    bool descending,
    at::Tensor& result) {
  auto outputSize = input_same_output_size(self);

  OpPreparation::CheckOut(
      {self},
      result,
      self);
  
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::make_wrap_dim(-1, self.dim());

  if (dim != lastDim) {
    c10::SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);
    at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm);

    auto outputSize = transpose_npu_output_size(result, perm);
    at::Tensor transposeResult = OpPreparation::ApplyTensorWithSizes(outputSize, result.options());

    sort_without_indices_no_transpose(transposeResult, transposeSelf, lastDim, descending);
    NPUNativeFunctions::npu_transpose_out(transposeResult, perm, result);
  } else {
    if (!NpuUtils::check_match(&result)) {
      at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
      sort_without_indices_no_transpose(contiguousResult, self, dim, descending);
      NpuUtils::format_fresh_view(result, contiguousResult);
    } else {
      sort_without_indices_no_transpose(result, self, dim, descending);
    }
  }

  return result;
}

at::Tensor NPUNativeFunctions::npu_sort_v2(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(self);

  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::make_wrap_dim(-1, self.dim());

  if (dim != lastDim) {
    c10::SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);
    at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm);

    auto outputSize = transpose_npu_output_size(result, perm);
    at::Tensor transposeResult = OpPreparation::ApplyTensorWithSizes(outputSize, result.options());

    sort_without_indices_no_transpose(transposeResult, transposeSelf, lastDim, descending);
    NPUNativeFunctions::npu_transpose_out(transposeResult, perm, result);
  } else {
    sort_without_indices_no_transpose(result, self, dim, descending);
  }

  return result;
}
} // namespace native
} // namespace at_npu