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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> sort_out_npu_no_transpose(
    const at::Tensor& self,
    int64_t dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices) {
  OpCommand cmd;
  cmd.Name("Sort")
     .Input(self)
     .Output(values)
     .Output(indices)
     .Attr("axis", dim)
     .Attr("descending", descending)
     .Run();

  return std::tie(values, indices);
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::sort_out(
    const at::Tensor& self,
    int64_t dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices) {
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::make_wrap_dim(-1, self.dim());

  if (dim != lastDim) {
    at::SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);

    at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm);
    auto outputSize = transpose_npu_output_size(values, perm);
    at::Tensor transposeValues = OpPreparation::ApplyTensor(values, outputSize);
    at::Tensor transposeIndices =OpPreparation::ApplyTensor(indices, outputSize);

    sort_out_npu_no_transpose(
        transposeSelf, lastDim, descending, transposeValues, transposeIndices);
    
    NPUNativeFunctions::npu_transpose_out(transposeValues, perm, values);
    NPUNativeFunctions::npu_transpose_out(transposeIndices, perm, indices);
  } else {
    sort_out_npu_no_transpose(
        self, lastDim, descending, values, indices);
  }
  
  // indices dtype transform Int64
  indices = indices.to(at::kLong);
  
  return std::tie(values, indices);
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::sort_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices) {
  return NPUNativeFunctions::sort_out(self, dimname_to_position(self, dim), descending, values, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::sort(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  auto outputSize = input_same_output_size(self);

  at::Tensor values = OpPreparation::ApplyTensor(self);
  at::Tensor indices = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options().dtype(at::kInt), ACL_FORMAT_NCHW);

  NPUNativeFunctions::sort_out(self, dim, descending, values, indices);

  return std::tie(values, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::sort(
    const at::Tensor& self,
    at::Dimname dim,
    bool descending) {
  return NPUNativeFunctions::sort(self, dimname_to_position(self, dim), descending);
}

} // namespace native
} // namespace at_npu
