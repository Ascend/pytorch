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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> topk_out_npu_no_transpose(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  c10::SmallVector<int64_t, N> kVec = {k};
  OpCommand cmd;
  cmd.Name("TopKV2")
    .Input(self)
    .Input(kVec, at::kInt) 
    .Output(values)
    .Output(indices)
    .Attr("dim", dim)
    .Attr("largest", largest)
    .Attr("sorted", sorted)
    .Run();
  return tuple<at::Tensor&, at::Tensor&>(values, indices);
}

tuple<at::Tensor&, at::Tensor&> topk_out_npu_nocheck(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {

  dim = CalcuOpUtil::MakeWrapDim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::MakeWrapDim(-1, self.dim());

  if (dim != lastDim) {
    c10::SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);

    // construct the output tensor of the NPU
    at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm, true);
    auto outputSize = transpose_npu_output_size(values, perm);
    at::Tensor transposeValue = OpPreparation::ApplyTensor(values, outputSize);
    at::Tensor transposeIndices = OpPreparation::ApplyTensor(indices, outputSize);
    topk_out_npu_no_transpose(
        transposeValue,
        transposeIndices,
        transposeSelf,
        k,
        lastDim,
        largest,
        sorted);
    NPUNativeFunctions::npu_transpose_out(transposeValue, perm, true, values);
    NPUNativeFunctions::npu_transpose_out(transposeIndices, perm, true, indices);
  } else {
    topk_out_npu_no_transpose(
        values, indices, self, k, lastDim, largest, sorted);
  }

  return tuple<at::Tensor&, at::Tensor&>(values, indices);
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::topk_out(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    at::Tensor& values,
    at::Tensor& indices) {
  at::Tensor selfCp = OpPreparation::CastBackToOriFormat(self);

  // calculate the output size
  auto outputSize = topk_npu_output_size(selfCp, k, dim, largest, sorted);
  OpPreparation::CheckOut(
      {self},
      values,
      self,
      outputSize);
  OpPreparation::CheckOut(
      {self},
      indices,
      ACL_FORMAT_ND,
      at::ScalarType::Long,
      outputSize);
  c10::SmallVector<int64_t, SIZE> indicesSize = outputSize;

  // calculate the output result of the NPU
  auto func = [&selfCp, k, dim, largest, sorted](at::Tensor& values, at::Tensor& indices) {
    topk_out_npu_nocheck(values, indices, selfCp, k, dim, largest, sorted);
  };

  at::Tensor indices_tmp;
  OpPipeWithMultiOut<at::Tensor&, at::Tensor&> pipe(values, indices_tmp);
  return pipe.FixOutputSizeAndFormat<0>({selfCp}, selfCp, CalcuOpUtil::GetTensorNpuFormat(selfCp), outputSize)
      .ApplyOutputWithSpecailParams<1>(indicesSize, selfCp.options().dtype(at::kInt), ACL_FORMAT_ND)
      .Call(func)
      .ReflushOutputDtype<1>(at::ScalarType::Long)
      .FixOutputExceptDtype<1>({selfCp}, ACL_FORMAT_ND, at::ScalarType::Long, indicesSize)
      .FixOutputWithReplace<1>(indices)
      .ReturnRef<at::Tensor&, at::Tensor&>();
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::topk(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  at::Tensor selfCp = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = topk_npu_output_size(selfCp, k, dim, largest, sorted);
  // construct the output tensor of the NPU
  at::Tensor values = OpPreparation::ApplyTensor(selfCp, outputSize);
  at::Tensor indices = OpPreparation::ApplyTensorWithFormat(
      outputSize, selfCp.options().dtype(at::kInt), ACL_FORMAT_ND);
  // calculate the output result of the NPU
  topk_out_npu_nocheck(values, indices, selfCp, k, dim, largest, sorted);

  // indices dtype transform Int64
  indices = NPUNativeFunctions::npu_dtype_cast(indices, at::kLong);

  return tuple<at::Tensor, at::Tensor>(values, indices);
}
} // namespace native
} // namespace at_npu
