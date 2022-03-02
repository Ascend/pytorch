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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor&, Tensor&> topk_out_npu_no_transpose(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  SmallVector<int64_t, N> kVec = {k};
  Tensor kCpuTensor = from_blob((void*)kVec.data(), {1}, at::kLong).to(at::kInt);
  if (!c10::npu::OptionsManager::CheckDynamicEnable()){
    OpCommand cmd;
    cmd.Name("TopKV2")
      .Input(self)
      .Input(kCpuTensor, kVec, "k")
      .Output(values)
      .Output(indices)
      .Attr("dim", dim)
      .Attr("largest", largest)
      .Attr("sorted", sorted)
      .Run();
  } else{
    OpDynamicCommand cmd;
    // Although the value is fixed to false, only the value of sorted can be true.
    cmd.Name("TopKV2")
      .Input(self)
      .Input(kCpuTensor, kVec, "k")
      .Output(values)
      .Output(indices)
      .Attr("dim", dim)
      .Attr("largest", largest)
      .Attr("sorted", sorted);
    cmd.DynamicName("TopKV2")
        .DynamicInput(self)
        .DynamicInput(kVec, at::kLong, at::kInt, "k")
        .DynamicOutput(values)
        .DynamicOutput(indices)
        .DynamicAttr("dim", dim)
        .DynamicAttr("largest", largest)
        .DynamicAttr("sorted", sorted)
        .DynamicOpRun();
  }
  return tuple<Tensor&, Tensor&>(values, indices);
}

tuple<Tensor&, Tensor&> topk_out_npu_nocheck(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::make_wrap_dim(-1, self.dim());

  if (dim != lastDim) {
    SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);

    // construct the output tensor of the NPU
    Tensor transposeSelf = at::npu_transpose(self, perm);
    auto outputSize = transpose_npu_output_size(values, perm);
    Tensor transposeValue = at::empty_with_format(
        outputSize,
        values.options(),
        CalcuOpUtil::get_tensor_npu_format(values));
    Tensor transposeIndices = at::empty_with_format(
        outputSize,
        indices.options(),
        CalcuOpUtil::get_tensor_npu_format(indices));
    topk_out_npu_no_transpose(
        transposeValue,
        transposeIndices,
        transposeSelf,
        k,
        lastDim,
        largest,
        sorted);
    at::npu_transpose_out(values, transposeValue, perm);
    at::npu_transpose_out(indices, transposeIndices, perm);
  } else {
    topk_out_npu_no_transpose(
        values, indices, self, k, lastDim, largest, sorted);
  }

  return tuple<Tensor&, Tensor&>(values, indices);
}

tuple<Tensor&, Tensor&> topk_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  Tensor selfCp = OpPreparation::CastBackToOriFormat(self);

  // calculate the output size
  auto outputSize = topk_npu_output_size(selfCp, k, dim, largest, sorted);
  SmallVector<int64_t, SIZE> indicesSize = outputSize;

  // calculate the output result of the NPU
  auto func = [&selfCp, k, dim, largest, sorted](Tensor& values, Tensor& indices) {
    topk_out_npu_nocheck(values, indices, selfCp, k, dim, largest, sorted);
  };

  Tensor indices_tmp;
  OpPipeWithMultiOut<Tensor&, Tensor&> pipe(values, indices_tmp);
  return pipe.FixOutputSizeAndFormat<0>({selfCp}, selfCp, CalcuOpUtil::get_tensor_npu_format(selfCp), outputSize)
      .ApplyOutputWithSpecailParams<1>(indicesSize, selfCp.options().dtype(kInt), ACL_FORMAT_ND)
      .Call(func)
      .ReflushOutputDtype<1>(ScalarType::Long)
      .FixOutputExceptDtype<1>({selfCp}, ACL_FORMAT_ND, ScalarType::Long, indicesSize)
      .FixOutputWithReplace<1>(indices)
      .ReturnRef<Tensor&, Tensor&>();
}

tuple<Tensor, Tensor> topk_npu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  Tensor selfCp = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = topk_npu_output_size(selfCp, k, dim, largest, sorted);
  // construct the output tensor of the NPU
  Tensor values = at::empty_with_format(
      outputSize, selfCp.options(), CalcuOpUtil::get_tensor_npu_format(selfCp));
  Tensor indices = at::empty_with_format(
      outputSize, selfCp.options().dtype(kInt), ACL_FORMAT_ND);

  // calculate the output result of the NPU
  topk_out_npu_nocheck(values, indices, selfCp, k, dim, largest, sorted);

  // indices dtype transform Int64
  indices = indices.to(at::kLong);

  return tuple<Tensor, Tensor>(values, indices);
}

} // namespace native
} // namespace at
