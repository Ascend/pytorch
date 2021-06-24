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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> cummin_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> cummin_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> cummin_npu_attr(int64_t dim) {
  NPUAttrDesc npuAttrDim = NPUAttrDesc("dim", dim);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrDim};
  return attrs;
}

tuple<Tensor&, Tensor&> cummin_out_npu_no_transpose (   
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim) {
  // constructs the input and output NPUTensorDesc
  auto inputs = cummin_npu_input({self});
  auto outputs = cummin_npu_output({values, indices});
  // constructs the attr of the NPUAttrDesc
  auto attrs = cummin_npu_attr(dim);
  
  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Cummin", inputs, outputs, attrs);
  return std::tie(values, indices);
}

tuple<Tensor&, Tensor&> cummin_out_npu (   
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim) {
  auto indices_dtype = indices.scalar_type();
  if(indices_dtype != c10::ScalarType::Long) {
    AT_ERROR("indices must be int64 type.");
  }
  Tensor indices_int32 = indices.to(at::kInt);
  const auto names = self.names();
  Tensor self_no_name = self.rename(nullopt);
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  int64_t firstDim = CalcuOpUtil::make_wrap_dim(0, self.dim());
  if (dim != firstDim) {
    SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[firstDim]);
    // construct the output tensor of the NPU
    Tensor transposeSelf = at::npu_transpose(self_no_name, perm);
    auto outputSize = transpose_npu_output_size(values, perm);
    Tensor transposeValue = at::empty_with_format(
        outputSize,
        values.options(),
        CalcuOpUtil::get_tensor_npu_format(values));
    Tensor transposeIndices = at::empty_with_format(
        outputSize,
        indices_int32.options(),
        CalcuOpUtil::get_tensor_npu_format(indices_int32));
    cummin_out_npu_no_transpose(
        transposeValue,
        transposeIndices,
        transposeSelf,
        firstDim);
    at::npu_transpose_out(values, transposeValue, perm);
    at::npu_transpose_out(indices_int32, transposeIndices, perm);
  } else {
    cummin_out_npu_no_transpose(
        values, indices_int32, self, firstDim);
  }
  indices.npu_dtype_cast_(indices_int32);
  values.rename_(names);
  indices.rename_(names);
  return std::tie(values, indices);
}

tuple<Tensor, Tensor> cummin_npu(const Tensor& self, int64_t dim) {
  // calculate the output size  
  auto outputSize = input_same_output_size(self);

  Tensor values = at::empty_with_format(
      outputSize,
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));
  Tensor indices = at::empty_with_format(
      outputSize, 
      self.options().dtype(kLong), 
      CalcuOpUtil::get_tensor_npu_format(self));

  cummin_out_npu(values, indices, self, dim);

  return std::tie(values, indices);
}

tuple<Tensor&, Tensor&> cummin_out_npu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Dimname dim) {
  // dimname_to_position(self, dim) :  Dimname --> Int
  return cummin_out_npu(
      values, indices, self, dimname_to_position(self, dim));
}

tuple<Tensor, Tensor> cummin_npu(const Tensor& self, Dimname dim) {
  return cummin_npu(self, dimname_to_position(self, dim));
}

} // namespace native
} // namespace at