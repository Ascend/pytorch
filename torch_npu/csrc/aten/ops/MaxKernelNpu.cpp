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

tuple<at::Tensor&, at::Tensor&> max_out_npu_nocheck(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  OpCommand cmd;
  cmd.Name("ArgMaxWithValue")
      .Input(self)
      .Output(indices)
      .Output(output)
      .Attr("dimension", dim)
      .Attr("keep_dims", keepdim)
      .Run();
  return std::tie(output, indices);
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::max_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  at::SmallVector<int64_t, SIZE> indicesSize = outputSize;

  OpPreparation::CheckOut(
      {self},
      output,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      outputSize);
  OpPreparation::CheckOut(
      {self},
      indices,
      ACL_FORMAT_ND,
      at::ScalarType::Long,
      outputSize);
  auto func = [&self, dim, keepdim](at::Tensor& output, at::Tensor& indices) {
    max_out_npu_nocheck(self, dim, keepdim, output, indices);
  };

  at::Tensor indices_tmp;
  OpPipeWithMultiOut<at::Tensor&, at::Tensor&> pipe(output, indices_tmp);
  return pipe.FixOutputSizeAndFormat<0>({self}, self, ACL_FORMAT_ND, outputSize)
            .ApplyOutputWithSpecailParams<1>(
                indicesSize, 
                self.options().dtype(at::ScalarType::Int), 
                ACL_FORMAT_ND)
            .Call(func)
            .ReflushOutputDtype<1>(at::ScalarType::Long)
            .FixOutputExceptDtype<1>({self}, ACL_FORMAT_ND, at::ScalarType::Long, indicesSize)
            .FixOutputWithReplace<1>(indices)
            .ReturnRef<at::Tensor&, at::Tensor&>();
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::max_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  return max_out(self, dimname_to_position(self, dim), keepdim, output, indices);
  }

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::max(
    const at::Tensor& self, 
    int64_t dim, 
    bool keepdim) {
  at::Tensor selfCast = self;
  if(self.dtype() == at::ScalarType::Bool || self.dtype() == at::ScalarType::Int){
    selfCast = self.to(at::ScalarType::Float);
  }
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(selfCast, dims, keepdim);
  at::SmallVector<int64_t, SIZE> indicesSize = outputSize;
  auto func = [&selfCast, dim, keepdim](at::Tensor outputs, at::Tensor indices) {
    max_out_npu_nocheck(selfCast, dim, keepdim, outputs, indices);
  };

  at::Tensor outputs, indices;
  OpPipeWithDefinedMultiOut<at::Tensor, at::Tensor> pipe(outputs, indices);
  std::tie(outputs, indices) = pipe.ApplyOutputWithSpecailParams<0>(outputSize, selfCast.options(), ACL_FORMAT_ND)
      .ApplyOutputWithSpecailParams<1>(indicesSize, selfCast.options().dtype(at::ScalarType::Int), ACL_FORMAT_ND) // use default format
      .Call(func)
      .ReflushOutputDtype<1>(at::ScalarType::Long)
      .Return<at::Tensor, at::Tensor>();
  if(self.dtype() == at::ScalarType::Bool || self.dtype() == at::ScalarType::Int){
    outputs = outputs.to(self.dtype());
  }
  return std::tie(outputs, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::max(
    const at::Tensor& self, 
    at::Dimname dim, 
    bool keepdim) {
  return at::max(self, dimname_to_position(self, dim), keepdim);
}

at::Tensor& max_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Maximum")
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::max_out(
    const at::Tensor& self, 
    const at::Tensor& other,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      self.sizes());
  max_out_npu_nocheck(self, other, result);
  return result;
}

at::Tensor NPUNativeFunctions::max(
    const at::Tensor& self, 
    const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  max_out_npu_nocheck(self, other, result);
  return result;
}

at::Tensor& max_out_npu_nocheck(
    const at::Tensor& self,
    at::IntArrayRef dims,
    bool keepdim,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("ReduceMax")
      .Input(self)
      .Input(dims)
      .Output(result)
      .Attr("keep_dims", keepdim)
      .Run();
    return result;
}

at::Tensor NPUNativeFunctions::amax(
    const at::Tensor& self, 
    at::IntArrayRef dims, 
    bool keepdim) {
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(self, outputSize, npu_format);
  max_out_npu_nocheck(self, dims, keepdim, result);
  return result;
}

at::Tensor NPUNativeFunctions::max(
    const at::Tensor& self) {
  at::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::get_dimlist_for_tensor(self);
  return amax(self, dims, false);
}

at::Tensor& NPUNativeFunctions::amax_out(
    const at::Tensor& self, 
    at::IntArrayRef dims, 
    bool keepdim,
    at::Tensor& result) {
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      outputSize);
  max_out_npu_nocheck(self, dims, keepdim, result);
  return result;
}

} // namespace native
} // namespace at_npu 
