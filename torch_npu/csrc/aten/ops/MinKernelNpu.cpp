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

#include "torch_npu/csrc/aten/XLANativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> min_out_npu_nocheck(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  OpCommand cmd;
  cmd.Name("ArgMinWithValue")
      .Input(self)
      .Output(indices)      
      .Output(output)
      .Attr("dimension", dim)
      .Attr("keep_dims", keepdim)
      .Run();
  return std::tie(output, indices);
}

tuple<at::Tensor&, at::Tensor&> XLANativeFunctions::min_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  c10::SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  c10::SmallVector<int64_t, SIZE> indicesSize = outputSize;
  auto func = [&self, dim, keepdim](at::Tensor& output, at::Tensor& indices) {
    min_out_npu_nocheck(self, dim, keepdim, output, indices);
  };

  at::Tensor indices_tmp;
  OpPipeWithMultiOut<at::Tensor&, at::Tensor&> pipe(output, indices_tmp);
  return pipe.FixOutputSizeAndFormat<0>({self}, self, ACL_FORMAT_ND, outputSize)
            .ApplyOutputWithSpecailParams<1>(indicesSize, self.options().dtype(at::ScalarType::Int), ACL_FORMAT_ND)
            .Call(func)
            .ReflushOutputDtype<1>(at::ScalarType::Long)
            .FixOutputExceptDtype<1>({self}, ACL_FORMAT_ND, at::ScalarType::Long, indicesSize)
            .FixOutputWithReplace<1>(indices)
            .ReturnRef<at::Tensor&, at::Tensor&>();
}

tuple<at::Tensor, at::Tensor> XLANativeFunctions::min(const at::Tensor& self, int64_t dim, bool keepdim) {
  at::Tensor selfCast = self;
  if(self.dtype() == at::ScalarType::Bool){
    selfCast = self.to(at::ScalarType::Float);
  }
  c10::SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(selfCast, dims, keepdim);
  c10::SmallVector<int64_t, SIZE> indicesSize = outputSize;
  auto func = [&selfCast, dim, keepdim](at::Tensor outputs, at::Tensor indices) {
    min_out_npu_nocheck(selfCast, dim, keepdim, outputs, indices);
  };

  at::Tensor outputs, indices;
  OpPipeWithDefinedMultiOut<at::Tensor, at::Tensor> pipe(outputs, indices);
  std::tie(outputs, indices) = pipe.ApplyOutputWithSpecailParams<0>(outputSize, selfCast.options(), ACL_FORMAT_ND)
      .ApplyOutputWithSpecailParams<1>(indicesSize, selfCast.options().dtype(at::ScalarType::Int), ACL_FORMAT_NCHW)
      .Call(func)
      .ReflushOutputDtype<1>(at::ScalarType::Long)
      .Return<at::Tensor, at::Tensor>();

  if(self.dtype() == at::ScalarType::Bool){
    outputs = outputs.to(at::ScalarType::Bool);
  }
  return std::tie(outputs, indices);
}

tuple<at::Tensor&, at::Tensor&> XLANativeFunctions::min_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices) {
  return min_out(self, dimname_to_position(self, dim), keepdim, output, indices);
}

tuple<at::Tensor, at::Tensor> XLANativeFunctions::min(const at::Tensor& self, at::Dimname dim, bool keepdim) {
  return min(self, dimname_to_position(self, dim), keepdim);
}

at::Tensor& min_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Minimum")
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& XLANativeFunctions::min_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self}, 
      result, 
      ACL_FORMAT_ND,
      self.scalar_type(), 
      self.sizes());
  min_out_npu_nocheck(self, other, result);
  return result;
}

at::Tensor XLANativeFunctions::minimum(const at::Tensor& self, const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  min_out_npu_nocheck(self, other, result);
  return result;
}

at::Tensor& min_out_npu_nocheck(
    const at::Tensor& self,
    at::IntArrayRef dims,
    bool keepdim,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("ReduceMin")
    .Input(self)
    .Input(dims)
    .Output(result)
    .Attr("keep_dims", keepdim)
    .Run();
  return result;
}

at::Tensor XLANativeFunctions::amin(const at::Tensor& self, at::IntArrayRef dims, bool keepdim) {
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_NCHW;
  }
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(self, outputSize, npu_format);
  min_out_npu_nocheck(self, dims, keepdim, result);
  return result;
}

at::Tensor XLANativeFunctions::min(const at::Tensor& self) {
  c10::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::get_dimlist_for_tensor(self);
  return amin(self, dims, false);
}

at::Tensor& XLANativeFunctions::amin_out(
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
  min_out_npu_nocheck(self, dims, keepdim, result);
  return result;
}
} // namespace native
} // namespace at_npu
