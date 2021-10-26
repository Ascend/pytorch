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

tuple<Tensor&, Tensor&> min_out_npu_nocheck(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
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

tuple<Tensor&, Tensor&> min_dim_out_npu(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& output,
    Tensor& indices) {
  SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  SmallVector<int64_t, SIZE> indicesSize = outputSize;

  auto func = [&self, dim, keepdim](Tensor& output, Tensor& indices) {
    min_out_npu_nocheck(output, indices, self, dim, keepdim);
  };

  Tensor indices_tmp;
  OpPipeWithMultiOut<Tensor&, Tensor&> pipe(output, indices_tmp);
  return pipe.FixOutputSizeAndFormat<0>({self}, self, ACL_FORMAT_ND, outputSize)
            .ApplyOutputWithSpecailParams<1>(indicesSize, self.options().dtype(ScalarType::Int), ACL_FORMAT_ND)
            .Call(func)
            .ReflushOutputDtype<1>(ScalarType::Long)
            .FixOutputExceptDtype<1>({self}, ACL_FORMAT_ND, ScalarType::Long, indicesSize)
            .FixOutputWithReplace<1>(indices)
            .ReturnRef<Tensor&, Tensor&>();
}

tuple<Tensor, Tensor> min_dim_npu(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor selfCast = self;
  if(self.dtype() == ScalarType::Bool){
    selfCast = self.to(ScalarType::Float);
  }

  SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(selfCast, dims, keepdim);
  SmallVector<int64_t, SIZE> indicesSize = outputSize;

  auto func = [&selfCast, dim, keepdim](Tensor outputs, Tensor indices) {
    min_out_npu_nocheck(outputs, indices, selfCast, dim, keepdim);
  };

  Tensor outputs, indices;
  OpPipeWithDefinedMultiOut<Tensor, Tensor> pipe(outputs, indices);
  std::tie(outputs, indices) = pipe.ApplyOutputWithSpecailParams<0>(outputSize, selfCast.options(), ACL_FORMAT_ND)
      .ApplyOutputWithSpecailParams<1>(indicesSize, selfCast.options().dtype(ScalarType::Int), ACL_FORMAT_NCHW)
      .Call(func)
      .ReflushOutputDtype<1>(ScalarType::Long)
      .Return<Tensor, Tensor>();

  if(self.dtype() == ScalarType::Bool){
    outputs = outputs.to(ScalarType::Bool);
  }

  return std::tie(outputs, indices);
}

tuple<Tensor&, Tensor&> min_names_dim_out_npu(
    const Tensor& self,
    Dimname dim,
    bool keepdim,
    Tensor& output,
    Tensor& indices) {
  return min_dim_out_npu(self, dimname_to_position(self, dim), keepdim, output, indices);
}

tuple<Tensor, Tensor> min_names_dim_npu(const Tensor& self, Dimname dim, bool keepdim) {
  return min_dim_npu(self, dimname_to_position(self, dim), keepdim);
}

Tensor& min_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  OpCommand cmd;
  cmd.Name("Minimum")
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

Tensor& min_out_npu(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  OpPreparation::CheckOut(
      {self}, 
      result, 
      ACL_FORMAT_ND,
      self.scalar_type(), 
      self.sizes());
  min_out_npu_nocheck(result, self, other);

  return result;
}

Tensor min_other_npu(const Tensor& self, const Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  min_out_npu_nocheck(result, self, other);
  return result;
}

Tensor& min_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim) {
  OpCommand cmd;
  cmd.Name("ReduceMin")
    .Input(self)
    .Input(dims)
    .Output(result)
    .Attr("keep_dims", keepdim)
    .Run();

  return result;
}

Tensor min_values_npu(const Tensor& self, IntArrayRef dims, bool keepdim) {
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_NCHW;
  }
  Tensor result = OpPreparation::ApplyTensorWithFormat(self, outputSize, npu_format);
  min_out_npu_nocheck(result, self, dims, keepdim);
  return result;
}

Tensor min_values_names_npu(const Tensor& self, DimnameList dims, bool keepdim) {
  return min_values_npu(self, dimnames_to_positions(self, dims), keepdim);
}

Tensor min_npu(const Tensor& self) {
  SmallVector<int64_t, SIZE> dims = CalcuOpUtil::get_dimlist_for_tensor(self);
  return min_values_npu(self, dims, false);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("min", TORCH_FN(min_npu));
  m.impl("min.other", TORCH_FN(min_other_npu));
  m.impl("min.out", TORCH_FN(min_out_npu));
  m.impl("min.dim", TORCH_FN(min_dim_npu));
  m.impl("min.dim_min", TORCH_FN(min_dim_out_npu));
  m.impl("min_values", TORCH_FN(min_values_npu));
  m.impl("min.names_dim", TORCH_FN(min_names_dim_npu));
  m.impl("min.names_dim_min", TORCH_FN(min_names_dim_out_npu));
  m.impl("min_values.names", TORCH_FN(min_values_names_npu));
}
} // namespace native
} // namespace at
