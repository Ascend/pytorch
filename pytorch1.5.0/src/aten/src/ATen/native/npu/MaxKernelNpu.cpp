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

tuple<Tensor&, Tensor&> max_out_npu_nocheck(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  OpCommand cmd;
  cmd.Name("ArgMaxWithValue")
      .Input(self)
      .Output(indices)
      .Output(output)
      .Attr("dimension", dim)
      .Attr("keep_dims", keepdim)
      .Run();
  return tuple<Tensor&, Tensor&>(output, indices);
}

tuple<Tensor&, Tensor&> max_out_npu(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  SmallVector<int64_t, SIZE> indicesSize = outputSize;

  auto func = [&self, dim, keepdim](Tensor& output, Tensor& indices) {
    max_out_npu_nocheck(output, indices, self, dim, keepdim);
  };

  OpPipeWithDefinedOut check;
  check.CheckMemory({self}, {output, indices});

  Tensor indices_tmp;
  OpPipeWithMultiOut<Tensor&, Tensor&> pipe(output, indices_tmp);
  return pipe.FixOutputSizeAndFormat<0>({self}, self, ACL_FORMAT_ND, outputSize)
            .ApplyOutputWithSpecailParams<1>(indicesSize, self.options().dtype(ScalarType::Int), ACL_FORMAT_ND) // use default format
            .Call(func)
            .ReflushOutputDtype<1>(ScalarType::Long)
            .FixOutputExceptDtype<1>({self}, ACL_FORMAT_ND, ScalarType::Long, indicesSize)
            .FixOutputWithReplace<1>(indices)
            .ReturnRef<Tensor&, Tensor&>();
}

tuple<Tensor, Tensor> max_npu(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor selfCast = self;
  if(self.dtype() == ScalarType::Bool || self.dtype() == ScalarType::Int){
    selfCast = self.to(ScalarType::Float);
  }

  SmallVector<int64_t, SIZE> dims = {dim};
  auto outputSize = reduce_ops_npu_output_size(selfCast, dims, keepdim);
  SmallVector<int64_t, SIZE> indicesSize = outputSize;

  auto func = [&selfCast, dim, keepdim](Tensor outputs, Tensor indices) {
    max_out_npu_nocheck(outputs, indices, selfCast, dim, keepdim);
  };

  Tensor outputs, indices;
  OpPipeWithDefinedMultiOut<Tensor, Tensor> pipe(outputs, indices);
  std::tie(outputs, indices) = pipe.ApplyOutputWithSpecailParams<0>(outputSize, selfCast.options(), ACL_FORMAT_ND)
      .ApplyOutputWithSpecailParams<1>(indicesSize, selfCast.options().dtype(ScalarType::Int), ACL_FORMAT_ND) // use default format
      .Call(func)
      .ReflushOutputDtype<1>(ScalarType::Long)
      .Return<Tensor, Tensor>();

  if(self.dtype() == ScalarType::Bool || self.dtype() == ScalarType::Int){
    outputs = outputs.to(self.dtype());
  }

  return std::tie(outputs, indices);
}

tuple<Tensor&, Tensor&> max_out_npu(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return max_out_npu(
      output, indices, self, dimname_to_position(self, dim), keepdim);
}

tuple<Tensor, Tensor> max_npu(const Tensor& self, Dimname dim, bool keepdim) {
  return max_npu(self, dimname_to_position(self, dim), keepdim);
}

tuple<Tensor&, Tensor&> _max_out_npu(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return max_out_npu(output, indices, self, dim, keepdim);
}

tuple<Tensor, Tensor> _max_npu(const Tensor& self, int64_t dim, bool keepdim) {
  return max_npu(self, dim, keepdim);
}

Tensor& max_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  OpCommand cmd;
  cmd.Name("Maximum")
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

Tensor& max_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      self.scalar_type(),
      self.sizes());
  max_out_npu_nocheck(result, self, other);

  return result;
}

Tensor max_npu(const Tensor& self, const Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  max_out_npu_nocheck(result, self, other);
  return result;
}

Tensor& max_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim) {
  OpCommand cmd;
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    cmd.Name("ReduceMax")
      .Input(self)
      .Input(dims)
      .Output(result)
      .Attr("keep_dims", keepdim)
      .Run();
  } else {
    cmd.Name("ReduceMaxD")
      .Input(self)
      .Output(result)
      .Attr("axes", dims)
      .Attr("keep_dims", keepdim)
      .Run();
  }
    return result;
}

Tensor max_npu(const Tensor& self, IntArrayRef dims, bool keepdim) {
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);
  int64_t npu_format = CalcuOpUtil::get_tensor_npu_format(self);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  Tensor result = at::empty_with_format(outputSize, self.options(), npu_format);
  max_out_npu_nocheck(result, self, dims, keepdim);
  return result;
}

Tensor max_npu(const Tensor& self, DimnameList dims, bool keepdim) {
  return max_npu(self, dimnames_to_positions(self, dims), keepdim);
}

Tensor max_npu(const Tensor& self) {
  SmallVector<int64_t, SIZE> dims = CalcuOpUtil::get_dimlist_for_tensor(self);
  return max_npu(self, dims, false);
}

} // namespace native
} // namespace at