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

at::Tensor& scatter_npu_nocheck(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  OpCommand cmd;
  cmd.Name("ScatterElements")
     .Input(self)
     .Input(index)
     .Input(src)
     .Output(self)
     .Attr("axis", dim)
     .Run();
  return self;
}

at::Tensor& scatter_npu_src_impl(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index_ex,
    const at::Tensor& src_ex) {
  at::ScalarType selfType = self.scalar_type();
  if (selfType == at::ScalarType::Half) {
    self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }

  at::Tensor index(index_ex);
  if (index.scalar_type() == at::ScalarType::Half) {
    index = NPUNativeFunctions::npu_dtype_cast(index, at::ScalarType::Float);
  }

  at::Tensor src(src_ex);
  if (src.scalar_type() != self.scalar_type()) {
    src = NPUNativeFunctions::npu_dtype_cast(src, self.scalar_type());
  }

  scatter_npu_nocheck(self, dim, index, src);
  
  if(self.scalar_type() != selfType){
    self = NPUNativeFunctions::npu_dtype_cast(self, selfType);
  } 

  return self;
}

at::Tensor& NPUNativeFunctions::scatter_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self, src, index},
      result,
      self);
  result = NPUNativeFunctions::copy_(result, self, false);
  scatter_npu_src_impl(result, dim, index, src);
  return result;
}

at::Tensor& NPUNativeFunctions::scatter_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value,
    at::Tensor& result) {
  at::Tensor srcTensor = scalar_to_tensor(value).to(at::ScalarType::Float);
  srcTensor = CalcuOpUtil::CopyTensorHostToDevice(srcTensor);
  at::Tensor srcTensor_broadcast = NPUNativeFunctions::npu_broadcast(srcTensor, array_to_small_vector(index.sizes()));
  OpPreparation::CheckOut(
      {self, index, srcTensor_broadcast},
      result,
      self);
  result = NPUNativeFunctions::copy_(result, self, false);
  scatter_npu_src_impl(result, dim, index, srcTensor_broadcast);
  return result;
}

at::Tensor& NPUNativeFunctions::scatter_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    c10::string_view reduce,
    at::Tensor& result) {
    TORCH_CHECK(false, "scatter.reduce_out is not supported.");
    return result;
}

at::Tensor& NPUNativeFunctions::scatter_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value,
    c10::string_view reduce,
    at::Tensor& result) {
    TORCH_CHECK(false, "scatter.value_reduce_out is not supported.");
    return result;
}

} // namespace native
} // namespace at_npu
