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

at::Tensor& scatter_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  OpCommand cmd;
  cmd.Name("ScatterElements")
     .Input(self)
     .Input(index)
     .Input(src)
     .Output(result)
     .Attr("axis", dim)
     .Run();
  return result;
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
  scatter_out_npu_nocheck(result, self, dim, index, src);
  return result;
}

at::Tensor& NPUNativeFunctions::scatter_out(
  const at::Tensor& self,
  int64_t dim,
  const at::Tensor& index,
  const at::Scalar& value,
  at::Tensor& result) {
  OpPreparation::CheckOut(
      {self, index},
      result,
      self);
  at::Tensor srcTensor = scalar_to_tensor(value).to(at::ScalarType::Float);
  srcTensor = CalcuOpUtil::copy_tensor_host_to_device(srcTensor);
  at::Tensor srcTensor_broadcast = NPUNativeFunctions::npu_broadcast(srcTensor, array_to_small_vector(index.sizes()));
  scatter_out_npu_nocheck(result, self, dim, index, srcTensor_broadcast);
  return result;
}

} // namespace native
} // namespace at_npu
