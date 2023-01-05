// Copyright (c) 2022 Huawei Technologies Co., Ltd
// Copyright (c) 2022, Facebook CORPORATION.
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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& searchsorted_out_npu_nocheck(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    at::Tensor& result) {
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  OpCommand cmd;
  cmd.Name("SearchSorted")
     .Input(sorted_sequence)
     .Input(self)
     .Attr("dtype", scalar_type)
     .Attr("right", right)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::searchsorted_out(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    at::Tensor& result) {
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  OpPreparation::CheckOut(
      {sorted_sequence, self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      scalar_type,
      self.sizes());
  searchsorted_out_npu_nocheck(sorted_sequence, self, out_int32, right, result);
  return result;
}

at::Tensor NPUNativeFunctions::searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right) {
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  at::Tensor result = OpPreparation::ApplyTensor(self.sizes(), self.options().dtype(scalar_type), self);
  searchsorted_out_npu_nocheck(sorted_sequence, self, out_int32, right, result);
  return result;
}

at::Tensor NPUNativeFunctions::searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right) {
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  at::Tensor selfOp = CalcuOpUtil::CopyScalarToDevice(self, sorted_sequence.scalar_type());
  selfOp = selfOp.unsqueeze(0);
  at::Tensor result = OpPreparation::ApplyTensor({}, sorted_sequence.options().dtype(scalar_type), sorted_sequence);
  searchsorted_out_npu_nocheck(sorted_sequence, selfOp, out_int32, right, result);
  return result;
}
} // namespace native
} // namespace at_npu
