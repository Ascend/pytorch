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
#include "c10/npu/OptionsManager.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& broadcast_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef size) {
  // executing the NPU operator
  int value = size.size();
  vector<int> tmp_vector = {};
  for (int i = 0; i < value; i++){
    tmp_vector.emplace_back(size[i]);
  }
  Tensor shapeCpuTensor = from_blob((void*)tmp_vector.data(), {value}, at::kInt);
  Tensor shapeNpuTensor = CalcuOpUtil::copy_tensor_host_to_device(shapeCpuTensor);
  OpCommand cmd;
  cmd.Name("BroadcastTo")
      .Input(self)
      .InputPair(shapeNpuTensor, shapeCpuTensor)
      .Output(result)
      .Run();
  return result;
}

Tensor broadcast_npu(const Tensor& self, IntArrayRef size) {
  Tensor input = self;
  if (self.dtype() == at::kBool) {
    input = input.to(at::kInt);
  }

  Tensor result = at::empty_with_format(
      size, 
      input.options(), 
      CalcuOpUtil::get_tensor_npu_format(self));

  broadcast_out_npu(result, input, size);

  if (self.dtype() == at::kBool) {
    result = result.to(at::kBool);
  }

  return result;
}
} // namespace native
} // namespace at