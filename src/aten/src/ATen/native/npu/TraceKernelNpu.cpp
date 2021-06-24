// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
SmallVector<NPUTensorDesc, N> trace_npu_input(const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> trace_npu_output(const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> trace_npu_attr() {
  SmallVector<NPUAttrDesc, N> attrs = {};
  return attrs;
}

Tensor& trace_out_npu(Tensor& result, const Tensor& self) {
  auto inputs = trace_npu_input({self});
  auto outputs = trace_npu_output({result});
  auto attrs = trace_npu_attr();
  CalcuOpUtil::execute_npu_operate("Trace", inputs, outputs, attrs);
  return result;
}

Tensor trace_npu(const Tensor& self) {
  auto outputSize = trace_npu_output_size(self);
  Tensor result = at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  trace_out_npu(result, self);
  return result.reshape({});
}
}
}