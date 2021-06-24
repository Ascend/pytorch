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

Tensor erfinv_npu(const Tensor &self) {
  auto output_size = self.sizes();
  auto output_t = at::empty_with_format(output_size, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  auto inputs = CalcuOpUtil::create_npu_input_tensor_desc({self});
  auto outputs = CalcuOpUtil::create_npu_output_tensor_desc({output_t});
  SmallVector<NPUAttrDesc, N> attrs = {};
  CalcuOpUtil::execute_npu_operate("Erfinv", inputs, outputs, attrs);
  return output_t;
}

SmallVector<NPUTensorDesc, N> erfinv_npu_input(const SmallVector<Tensor, N>& inputTensor)
{
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> erfinv_npu_output(const SmallVector<Tensor, N>& outputTensor)
{
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> erfinv_npu_attr(const Tensor& self)
{
  SmallVector<NPUAttrDesc, N> attrs = { };
  return attrs;
}

Tensor& erfinv_out_npu(Tensor& result, const Tensor& self)
{
  //constructs the input and output NPUTensorDesc
  auto inputs = erfinv_npu_input({self});
  auto outputs = erfinv_npu_output({result});
  //constructs the attr of the NPUAttrDesc
  auto attrs = erfinv_npu_attr(self);
  //executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Erfinv", inputs, outputs, attrs);
  return result;
}

Tensor& erfinv_npu_(Tensor& self)
{
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = erfinv_out_npu(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    erfinv_out_npu(self, self);
  }
  return self;
}

}  // namespace native
}  // namespace at