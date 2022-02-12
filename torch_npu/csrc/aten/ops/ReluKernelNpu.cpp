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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    c10::SmallVector<NPUTensorDesc, N> relu_npu_input(
        const c10::SmallVector<at::Tensor, N> &inputTensor)
    {
      return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
    }

    c10::SmallVector<NPUTensorDesc, N> relu_npu_output(
        const c10::SmallVector<at::Tensor, N> &outputTensor)
    {
      return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
    }

    c10::SmallVector<NPUAttrDesc, N> relu_npu_attr(const at::Tensor &self)
    {
      c10::SmallVector<NPUAttrDesc, N> attrs = {};
      return attrs;
    }

    at::Tensor &relu_out_npu(const at::Tensor &self, at::Tensor &result)
    {
      // constructs the input and output NPUTensorDesc
      auto inputs = relu_npu_input({self});
      auto outputs = relu_npu_output({result});

      // constructs the attr of the NPUAttrDesc
      auto attrs = relu_npu_attr(self);

      // executing the NPU operator
      CalcuOpUtil::execute_npu_operate("Relu", inputs, outputs, attrs);

      return result;
    }

    at::Tensor NPUNativeFunctions::relu(const at::Tensor &self)
    {
      // return at::threshold(self, 0, 0);
      // calculate the output size
      auto outputSize = input_same_output_size(self);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

      // calculate the output result of the NPU
      relu_out_npu(self, result);
      return result;
    }

    at::Tensor &NPUNativeFunctions::relu_(at::Tensor &self)
    {
      // return at::threshold_(self, 0, 0);
      if (!NpuUtils::check_match(&self))
      {
        at::Tensor selfContiguous = NpuUtils::format_contiguous(self);
        at::Tensor result = relu_out_npu(selfContiguous, selfContiguous);
        NpuUtils::format_fresh_view(self, result);
      }
      else
      {
        relu_out_npu(self, self);
      }

      return self;
    }

  } // namespace native
} // namespace at_npu