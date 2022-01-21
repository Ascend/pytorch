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

#include <ATen/record_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {
    at::Tensor &NPUNativeFunctions::npu_transpose_out(
        const at::Tensor &self,
        at::IntArrayRef perm,
        at::Tensor &result)
    {
      c10::SmallVector<int64_t, N> permVec = array_to_small_vector(perm);
      OpCommand cmd;
      cmd.Name("Transpose")
          .Input(self)
          .Input(perm)
          .Output(result)
          .Run();
      return result;
    }

    at::Tensor NPUNativeFunctions::npu_transpose(const at::Tensor &self, at::IntArrayRef perm)
    {
      auto outputSize = transpose_npu_output_size(self, perm);
      at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
      NPUNativeFunctions::npu_transpose_out(self, perm, result);

      return result;
    }

    c10::SmallVector<NPUTensorDesc, N> transpose_to_contiguous_npu_input(
        const c10::SmallVector<at::Tensor, N> &src)
    {

      c10::SmallVector<NPUTensorDesc, N> inputs;
      for (int i = 0; i < src.size(); i++)
      {
        inputs.emplace_back(
            NPUTensorDesc(src[i]));

        if (src[i].dim() == 0)
        {
          inputs[i].tensorDescType = NPUTensorDesc::TensorDescType::TENSOR_SCALAR;
        }
      }
      return inputs;
    }

    c10::SmallVector<NPUTensorDesc, N> transpose_to_contiguous_npu_output(
        const c10::SmallVector<at::Tensor, N> &result)
    {
      return CalcuOpUtil::create_npu_output_tensor_desc(result);
    }

    at::Tensor NPUNativeFunctions::npu_transpose_to_contiguous(const at::Tensor &self)
    {
      RECORD_FUNCTION("transpose_to_contiguous", vector<c10::IValue>({self}));
      int64_t self_format = CalcuOpUtil::get_tensor_npu_format(self);
      at::Tensor result = at::empty_with_format(self.sizes(), self.options(), self_format);

      // obtain the transpose axises
      at::IntArrayRef dim;
      if ((self.dim() == 2) && (self.stride(self.dim() - 2) == 1))
      {
        dim = at::IntArrayRef({1, 0});
      }
      else if ((self.dim() == 3) && (self.stride(self.dim() - 2) == 1))
      {
        dim = at::IntArrayRef({0, 2, 1});
      }
      else if ((self.dim() == 3) && (self.stride(0) <= self.stride(1)))
      {
        dim = at::IntArrayRef({1, 0, 2});
      }
      // constructs the input and output NPUTensorDesc
      auto inputs = transpose_to_contiguous_npu_input({self});
      auto outputs = transpose_to_contiguous_npu_output({result});

      // constructs the attr of the NPUAttrDesc
      NPUAttrDesc npuAttrTranspose = NPUAttrDesc("perm", dim);
      c10::SmallVector<NPUAttrDesc, N> attrs = {npuAttrTranspose};

      CalcuOpUtil::execute_npu_operate("TransposeD", inputs, outputs, attrs);
      return result;
    }

  } // namespace native
} // namespace at_npu