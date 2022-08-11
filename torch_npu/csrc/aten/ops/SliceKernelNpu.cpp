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
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    at::Tensor &XLANativeFunctions::npu_slice_out(
        const at::Tensor &self,
        c10::IntArrayRef offsets,
        c10::IntArrayRef size,
        at::Tensor &result)
    {

      c10::SmallVector<int64_t, N> offsetVec = array_to_small_vector(offsets);
      c10::SmallVector<int64_t, N> sizeVec = array_to_small_vector(size);
      OpCommand cmd;
      cmd.Name("Slice")
          .Input(self)
          .Input(offsetVec)
          .Input(sizeVec)
          .Output(result)
          .Run();
      return result;
    }

    at::Tensor XLANativeFunctions::npu_slice(const at::Tensor &self, c10::IntArrayRef offsets, c10::IntArrayRef size)
    {
      // calculate the output size
      c10::SmallVector<int64_t, SIZE> outputSize =
          CalcuOpUtil::ConvertIntArrayRefToSmallVector(size);
      // construct the output at::Tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

      // calculate the output result of the NPU
      npu_slice_out(self, offsets, size, result);

      return result;
    }

  } // namespace native
} // namespace at