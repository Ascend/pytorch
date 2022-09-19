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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
  namespace native
  {

    // 将要废弃，当前仅针对strid_add特殊算子使用该逻辑
    c10::SmallVector<int64_t, SIZE> deprecated_broadcast_ops_npu_output_size(
        c10::IntArrayRef shape1_,
        c10::IntArrayRef shape2_)
    {
      auto shape1 = array_to_small_vector(shape1_);
      auto shape2 = array_to_small_vector(shape2_);

      c10::SmallVector<int64_t, SIZE> output_shape;

      if (shape1.size() < shape2.size())
      {
        c10::SmallVector<int64_t, SIZE> shapeTemp = shape1;
        shape1 = shape2;
        shape2 = shapeTemp;
      }

      int64_t shape1_size = shape1.size();
      int64_t shape2_size = shape2.size();
      for (int i = 0; i < shape1_size - shape2_size; i++)
      {
        shape2.insert(shape2.begin(), 1);
      }

      for (int i = 0; i < shape1_size; i++)
      {
        if (shape1[i] == 0 || shape2[i] == 0)
        {
          output_shape.emplace_back((int64_t)0);
        }
        else
        {
          output_shape.emplace_back((shape1[i] > shape2[i]) ? shape1[i] : shape2[i]);
        }
      }

      return output_shape;
    }

    at::Tensor &stride_add_out_npu_nocheck(
        at::Tensor &result,
        const at::Tensor &self,
        const at::Tensor &other,
        c10::Scalar offset1,
        c10::Scalar offset2,
        c10::Scalar c1_len)
    {
      OpCommand cmd;
      cmd.Name("StrideAdd")
         .Input(self, "x1", ACL_FORMAT_NCHW)
         .Input(other, "x2", ACL_FORMAT_NCHW)
         .Output(result, "y", ACL_FORMAT_NCHW)
         .Attr("x1_c1_offset", (int64_t)offset1.toInt())
         .Attr("x2_c1_offset", (int64_t)offset2.toInt())
         .Attr("c1_len", (int64_t)c1_len.toInt())
         .Run();
      return result;
    }

    at::Tensor NPUNativeFunctions::npu_stride_add(
        const at::Tensor &self,
        const at::Tensor &other,
        c10::Scalar offset1,
        c10::Scalar offset2,
        c10::Scalar c1_len)
    {
      // calculate the output size
      auto outputSize = deprecated_broadcast_ops_npu_output_size(self.sizes(), other.sizes());
      outputSize[1] = c1_len.toInt() * 16;

      // construct the output at::Tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

      // calculate the output result of the NPU
      stride_add_out_npu_nocheck(result, self, other, offset1, offset2, c1_len);

      return result;
    }

  } // namespace native
} // namespace at
