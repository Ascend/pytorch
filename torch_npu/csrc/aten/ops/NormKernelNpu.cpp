// Copyright (c) 2020 Huawei Technologies Co., Ltd
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
#include "climits"

namespace at_npu
{
  namespace native
  {

    int64_t calculate_p(c10::optional<at::Scalar> p)
    {
      if (p.has_value())
      {
        float val = CalcuOpUtil::get_scalar_float_value(p.value());
        if (val == INFINITY)
        {
          return static_cast<int64_t>(INT_MAX); // p = inf
        }
        else if (val == -INFINITY)
        {
          return static_cast<int64_t>(INT_MIN); // p = -inf
        }
        else
        {
          return static_cast<int64_t>(val);
        }
      }
      else
      {
        return static_cast<int64_t>(2); // default: p = 2
      }
    }

    // norm.dtype_out
    at::Tensor &norm_out_npu_nocheck(
        at::Tensor &out,
        const at::Tensor &self,
        c10::optional<at::Scalar> p,
        at::IntArrayRef dim,
        bool keepdim,
        at::ScalarType dtype)
    {
      // calculate the output size
      auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithSizes(outputSize, self.options());

      auto pvalue = calculate_p(p);
      OpCommand cmd;
      cmd.Name("LpNorm")
          .Input(self)
          .Output(result)
          .Attr("p", pvalue)
          .Attr("axes", dim)
          .Attr("keepdim", keepdim)
          .Run();

      // trans dtype for output
      if (result.scalar_type() != dtype)
      {
        result = result.to(dtype);
      }

      // until now, can not support resize shape of out correctly,
      // so the shape of out must be equal to outputSize
      out = out.copy_(result);

      return out;
    }

    // norm.out
    at::Tensor &NPUNativeFunctions::norm_out(
        const at::Tensor &self,
        c10::optional<at::Scalar> p,
        at::IntArrayRef dim,
        bool keepdim,
        at::Tensor &out)
    {
      auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
      OpPreparation::CheckOut(
          {self},
          out,
          ACL_FORMAT_ND,
          self.scalar_type(),
          outputSize);

      OpPipeWithDefinedOut pipe;
      return pipe.CheckMemory({self}, {out})
          .Func([&self, &p, &dim, &keepdim](at::Tensor &out)
                { norm_out_npu_nocheck(out, self, p, dim, keepdim, self.scalar_type()); })
          .Call(out);
    }

    // norm.dtype_out
    at::Tensor &NPUNativeFunctions::norm_out(
        const at::Tensor &self,
        c10::optional<at::Scalar> p,
        at::IntArrayRef dim,
        bool keepdim,
        at::ScalarType dtype,
        at::Tensor &out)
    {
      auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
      OpPreparation::CheckOut(
          {self},
          out,
          ACL_FORMAT_ND,
          self.scalar_type(),
          outputSize);

      OpPipeWithDefinedOut pipe;
      return pipe.CheckMemory({self}, {out})
          .Func([&self, &p, &dim, &keepdim, &dtype](at::Tensor &out)
                { norm_out_npu_nocheck(out, self, p, dim, keepdim, dtype); })
          .Call(out);
    }
    // norm.ScalarOpt_dim_dtype
    at::Tensor NPUNativeFunctions::norm(
        const at::Tensor &self,
        c10::optional<at::Scalar> p,
        at::IntArrayRef dim,
        bool keepdim,
        at::ScalarType dtype)
    {
      auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
      at::Tensor out = OpPreparation::ApplyTensorWithSizes(outputSize, self.options().dtype(dtype));
      norm_out_npu_nocheck(out, self, p, dim, keepdim, dtype);

      return out;
    }

    // norm.ScalarOpt_dtype
    at::Tensor NPUNativeFunctions::norm(
        const at::Tensor &self,
        c10::optional<at::Scalar> p,
        at::ScalarType dtype)
    {
      return NPUNativeFunctions::norm(self, p, {}, false, dtype);
    }

    // norm.Scalar
    at::Tensor NPUNativeFunctions::norm(
        const at::Tensor &self,
        at::Scalar p)
    {
      return NPUNativeFunctions::norm(self, p, {}, false, self.scalar_type());
    }

    // norm.ScalarOpt_dim
    at::Tensor NPUNativeFunctions::norm(
        const at::Tensor &self,
        c10::optional<at::Scalar> p,
        at::IntArrayRef dim,
        bool keepdim)
    {
      return NPUNativeFunctions::norm(self, p, dim, keepdim, self.scalar_type());
    }

  } // namespace native
} // namespace at_npu
