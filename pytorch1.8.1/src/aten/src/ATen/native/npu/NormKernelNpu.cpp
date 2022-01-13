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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "climits"

namespace at {
namespace native {
using namespace at::native::npu;

int64_t calculate_p(optional<Scalar> p) {
  if (p.has_value()) {
    float val = CalcuOpUtil::get_scalar_float_value(p.value());
    if (val == INFINITY) {
      return static_cast<int64_t>(INT_MAX); // p = inf
    } else if (val == -INFINITY) {
      return static_cast<int64_t>(INT_MIN); // p = -inf
    } else {
      return static_cast<int64_t>(val);
    }
  } else {
    return static_cast<int64_t>(2); // default: p = 2
  }
}

// norm.dtype_out
Tensor& norm_out_npu_nocheck(
    Tensor& out,
    const Tensor& self,
    optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim,
    ScalarType dtype) {
  // calculate the output size
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithSizes(outputSize, self.options());

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
  if (result.scalar_type() != dtype) {
    result = result.to(dtype);
  }

  // until now, can not support resize shape of out correctly,
  // so the shape of out must be equal to outputSize
  out = out.copy_(result);

  return out;
}

// norm.out
Tensor& norm_out_npu(
    const Tensor& self,
    optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim,
    Tensor& out) {
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  OpPreparation::CheckOut(
      {self},
      out,
      ACL_FORMAT_ND,
      self.scalar_type(),
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {out})
    .Func([&self, &p, &dim, &keepdim](Tensor& out){norm_out_npu_nocheck(out, self, p, dim, keepdim, self.scalar_type());})
    .Call(out);  
}

// norm.dtype_out
Tensor& norm_dtype_out_npu(
    const Tensor& self,
    optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim,
    ScalarType dtype,
    Tensor& out) {
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  OpPreparation::CheckOut(
      {self},
      out,
      ACL_FORMAT_ND,
      self.scalar_type(),
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {out})
    .Func([&self, &p, &dim, &keepdim, &dtype](Tensor& out){norm_out_npu_nocheck(out, self, p, dim, keepdim, dtype);})
    .Call(out);
}
// norm.ScalarOpt_dim_dtype
Tensor norm_scalaropt_dim_dtype_npu(
    const Tensor& self,
    optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim,
    ScalarType dtype) {
  auto outputSize = reduce_ops_npu_output_size(self, dim, keepdim);
  Tensor out = OpPreparation::ApplyTensorWithSizes(outputSize, self.options().dtype(dtype));
  norm_out_npu_nocheck(out, self, p, dim, keepdim, dtype);

  return out;
}

// norm.ScalarOpt_dtype
Tensor norm_scalaropt_dtype_npu(
    const Tensor& self,
    optional<Scalar> p,
    ScalarType dtype) {
  return norm_scalaropt_dim_dtype_npu(self, p, {}, false, dtype);
}

// norm.Scalar
Tensor norm_scalar_npu(
    const Tensor& self,
    Scalar p) {
  return norm_scalaropt_dim_dtype_npu(self, p, {}, false, self.scalar_type());
}

// norm.ScalarOpt_dim
Tensor norm_scalaropt_dim_npu(
    const Tensor& self,
    optional<Scalar> p,
    IntArrayRef dim,
    bool keepdim) {
  return norm_scalaropt_dim_dtype_npu(self, p, dim, keepdim, self.scalar_type());
}

// norm.names_ScalarOpt_dim_dtype
Tensor norm_name_scalaropt_dim_dtype_npu(
    const Tensor& self,
    optional<Scalar> p,
    DimnameList dim,
    bool keepdim,
    ScalarType dtype) {
  return norm_scalaropt_dim_dtype_npu(self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}

// norm.names_ScalarOpt_dim
Tensor norm_names_scalaropt_dim_npu(
    const Tensor& self,
    optional<Scalar> p,
    DimnameList dim,
    bool keepdim) {
  return norm_scalaropt_dim_npu(self, p, dimnames_to_positions(self, dim), keepdim);
}

// norm.names_dtype_out
Tensor& norm_names_dtype_out_npu(
    const Tensor& self,
    optional<Scalar> p,
    DimnameList dim,
    bool keepdim, 
    ScalarType dtype, 
    Tensor& out) {
  return norm_dtype_out_npu(self, p, dimnames_to_positions(self, dim), keepdim, dtype, out);
}

// norm.names_out
Tensor& norm_names_out_npu(
    const Tensor& self,
    optional<Scalar> p,
    DimnameList dim,
    bool keepdim,
    Tensor& out) {
  return norm_out_npu(self, p, dimnames_to_positions(self, dim), keepdim, out);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("norm.ScalarOpt_dtype", TORCH_FN(norm_scalaropt_dtype_npu));
  m.impl("norm.Scalar", TORCH_FN(norm_scalar_npu));
  m.impl("norm.ScalarOpt_dim_dtype", TORCH_FN(norm_scalaropt_dim_dtype_npu));
  m.impl("norm.ScalarOpt_dim", TORCH_FN(norm_scalaropt_dim_npu));
  m.impl("norm.dtype_out", TORCH_FN(norm_dtype_out_npu));
  m.impl("norm.out", TORCH_FN(norm_out_npu));
  m.impl("norm.names_ScalarOpt_dim_dtype", TORCH_FN(norm_name_scalaropt_dim_dtype_npu));
  m.impl("norm.names_ScalarOpt_dim", TORCH_FN(norm_names_scalaropt_dim_npu));
  m.impl("norm.names_dtype_out", TORCH_FN(norm_names_dtype_out_npu));
  m.impl("norm.names_out", TORCH_FN(norm_names_out_npu));
}

} // namespace native
} // namespace at
