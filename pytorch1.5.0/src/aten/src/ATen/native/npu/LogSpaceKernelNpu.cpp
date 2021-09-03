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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& logspace_out_npu_nocheck(
    Tensor& result,
    Scalar start,
    Scalar end,
    int64_t steps,
    double base) {

  if (steps < 0){
    TORCH_CHECK("please input steps > 0");
  }

  if (base <= 0) {
    printf("if base<=0, please input intenger start, end, (end-start)/(steps-1)");
  }

  Tensor inputs;
  if (result.scalar_type() == at::ScalarType::Half) {
    inputs = at::arange(0, steps, at::device(at::kNPU)).to(at::kHalf);
  } else if (result.scalar_type() == at::ScalarType::Float) {
    inputs = at::arange(0, steps, at::device(at::kNPU).dtype(at::kFloat));
  }

  int64_t dtype = 0;
  if (result.scalar_type() == at::ScalarType::Half) {
    dtype = 0;
  } else if (result.scalar_type() == at::ScalarType::Float) {
    dtype = 1;
  } else {
    TORCH_CHECK("only support float32 and float16");
  }

  OpCommand cmd;
  cmd.Name("LogSpaceD")
      .Input(inputs)
      .Output(result)
      .Attr("start", start)
      .Attr("end", end)
      .Attr("steps", steps)
      .Attr("base", static_cast<float>(base))
      .Attr("dtype", dtype)
      .Run();

  return result;
}

Tensor& logspace_out_npu(
    Tensor& result,
    Scalar start,
    Scalar end,
    int64_t steps,
    double base) {
  OpPreparation::CheckOut(
      { },
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      {steps});

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({},{result})
    .Func([&start, &end, &steps, &base](Tensor& result)
    {logspace_out_npu_nocheck(result, start, end, steps, base);})
    .Call(result);

  return result;
}

Tensor logspace_npu(
    Scalar start,
    Scalar end,
    int64_t steps,
    double base,
    const TensorOptions& options) {

  Tensor result = OpPreparation::ApplyTensorWithFormat({steps}, options, ACL_FORMAT_ND);

  return logspace_out_npu_nocheck(result, start, end, steps, base);
}

}
}
