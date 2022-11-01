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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor linspace_assist(int64_t steps) {
  SmallVector<float, N> assist;
  assist.resize(steps);

  for (int64_t i = 0; i < steps; i++) {
    assist[i] = (float)(i);
  }
  Tensor assistTensor =
      from_blob(assist.data(), {steps}, dtype(ScalarType::Float));
  return CalcuOpUtil::copy_tensor_host_to_device(assistTensor);
}

Tensor& linspace_out_npu(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();
  r = r.npu_dtype_cast(at::kFloat);
  if(steps == 0){
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else {
    SmallVector<int64_t, N> sizeVec = {steps};
    OpCommand cmd;
    cmd.Name("LinSpace")
        .Input(start, ScalarType::Float)
        .Input(end, ScalarType::Float)
        .Input(sizeVec, ScalarType::Int)
        .Output(r)
        .Run();
  }

  if(r.dtype() != result.dtype()) {
    r = r.to(result.dtype());
  }

  return result.copy_(r);
}

Tensor linspace_npu(Scalar start, Scalar end, int64_t steps, const TensorOptions& options) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat({steps}, options, ACL_FORMAT_ND);
  Tensor resultCast = result.npu_dtype_cast(at::kFloat);

  // calculate the output result of the NPU
  linspace_out_npu(resultCast, start, end, steps);

  if(options.dtype() != resultCast.dtype()) {
    resultCast = resultCast.to(options.dtype());
  }

  return resultCast;
}

} // namespace native
} // namespace at
