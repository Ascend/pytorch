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

SmallVector<NPUTensorDesc, N> eye_npu_output(const Tensor& result) {
  return CalcuOpUtil::create_npu_output_tensor_desc({result});
}

SmallVector<NPUAttrDesc, N> eye_npu_attr(int64_t n) {
  NPUAttrDesc npuDescScalarRow = NPUAttrDesc("num_rows", n);
  NPUAttrDesc npuDescScalarCol = NPUAttrDesc("num_columns", n);
  SmallVector<NPUAttrDesc, N> attrs = {npuDescScalarRow, npuDescScalarCol};

  return attrs;
}

SmallVector<NPUAttrDesc, N> eye_npu_attr(int64_t n, int64_t m) {
  NPUAttrDesc npuDescScalarRow = NPUAttrDesc("num_rows", n);
  NPUAttrDesc npuDescScalarCol = NPUAttrDesc("num_columns", m);
  SmallVector<NPUAttrDesc, N> attrs = {npuDescScalarRow, npuDescScalarCol};

  return attrs;
}

Tensor& eye_out_npu(Tensor& result, int64_t n) {
  return eye_out_npu(result, n, -1);
}

Tensor& eye_out_npu(Tensor& result, int64_t n, int64_t m) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);

  if (m < 0) {
    m = n;
  }

  result.resize_({n, m});

  SmallVector<NPUTensorDesc, N> inputs;

  // constracts the output NPUTensorDesc
  auto outputs = eye_npu_output(result);

  // constructs the attr of the NPUAttrDesc
  auto attrs = eye_npu_attr(n, m);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Eye", inputs, outputs, attrs);

  return result;
}

Tensor eye_npu(int64_t n, const TensorOptions& options) {

  // get the output size
  auto outputSize = SmallVector<int64_t, N>{n, n};

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(outputSize, options, ACL_FORMAT_ND);

  // constructs the attr of the NPUAttrDesc
  eye_out_npu(result, n);

  // calculate the output result of the NPU
  return result;
}

Tensor eye_npu(int64_t n, int64_t m, const TensorOptions& options) {
  // get the output size
  auto outputSize = SmallVector<int64_t, N>{n, m};

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(outputSize, options, ACL_FORMAT_ND);

  // constructs the attr of the NPUAttrDesc
  eye_out_npu(result, n, m);

  // calculate the output result of the NPU
  return result;
}

} // namespace native
} // namespace at