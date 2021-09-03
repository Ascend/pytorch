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
#include <torch/script.h>

namespace at {
namespace native {
using namespace at::native::npu;

namespace{
Tensor& cast_nocheck(Tensor& result, const Tensor& self) {
  int64_t dstDataType = CalcuOpUtil::convert_to_acl_data_type(result.scalar_type());
  OpCommand cmd;
  cmd.Name("Cast")
      .Input(self)
      .Output(result)
      .Attr("dst_type", dstDataType)
      .Run();
  return result;
}
}//namespace

Tensor dtype_cast_npu(const Tensor& self, ScalarType dtype) {
  if (self.dtype() == dtype) {
    return self.clone();
  }
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result =
      OpPreparation::ApplyTensor(outputSize, self.options().dtype(dtype), self);

  // calculate the output result of the NPU
  cast_nocheck(result, self);

  return result;
}

Tensor& dtype_cast_npu_(Tensor& self, const Tensor& src) {
  if (self.dtype() == src.dtype()) {
    return self;
  }

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = cast_nocheck(contiguousSelf, src);
    NpuUtils::format_fresh_view(self, result);
  } else {
    cast_nocheck(self, src);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_dtype_cast", TORCH_FN(dtype_cast_npu));
  m.impl("npu_dtype_cast_", TORCH_FN(dtype_cast_npu_));
}

Tensor& npu_dtype_cast_(Tensor& self, const Tensor& src) {
  return dtype_cast_npu_(self, src);;
}

Tensor npu_dtype_cast(const Tensor& self, ScalarType dtype) {
  return dtype_cast_npu(self, dtype);
}

} // namespace native
} // namespace at
