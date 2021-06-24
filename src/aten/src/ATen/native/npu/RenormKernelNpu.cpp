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

SmallVector<int64_t, SIZE> renorm_npu_output_size(
    const Tensor& self,
    int64_t dim) {
  SmallVector<int64_t, SIZE> outSize;
  for(int64_t i=0; i < self.dim(); i++) {
    if(i != dim) {
      outSize.emplace_back(1);
    } else {
      outSize.emplace_back(self.sizes()[i]);
    }
  }
  return outSize;
}

SmallVector<NPUTensorDesc, N> renorm_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> renorm_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> renorm_npu_attr(Scalar p, int64_t dim, Scalar maxnorm) {
  float p_value = CalcuOpUtil::get_scalar_float_value(p);
  float maxnorm_value = CalcuOpUtil::get_scalar_float_value(maxnorm);
  NPUAttrDesc npuAttrScalarP = NPUAttrDesc("p", p_value);
  NPUAttrDesc npuAttrScalarMaxnorm = NPUAttrDesc("maxnorm", maxnorm_value);
  NPUAttrDesc npuAttrDim = NPUAttrDesc("dim", dim);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrScalarP, npuAttrDim, npuAttrScalarMaxnorm};
  return attrs;
}

Tensor& renorm_compute(   
    Tensor& result, 
    const Tensor& self,
    Scalar p, 
    int64_t dim, 
    Scalar maxnorm) {
  // constructs the input and output NPUTensorDesc
  auto inputs = renorm_npu_input({self});
  auto outputs = renorm_npu_output({result});
  // constructs the attr of the NPUAttrDesc
  auto attrs = renorm_npu_attr(p, dim, maxnorm);
  
  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Renorm", inputs, outputs, attrs);
  return result;
}

Tensor& renorm_out_npu(   
    Tensor& result, 
    const Tensor& self,
    Scalar p, 
    int64_t dim, 
    Scalar maxnorm) {
  auto ori_type = self.scalar_type();
  if(ori_type != c10::ScalarType::Half && ori_type != c10::ScalarType::Float) {
    AT_ERROR("Renorm only support float16 or float32 type.");
  }
  if(result.scalar_type() != ori_type) {
    AT_ERROR("result's type must be equal to input's.");
  }
  dim = CalcuOpUtil::make_wrap_dim(dim, self.dim());
  auto outputSize = renorm_npu_output_size(self, dim);
  Tensor result_bak = at::empty_with_format(
      outputSize,
      self.options().dtype(at::kFloat),
      CalcuOpUtil::get_tensor_npu_format(self));
  if(ori_type == c10::ScalarType::Half) {
    Tensor self_no_name = self.rename(nullopt);
    Tensor result_no_name = result.rename(nullopt);
    self_no_name = self_no_name.npu_dtype_cast(c10::ScalarType::Float);
    result_no_name = result_no_name.npu_dtype_cast(c10::ScalarType::Float);
    renorm_compute(   
      result_bak, 
      self_no_name,
      p, 
      dim, 
      maxnorm
    );
    // broadcast and mul
    Tensor result_broadcast = at::npu_broadcast(result_bak, self.sizes());
    at::mul_out(result_no_name, result_broadcast, self_no_name);
    result.npu_dtype_cast_(result_no_name);
  } else {
    renorm_compute(   
      result_bak, 
      self,
      p, 
      dim, 
      maxnorm
    );
    // broadcast and mul
    Tensor result_broadcast = at::npu_broadcast(result_bak, self.sizes());
    at::mul_out(result, result_broadcast, self);
  }
  return result;
}

Tensor renorm_npu(const Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
  // calculate the output size  
  auto outputSize = input_same_output_size(self);

  Tensor result = at::empty_with_format(
      outputSize,
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));

  renorm_out_npu(result, self, p, dim, maxnorm);

  return result;
}


Tensor& renorm_npu_(Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = renorm_out_npu(contiguousSelf, contiguousSelf, p, dim, maxnorm);
    NpuUtils::format_fresh_view(self, result);
  } else {
    renorm_out_npu(self, self, p, dim, maxnorm);
  }

  return self;
}

} // namespace native
} // namespace at