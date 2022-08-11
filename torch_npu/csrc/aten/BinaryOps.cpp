// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/aten/XLANativeFunctions.h"


namespace at_npu {
namespace native {

at::Tensor true_divide_Tensor(const at::Tensor& self, const at::Tensor& divisor) {

  return self.div(divisor);
}

at::Tensor& true_divide_out_Tensor(const at::Tensor& self, const at::Tensor& divisor, at::Tensor& result) {
  
  return at::div_out(result, self, divisor);
}

at::Tensor& true_divide__Tensor(at::Tensor& self, const at::Tensor& divisor) {

  return self.div_(divisor);
}

TORCH_LIBRARY_IMPL(aten, CPU, m){
  m.impl("true_divide.Tensor", TORCH_FN(true_divide_Tensor));
  m.impl("true_divide.out", TORCH_FN(true_divide_out_Tensor));
  m.impl("true_divide_.Tensor", TORCH_FN(true_divide__Tensor));
}
}
}