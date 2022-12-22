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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/utils/OverflowUtils.h"

namespace torch_npu {
namespace utils {

OverflowUtil::OverflowUtil(){
}

OverflowUtil::~OverflowUtil(){
}

bool OverflowUtil::CheckOverflowNpu(){
  auto options = at::TensorOptions(at_npu::key::NativeDeviceType).dtype(at::kFloat);
  at::Tensor tmp = at::empty({8}, options);
  auto floatStatus = at_npu::native::NPUNativeFunctions::npu_alloc_float_status(tmp);
  auto result = at_npu::native::NPUNativeFunctions::npu_get_float_status(floatStatus);
  if (floatStatus.cpu()[0].item().toInt() != 0){
    return true;
  }
  return false;
}

void OverflowUtil::ClearOverflowNpu(){
  auto options = at::TensorOptions(at_npu::key::NativeDeviceType).dtype(at::kFloat);
  at::Tensor tmp = at::empty({8}, options);
  auto floatStatus = at_npu::native::NPUNativeFunctions::npu_alloc_float_status(tmp);
  auto result = at_npu::native::NPUNativeFunctions::npu_clear_float_status(floatStatus);
  return;
}

}
}
