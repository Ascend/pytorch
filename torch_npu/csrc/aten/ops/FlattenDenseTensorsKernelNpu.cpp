// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::flatten_dense_tensors(at::TensorList tensors){
  static auto cast_back_to_ori_format = [](const at::Tensor& t) {
      return NPUNativeFunctions::npu_format_cast(t,
          torch_npu::NPUBridge::GetNpuStorageImpl(t)->npu_desc_.origin_format_);
  };
  static auto flatten = [](const at::Tensor& t) {
    return cast_back_to_ori_format(t).contiguous().view({-1});

  };
  if (tensors.size() == 1) {
    return flatten(tensors[0]);
  }
  return at::cat(c10::fmap(tensors, flatten));
}

}
}
