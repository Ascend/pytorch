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

#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"

namespace at_npu {
namespace native {

void npu_fast_reshape_(at::Tensor& tensor) {
  /**
    [NOTE] For some reshape cases such as view, unsqueeze, squeeze, flatten,
    storages of them remain unchanged. So we can refresh reshape tensor's metadata
    to obtain matched tensor.
    */

  // restriction 1
  if (!tensor.is_contiguous()) {
    return;
  }
  // restriction 2
  if (!FormatHelper::IsBaseFormatType(tensor)) {
    return;
  }
  // restriction 3: reshape case without any numels change
  if ((tensor.numel() != StorageDescHelper::GetMemorySize(tensor)) ||
      StorageDescHelper::MetaDataAreMatch(&tensor)) {
    return;
  }

  // refresh matadata to input tensor
  StorageDescHelper::ReflushDescBySelf(tensor);
  auto base_format = InferFormat::GuessBaseFormat(tensor.sizes());
  tensor.npu_format_cast_(base_format);
}
} // namespace native
} // namespace at_npu
