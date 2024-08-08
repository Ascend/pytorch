// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

#include "third_party/op-plugin/op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUOpApiNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::clone(const at::Tensor &src, c10::optional<c10::MemoryFormat> format)
{
    DO_COMPATIBILITY(aclnnInplaceCopy, NPUNativeFunctions::clone(src, format));
    auto baseSelf = OpPreparation::apply_tensor_without_format(src);
    baseSelf.copy_(src);
    at::namedinference::propagate_names(baseSelf, src);
    return baseSelf;
}

}  // namespace native
}  // namespace at_npu

