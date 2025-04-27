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

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

int64_t get_npu_format(const at::Tensor& self)
{
    return NPUNativeFunctions::get_npu_format(self);
}

std::vector<int64_t> get_npu_storage_sizes(const at::Tensor& self)
{
    torch_npu::utils::torch_check_npu(self);
    auto storage_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
    std::vector<int64_t> vec_storage_sizes(storage_sizes.begin(), storage_sizes.end());
    return vec_storage_sizes;
}

at::Tensor npu_format_cast(const at::Tensor& self, int64_t acl_format)
{
    return NPUNativeFunctions::npu_format_cast(self, acl_format);
}

at::Tensor empty_with_format(c10::IntArrayRef sizes, const c10::TensorOptions& options,
                             int64_t format, bool keep_format)
{
    return OpPreparation::ApplyTensorWithFormat(sizes, options, format, keep_format);
}

at::Tensor empty_with_swapped_memory(c10::IntArrayRef size,
                                     c10::optional<at::ScalarType> dtype_opt,
                                     c10::optional<c10::Device> device_opt)
{
    return NPUNativeFunctions::empty_with_swapped_memory(size, dtype_opt, device_opt);
}

} // namespace native
} // namespace at_npu
