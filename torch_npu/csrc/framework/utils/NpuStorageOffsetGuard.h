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

#ifndef __NPU_STORAGE_GUARD__
#define __NPU_STORAGE_GUARD__
#include <ATen/ATen.h>
#include <stdint.h>

#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace at_npu {
namespace native {
class NpuStorageOffsetGuard
{
public:
    NpuStorageOffsetGuard() = delete;
    NpuStorageOffsetGuard(const NpuStorageOffsetGuard &guard) = delete;
    NpuStorageOffsetGuard &operator= (const NpuStorageOffsetGuard &guard) = delete;

    NpuStorageOffsetGuard(NpuStorageOffsetGuard &&guard) = delete;
    NpuStorageOffsetGuard &operator= (NpuStorageOffsetGuard &&guard) = delete;

    explicit NpuStorageOffsetGuard(at::Tensor &tensor) noexcept : guard_(tensor) {
        SetTensorStorageOffset();
    }
    ~NpuStorageOffsetGuard() noexcept {
        RecoverTensorStorageOffset();
    }

private:
    void SetTensorStorageOffset() {
        origin_allow_tensor_metadata_change_ = guard_.unsafeGetTensorImpl()->allow_tensor_metadata_change();
        origin_storage_offset_ = guard_.storage_offset();

        guard_.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
        guard_.unsafeGetTensorImpl()->set_storage_offset(0);
    }
    void RecoverTensorStorageOffset() {
        guard_.unsafeGetTensorImpl()->set_storage_offset(origin_storage_offset_);
        guard_.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(origin_allow_tensor_metadata_change_);
    }
    int64_t origin_storage_offset_ = 0;
    bool origin_allow_tensor_metadata_change_ = true;
    at::Tensor guard_;
};
}  // namespace native
}  // namespace at_npu

#endif // __NPU_STORAGE_GUARD__


