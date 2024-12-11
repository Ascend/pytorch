// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/core/npu/NPUMemorySwap.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"

namespace at_npu {
namespace native {

void memory_swap(void* dst, size_t dst_len, void* src, size_t src_len, int type)
{
    TORCH_CHECK(dst != nullptr, "dst is nullptr", PTA_ERROR(ErrCode::PTR));
    TORCH_CHECK(src != nullptr, "src is nullptr", PTA_ERROR(ErrCode::PTR));
    TORCH_CHECK(dst_len > 0, "expect dst_len > 0, but got: ", dst_len, PTA_ERROR(ErrCode::VALUE));
    TORCH_CHECK(src_len > 0, "expect src_len > 0, but got: ", src_len, PTA_ERROR(ErrCode::VALUE));
    TORCH_CHECK(type >= 0 && type <= 3, "expect type in [0, 3], but got: ", type, PTA_ERROR(ErrCode::VALUE));

    aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_HOST;
    switch (type) {
        case 0:
            kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_HOST;
            break;
        case 1:
            kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;
            break;
        case 2:
            kind = aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST;
            break;
        case 3:
            kind = aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE;
            break;
        default:
            break;
    }

    NPU_CHECK_ERROR(c10_npu::queue::LaunchAsyncCopyTask(dst, dst_len, src, src_len, kind));
}

} // namespace native
} // namespace at_npu
