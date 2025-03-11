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

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUPeerToPeerAccess.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/aten/NPUOpApiNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "third_party/op-plugin/op_plugin/utils/op_api_common.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif


namespace at_npu {
namespace native {

// the format of dst and src is base format now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_between_host_and_device_opapi(at::Tensor& dst, const at::Tensor& src, aclrtMemcpyKind kind,
                                        bool non_blocking)
{
    int64_t nbytes = dst.numel() * dst.element_size();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();

    if (non_blocking) {
        auto ret = CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(dst, nbytes, src, nbytes, kind);
        NPU_CHECK_ERROR(ret);
        ASCEND_LOGD("non_blocking copy without StreamSynchronize.");
        void* ptr = torch_npu::utils::is_npu(dst) ? get_base_data_ptr(src) : get_base_data_ptr(dst);
        NPU_CHECK_ERROR(CachingHostAllocator_recordEvent(ptr, stream), "aclrtSynchronizeStreamWithTimeout");
    } else {
        aclError error = aclrtSynchronizeStream(stream);
        auto ret = CalcuOpUtil::AclrtMemcpyWithModeSwitch(
            std::make_pair(dst.storage().unsafeGetStorageImpl(), dst.storage_offset() * dst.itemsize()), nbytes,
            std::make_pair(src.storage().unsafeGetStorageImpl(), src.storage_offset() * src.itemsize()), nbytes, kind);
        NPU_CHECK_ERROR(ret, "aclrtMemcpy");
        if (error != ACL_ERROR_NONE) {
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(error);
            C10_NPU_SHOW_ERR_MSG();
            if (c10_npu::option::OptionsManager::IsResumeModeEnable()) {
                TORCH_NPU_WARN("ACL stream synchronize failed, error code:", error,
                               ". But in checkpoint-resume mode will not throw exceptions.");
            } else {
                AT_ERROR("ACL stream synchronize failed, error code:", error);
            }
        } else {
#ifndef BUILD_LIBTORCH
            const c10_npu::impl::PyCallbackTrigger* trigger = c10_npu::impl::NPUTrace::getTrace();
            if (C10_UNLIKELY(trigger)) {
                trigger->traceNpuStreamSynchronization(
                    reinterpret_cast<uintptr_t>(stream.stream(false)));
            }
#endif
        }
    }
}

// the format of dst and src is baseformat now, copy d2d
void copy_d2d_baseformat_opapi(at::Tensor& dst, const at::Tensor& src, bool non_blocking)
{
    if (dst.device().index() != src.device().index()) {
        return copy_d2d(dst, src, non_blocking);
    }

    c10_npu::NPUGuard guard(src.device());
    c10::SmallVector<at::Tensor, N> inputs = {src};
    c10::SmallVector<at::Tensor, N> outputs = {dst};
    CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

    EXEC_NPU_CMD(aclnnInplaceCopy, dst, src);
}

// the format of dst and src is base format now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_h2d_baseformat_dtype_contigous_opapi(at::Tensor& dst, const at::Tensor& src, bool non_blocking)
{
    c10_npu::OptionalNPUGuard device_guard;
    device_guard.set_device(dst.device());
    aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;
    copy_between_host_and_device_opapi(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_d2h_baseformat_dtype_contigous_opapi(at::Tensor& dst, const at::Tensor& src, bool non_blocking)
{
    c10_npu::OptionalNPUGuard device_guard;
    device_guard.set_device(src.device());
    aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST;
    copy_between_host_and_device_opapi(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
void copy_h2d_baseformat_opapi(at::Tensor& dst, const at::Tensor& src, bool non_blocking,
                               bool dst_must_be_contiguous = false)
{
    c10_npu::NPUGuard guard(dst.device());
    bool same_type = (src.dtype() == dst.dtype());
    bool same_size = (src.sizes() == dst.sizes());
    bool dst_is_contiguous = dst_must_be_contiguous ? true : dst.is_contiguous();
    if (same_type && dst_is_contiguous && src.is_contiguous() && same_size) {
        copy_h2d_baseformat_dtype_contigous_opapi(dst, src, non_blocking);
        return;
    }

    at::Tensor dst_contig = dst_is_contiguous ? dst : at::empty_like(dst);
    at::Tensor src_contig;
    if (!same_type) {
        src_contig = src.to(dst.dtype()).expand_as(dst).contiguous();
    } else {
        src_contig = src.expand_as(dst).contiguous();
    }
    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()), OPS_ERROR(ErrCode::VALUE));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type(), OPS_ERROR(ErrCode::VALUE));
    copy_h2d_baseformat_dtype_contigous_opapi(dst_contig, src_contig, non_blocking);
    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
        TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device(), OPS_ERROR(ErrCode::VALUE));
        copy_d2d_baseformat_opapi(dst, dst_contig, non_blocking);
    }
}

// the format of dst and src is baseformat now
void copy_d2h_baseformat_opapi(at::Tensor& dst, const at::Tensor& src, bool non_blocking)
{
    c10_npu::NPUGuard guard(src.device());
    bool same_type = (src.dtype() == dst.dtype());
    bool same_size = (src.sizes() == dst.sizes());
    bool dst_is_contiguous = dst.is_contiguous();
    if (same_type && dst_is_contiguous && src.is_contiguous() && same_size) {
        copy_d2h_baseformat_dtype_contigous_opapi(dst, src, non_blocking);
        return;
    }
    at::Tensor dst_contig = (dst_is_contiguous && same_type) ? dst : at::empty_like(dst, src.dtype());
    at::Tensor src_contig = src.expand_as(dst).contiguous();
    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()), OPS_ERROR(ErrCode::VALUE));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type(), OPS_ERROR(ErrCode::VALUE));
    copy_d2h_baseformat_dtype_contigous_opapi(dst_contig, src_contig, non_blocking);
    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
        TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device(), OPS_ERROR(ErrCode::VALUE));
        dst.copy_(dst_contig, non_blocking); // h2h, use cpu copy
    }
}

at::Tensor& NPUNativeOpApiFunctions::copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking)
{
    DO_COMPATIBILITY(aclnnInplaceCopy, NPUNativeFunctions::copy_(self, src, non_blocking));
    if (self.numel() == 0) {
        return self;
    }

    auto maybe_outnames = at::namedinference::compute_broadcast_outnames(self, src);

    if (torch_npu::utils::is_npu(self)) {
        if (torch_npu::utils::is_npu(src)) {
            copy_d2d_baseformat_opapi(self, src, non_blocking);
        } else {
            copy_h2d_baseformat_opapi(self, src, non_blocking);
        }
        if (src.is_complex() && src.is_conj()) {
            auto real_tensor = at::real(self);
            auto imag_tensor = at::imag(self).neg();
            EXEC_NPU_CMD(aclnnComplex, real_tensor, imag_tensor, self);
        }
        if (src.is_neg()) {
            self.neg_();
        }
    } else {
        if (torch_npu::utils::is_npu(src)) {
            copy_d2h_baseformat_opapi(self, src, non_blocking);
            if (src.is_complex() && src.is_conj()) {
                self.conj_physical_();
            }
        }
    }
    at::namedinference::propagate_names_if_nonempty(self, maybe_outnames);
    return self;
}

} // namespace native
} // namespace at_npu
