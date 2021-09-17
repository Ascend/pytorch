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

#include <ATen/ATen.h>

#include <ATen/native/npu/contiguous/ContiguousOpt.h>
#include <ATen/native/npu/frame/FormatHelper.h>
#include <ATen/native/npu/frame/StorageDescHelper.h>
#include <ATen/native/npu/utils/OpTemplate.h>
#include <ATen/npu/Exceptions.h>
#include <THNPU/THNPUCachingHostAllocator.h>
#include <c10/npu/NPUGuard.h>
#include <c10/npu/OptionsManager.h>
#include "ATen/native/npu/common/FormatCastHelper.h"
#include "ATen/native/npu/common/InnerNpuNativeFunction.h"
#include <torch/library.h>

namespace at {
namespace native {
using namespace at::native::npu;

namespace {
// src :  host <-- device
//          |  copy src to dst on cpu
// dst :  host --> device
void copy_d2d_via_host(Tensor& self, const Tensor& src, bool same_type) {
  c10::npu::NPUStream copy_stream = c10::npu::getCurrentNPUStream();
  aclError error = aclrtSynchronizeStream(copy_stream);
  if (error != ACL_ERROR_NONE) {
    AT_ERROR("ACL stream synchronize failed.");
    return;
  }

  int64_t real_bytes =
      StorageDescHelper::GetValidMemorySize(src) * src.element_size();
  auto cpu_src = at::empty(
      real_bytes / src.element_size(), src.options().device(at::kCPU));
  cpu_src = cpu_src.as_strided(src.sizes(), src.strides());

  error = aclrtMemcpy(
      cpu_src.data_ptr(),
      real_bytes,
      src.data_ptr(),
      real_bytes,
      ACL_MEMCPY_DEVICE_TO_HOST);
  if (error != ACL_ERROR_NONE) {
    AT_ERROR("aclrtMemcpy device to cpu_src error.");
    return;
  }

  real_bytes =
      StorageDescHelper::GetValidMemorySize(self) * self.element_size();
  auto cpu_dst = at::empty(
      real_bytes / self.element_size(), self.options().device(at::kCPU));
  cpu_dst = cpu_dst.as_strided(self.sizes(), self.strides());

  if (!same_type) {
    cpu_src = cpu_src.to(cpu_dst.dtype());
  }

  // sometimes npu_dst just need part of cpu_dst's elements, so we do memory
  // copy from npu to cpu here, let npu_dst cover cpu_dst, to avoid unneeded
  // cpu_dst's elements cover npu_dst's original elements
  if ((!cpu_dst.is_contiguous()) && (self.defined())) {
    error = aclrtMemcpy(
        cpu_dst.data_ptr(),
        real_bytes,
        self.data_ptr(),
        real_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (error != ACL_ERROR_NONE) {
      AT_ERROR("ACL_Memcpy device to cpu_dst error.");
      return;
    }
  }

  cpu_dst.copy_(cpu_src);

  error = aclrtMemcpy(
      self.data_ptr(),
      real_bytes,
      cpu_dst.data_ptr(),
      real_bytes,
      ACL_MEMCPY_HOST_TO_DEVICE);
  if (error != ACL_ERROR_NONE) {
    AT_ERROR("aclrtMemcpy cpu_dst to device error.");
    return;
  }
  NPU_LOGD("Src or dst is not contiguous when do device to device copy.");
}

// NOTE: helper function of copy, the input parameter is not checked, The caller
// needs to ensure that the parameters are correct.

// the caller should ensure the tensor.is_npu == true
bool is_same_format(const Tensor& a, const Tensor& b) {
  bool isSameFormat = FormatHelper::GetFormat(a) == FormatHelper::GetFormat(b);
  if (!isSameFormat) {
    bool isBaseFormat =
        FormatHelper::IsBaseFormatType(a) && FormatHelper::IsBaseFormatType(b);
    return isBaseFormat;
  }
  return true;
}

bool try_to_optimize_copy_with_any_format(Tensor& self, const Tensor& src) {
  // Some Ops support inputs with 5HD/NZ format, Transdata is redundant
  // Record:
  // Op:Reshape; SliceD || Supportformat: 5HD/NZ
  return TransContiguous::ContiguousOptimizeWithAnyFormat(self, src);
}

// the dst and src are same format now
// the dst and src are base format now
// the dst and src maybe non-contiguous
void copy_d2d_last_method(
    Tensor& self,
    const Tensor& src,
    bool same_type,
    bool non_blocking) {
  // general copy method but Low performance
  if (c10::npu::OptionsManager::CheckPTcopy_Enable()) {
    RECORD_FUNCTION("d2dCopyWithPTCopy", std::vector<c10::IValue>({src}));
    copy_kernel_npu(self, src, non_blocking);
  } else {
    RECORD_FUNCTION(
        "d2dCopyWithStreamSynchronize", std::vector<c10::IValue>({src}));
    copy_d2d_via_host(self, src, same_type);
  }
}

// the dst and src are same format now
// the dst and src are base format now
// the dst and src maybe non-contiguous
void copy_d2d_dtype_baseformat(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  if (!self.is_contiguous()) {
    return copy_d2d_last_method(self, src, true, non_blocking);
  }

  if (!src.is_contiguous()) {
    // discontiguous
    if (TransContiguous::ContiguousOptimizeWithBaseFormat(self, src)) {
      return;
    }
  } else {
    int64_t numel = self.numel();
    if (numel == src.numel()) {
      RECORD_FUNCTION("d2dCopyAsync", std::vector<c10::IValue>({src}));
      NPU_LOGD("copy contiguous tensor inside device");
      return copy_d2d_by_memcpy(self, src, numel);
    }
  }
  copy_d2d_last_method(self, src, true, non_blocking);
}

// the dst and src are same format now
void copy_d2d_dtype_format(Tensor& self, const Tensor& src, bool non_blocking) {
  // Note: Src & Self have the same format.
  if (try_to_optimize_copy_with_any_format(self, src)) {
    return;
  }

  if (!FormatHelper::IsBaseFormatType(
          self)) { // TODO(ascend): 必须要非NCHW的才行？
    if (can_use_memcpy(self, src)) {
      RECORD_FUNCTION(
          "d2dCopyAsync with format", std::vector<c10::IValue>({src}));
      return copy_d2d_by_memcpy(self, src);
    }
  }

  if (!FormatHelper::IsBaseFormatType(self)) {
    Tensor src_4D = FormatCastHelper::ApplyBaseFormatTensorBy(src);
    Tensor dst_4D = FormatCastHelper::ApplyBaseFormatTensorBy(self);
    copy_d2d_dtype_baseformat(dst_4D, src_4D, non_blocking);
    self.npu_format_cast_(dst_4D);
    return;
  }
  copy_d2d_dtype_baseformat(self, src, non_blocking);
}

void copy_d2d(Tensor& self, const Tensor& src, bool non_blocking) {
  if (self.dtype() != src.dtype()) {
    self.npu_dtype_cast_(src); // npu_dtype_cast_ will call copy function.
    return;
  }
  copy_d2d_dtype(self, src, non_blocking);
}

// the format of dst and src is base format now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_between_host_and_device(
    Tensor& dst,
    const Tensor& src,
    aclrtMemcpyKind kind,
    bool non_blocking) {
  void* dst_ptr = dst.data_ptr();
  void* src_ptr = src.data_ptr();
  int64_t nbytes = dst.numel() * dst.element_size();
  c10::npu::NPUStream stream = c10::npu::getCurrentNPUStream();
  AT_NPU_CHECK(
      aclrtMemcpyAsync(dst_ptr, nbytes, src_ptr, nbytes, kind, stream));

  if (non_blocking) {
    NPU_LOGD("non_blocking copy without StreamSynchronize.");
    void* ptr = dst.is_npu() ? src_ptr : dst_ptr;
    AT_NPU_CHECK(THNPUCachingHostAllocator_recordEvent(ptr, stream));
  } else {
    aclError error = aclrtSynchronizeStream(stream);
    if (error != ACL_ERROR_NONE) {
      AT_ERROR("ACL stream synchronize failed, error code:", error);
    }
  }
}

// the format of dst and src is base format now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_h2d_baseformat_dtype_contigous(
    Tensor& dst,
    const Tensor& src,
    bool non_blocking) {
  c10::npu::OptionalNPUGuard device_guard;
  device_guard.set_device(dst.device());
  aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;
  copy_between_host_and_device(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_d2h_baseformat_dtype_contigous(
    Tensor& dst,
    const Tensor& src,
    bool non_blocking) {
  c10::npu::OptionalNPUGuard device_guard;
  device_guard.set_device(src.device());
  aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST;
  copy_between_host_and_device(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
void copy_h2d_baseformat(
    Tensor& dst,
    const Tensor& src,
    bool non_blocking,
    bool dst_must_be_contiguous = false) {
  bool same_type = (src.dtype() == dst.dtype());
  bool dst_is_contiguous = dst_must_be_contiguous ? true : dst.is_contiguous();
  if (same_type && dst_is_contiguous && src.is_contiguous()) {
    copy_h2d_baseformat_dtype_contigous(dst, src, non_blocking);
    return;
  }

  Tensor dst_contig = dst_is_contiguous ? dst : at::empty_like(dst);
  Tensor src_contig;
  if (!same_type) {
    src_contig = src.to(dst.dtype()).expand_as(dst).contiguous();
  } else {
    src_contig = src.expand_as(dst).contiguous();
  }
  // perform a same-dtype copy on contiguous tensors
  TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
  TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
  copy_h2d_baseformat_dtype_contigous(dst_contig, src_contig, non_blocking);
  // if necessary, copy back into dst
  if (!dst_contig.is_same(dst)) {
    TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
    copy_d2d_dtype(dst, dst_contig, non_blocking);
  }
}

// the format of dst and src is baseformat now
void copy_d2h_baseformat(Tensor& dst, const Tensor& src, bool non_blocking) {
  bool same_type = (src.dtype() == dst.dtype());
  bool dst_is_contiguous = dst.is_contiguous();
  if (same_type && dst_is_contiguous && src.is_contiguous()) {
    copy_d2h_baseformat_dtype_contigous(dst, src, non_blocking);
    return;
  }
  Tensor dst_contig =
      (dst_is_contiguous && same_type) ? dst : at::empty_like(dst, src.dtype());
  Tensor src_contig = src.expand_as(dst).contiguous();
  // perform a same-dtype copy on contiguous tensors
  TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
  TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
  copy_d2h_baseformat_dtype_contigous(dst_contig, src_contig, non_blocking);
  // if necessary, copy back into dst
  if (!dst_contig.is_same(dst)) {
    TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
    dst.copy_(dst_contig, non_blocking); // h2h, use cpu copy
  }
}

void copy_h2d(Tensor& self, const Tensor& src, bool non_blocking) {
  if (!FormatHelper::IsBaseFormatType(self)) {
    Tensor dst = OpPreparation::ApplyTensor(self);
    copy_h2d_baseformat(dst, src, non_blocking, true);
    self.npu_format_cast_(dst);
    return;
  }
  copy_h2d_baseformat(self, src, non_blocking);
}

void copy_d2h(Tensor& self, const Tensor& src, bool non_blocking) {
  if (!FormatHelper::IsBaseFormatType(src)) {
    Tensor src_4D = FormatCastHelper::ApplyBaseFormatTensorBy(src);
    copy_d2h_baseformat(self, src_4D, non_blocking);
    return;
  }
  copy_d2h_baseformat(self, src, non_blocking);
}
} // namespace

// the caller should guarantee that the format and dtype are same
bool can_use_memcpy(Tensor& dst, const Tensor& src) {
  if (StorageDescHelper::IsSameDesc(dst, src)) {
    // Make sure that the metadata are same.
    if (!dst.sizes().equals(src.sizes())) {
      return false;
    }
    if (!dst.strides().equals(src.strides())) {
      return false;
    }
    // Make sure that copy the whole memory.
    // we just need to compare one of them, because of the NpuStorageDesc
    // and metadata(sizes and stride) of src and dst are same.
    if (StorageDescHelper::GetValidMemorySize(src) != src.numel()) {
      return false;
    }
    if ((dst.storage_offset() != 0) || (src.storage_offset() != 0)) {
      return false;
    }
    return true;
  }
  return false;
}

// the dst and src are same dtype now
void copy_d2d_dtype(Tensor& self, const Tensor& src, bool non_blocking) {
  if (!is_same_format(self, src)) {
    Tensor src_4D = FormatCastHelper::ApplyBaseFormatTensorBy(src);
    // ApplyBaseFormatTensorBy is redundant for self tensor with base format.
    if (FormatHelper::IsBaseFormatType(self)) {
      copy_d2d_dtype_baseformat(self, src_4D, non_blocking);
      return;
    }
    Tensor dst_4D = FormatCastHelper::ApplyBaseFormatTensorBy(self);
    copy_d2d_dtype_baseformat(dst_4D, src_4D, non_blocking);
    self.npu_format_cast_(dst_4D);
    return;
  }
  copy_d2d_dtype_format(self, src, non_blocking);
}

bool try_to_optimize_copy_with_any_format(Tensor& self, const Tensor& src) {
  // Some Ops support inputs with 5HD/NZ format, Transdata is redundant
  // Record:
  // Op:Reshape; SliceD || Supportformat: 5HD/NZ
  return TransContiguous::ContiguousOptimizeWithAnyFormat(self, src);
}

Tensor& copy_npu_(Tensor& self, const Tensor& src, bool non_blocking) {
  if (self.numel() == 0) {
    return self;
  }
  // save tensor dim name
  optional<DimnameList> names = src.opt_names();
  if (names.has_value()) {
    internal_set_names_inplace(self, names);
  }

  if (self.is_npu()) {
    if (src.is_npu()) {
      copy_d2d(self, src, non_blocking);
    } else {
      copy_h2d(self, src, non_blocking);
    }
  } else {
    if (src.is_npu()) {
      copy_d2h(self, src, non_blocking);
    }
  }
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("copy_", TORCH_FN(copy_npu_));
}

} // namespace native
} // namespace at