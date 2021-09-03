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

#include <ATen/native/npu/utils/CalcuOpUtil.h>
#include <ATen/native/npu/frame/FormatHelper.h>
#include <ATen/npu/Exceptions.h>
#include <c10/npu/NPUStream.h>
#include <third_party/acl/inc/acl/acl.h>
#include "ATen/native/npu/utils/TensorInterface.h"
#include <torch/script.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& copy_memory_npu_(Tensor& self, const Tensor& src, bool non_blocking) {
  AT_ASSERT(src.is_npu(), "copy_memory_ only support npu tensor");
  AT_ASSERT(
      src.dtype() == self.dtype(),
      "input tensors of copy_memory_ should have same dtype");
  // AT_ASSERT(
  //     src.is_contiguous() && self.is_contiguous(),
  //     "input tensors of copy_memory_ should be contiguous");
  AT_ASSERT(
      src.device().index() == self.device().index(),
      "input tensors of copy_memory_ should have same device index");
  auto dst_desc = self.storage().unsafeGetStorageImpl()->npu_desc_;
  auto src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;

  int dst_size = 0;
  int src_size = 0;

  if (FormatHelper::IsPadded(&self)) {
    AT_ASSERT(self.storage_offset() == 0);
    dst_size = prod_intlist(dst_desc.storage_sizes_);
  } else {
    auto dst_element = prod_intlist(self.sizes());
    auto dst_storage = prod_intlist(dst_desc.storage_sizes_);
    dst_size = (dst_element > dst_storage) ? dst_storage : dst_element;
  }

  if (FormatHelper::IsPadded(&src)) {
    AT_ASSERT(src.storage_offset() == 0);
    src_size = prod_intlist(src_desc.storage_sizes_);
  } else {
    auto src_element = prod_intlist(src.sizes());
    auto src_storage = prod_intlist(src_desc.storage_sizes_);
    src_size = (src_element > src_storage) ? src_storage : src_element;
  }

  // TODO(Ascend): Temporary plan. Wait for ND plan to verify.
  c10::npu::NPUStream stream = c10::npu::getCurrentNPUStream();
  // Designed for the gather of tensors, ignoring npu_format_ and
  // copying continuous memory between npu tensors.
  AT_NPU_CHECK(aclrtMemcpyAsync(
      self.data_ptr(),
      dst_size * self.itemsize(),
      src.data_ptr(),
      dst_size * self.itemsize(),
      ACL_MEMCPY_DEVICE_TO_DEVICE,
      stream));
    if (!non_blocking) {
      AT_NPU_CHECK(aclrtSynchronizeStream(stream));
    }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("copy_memory_", TORCH_FN(copy_memory_npu_));
}

Tensor& copy_memory_(Tensor& self, const Tensor& src, bool non_blocking) {
  return copy_memory_npu_(self, src, non_blocking);
}

} // namespace native
} // namespace at