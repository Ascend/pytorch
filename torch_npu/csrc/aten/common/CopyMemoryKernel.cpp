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
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "third_party/acl/inc/acl/acl.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::copy_memory_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  AT_ASSERT(at_npu::key::isDeviceTensor(src), "copy_memory_ only support npu tensor");
  AT_ASSERT(
      src.dtype() == self.dtype(),
      "input tensors of copy_memory_ should have same dtype");
  // AT_ASSERT(
  //     src.is_contiguous() && self.is_contiguous(),
  //     "input tensors of copy_memory_ should be contiguous");
  AT_ASSERT(
      src.device().index() == self.device().index(),
      "input tensors of copy_memory_ should have same device index");
  auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
  auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;

  int dst_size = 0;
  int src_size = 0;

  if (FormatHelper::IsPadded(&self)) {
    AT_ASSERT(self.storage_offset() == 0);
    dst_size = c10::multiply_integers(dst_desc.storage_sizes_);
  } else {
    auto dst_element = c10::multiply_integers(self.sizes());
    auto dst_storage = c10::multiply_integers(dst_desc.storage_sizes_);
    dst_size = (dst_element > dst_storage) ? dst_storage : dst_element;
  }

  if (FormatHelper::IsPadded(&src)) {
    AT_ASSERT(src.storage_offset() == 0);
    src_size = c10::multiply_integers(src_desc.storage_sizes_);
  } else {
    auto src_element = c10::multiply_integers(src.sizes());
    auto src_storage = c10::multiply_integers(src_desc.storage_sizes_);
    src_size = (src_element > src_storage) ? src_storage : src_element;
  }

  // Designed for the gather of tensors, ignoring npu_format_ and
  // copying continuous memory between npu tensors.
  auto ret = CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
      self,
      dst_size * self.itemsize(),
      src,
      dst_size * self.itemsize(),
      ACL_MEMCPY_DEVICE_TO_DEVICE);
  C10_NPU_CHECK(ret);

  if (!non_blocking) {
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    C10_NPU_CHECK(aclrtSynchronizeStream(stream));
  }
  return self;
}

} // namespace native
} // namespace at_npu