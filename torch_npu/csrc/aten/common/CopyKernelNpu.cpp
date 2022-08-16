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
#include <ATen/record_function.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"
#include "torch_npu/csrc/core/npu/NPURunMode.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

// format are base format (the format of src and dst are all nchw now)
// dtype are same
// so the view_value and ReflushDescBySelf are base on the hypothesis above.
void copy_kernel_npu(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  RECORD_FUNCTION("d2dCopyWithViewCopy", std::vector<c10::IValue>({src}));

  GraphModeGuard mode_guard(c10_npu::ModeKind::SINGLE_OP_MODE);

  auto self_size = self.sizes();
  auto self_stride = self.strides();
  auto src_size = src.sizes();
  auto src_stride = src.strides();

  OpCommand cmd;
  cmd.Name("ViewCopy")
      .InputWithoutContiguous(self)
      .Input(self_size)
      .Input(self_stride)
      .InputScalarToNPUTensor(at::Scalar(0), at::kLong)
      .InputWithoutContiguous(src)
      .Input(src_size)
      .Input(src_stride)
      .InputScalarToNPUTensor(at::Scalar(0), at::kLong)
      .Output(self)
      .Run();

  return;
}

// the dst and src are same dtype
// the dst and src have same elemsize
// if exceptCopySize is not defined, we will copy dst storage size
// so: caller should make sure that the storage size of src and dst are reasonable.
void copy_d2d_by_memcpy(at::Tensor& dst, const at::Tensor& src, int64_t exceptSize) {
  int64_t size = exceptSize;
  auto dst_mem_size = StorageDescHelper::GetMemorySize(dst);
  if (exceptSize == 0) {
    size = dst_mem_size;
  }

  if (c10_npu::NpuRunMode::IsGraphMode()) {
    if (dst_mem_size != size ||
        dst_mem_size != StorageDescHelper::GetMemorySize(src)) {
      // In graph mode, using PTcopy to copy part data of src.
      copy_kernel_npu(dst, src, true);
      return;
    }

    // In graph mode, using identity to copy whole data of src.
    OpCommand cmd;
    cmd.Name("Identity")
        .InputWithoutContiguous(src)
        .Output(dst)
        .Run();
    return;
  }

  if(!dst.data_ptr()) {
    TORCH_WARN("copy_d2d_by_memcpy, dst.data_ptr() is null.");
    return;
  }

  if(!src.data_ptr()) {
    TORCH_WARN("copy_d2d_by_memcpy, src.data_ptr() is null.");
    return;
  }

  aclError error = c10_npu::queue::LaunchAsyncCopyTask(
      dst.data_ptr(),
      size * dst.element_size(),
      src.data_ptr(),
      size * dst.element_size(),
      ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (error != ACL_ERROR_NONE) {
    AT_ERROR("async copy device to device error.");
    return;
  }
}
} // namespace native
} // namespace at_npu
