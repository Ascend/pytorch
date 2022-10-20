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

#include <ATen/native/npu/utils/CalcuOpUtil.h>
#include <ATen/native/npu/frame/StorageDescHelper.h>
#include "ATen/native/npu/common/InnerNpuNativeFunction.h"
#include "ATen/native/npu/utils/OpTemplate.h"
#include <c10/npu/interface/AsyncTaskQueueInterface.h>
#include "c10/npu/NPUStream.h"
#include <torch/csrc/autograd/record_function.h>

namespace at {
namespace native {
using namespace at::native::npu;

// format are base format (the format of src and dst are all nchw now)
// dtype are same
void copy_kernel_npu(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  RECORD_HOST_FUNCTION("d2dCopyByViewCopy", std::vector<c10::IValue>({src}));
  E2E_RECORD_FUNCTION("d2dCopyByViewCopy");
  auto self_size = self.sizes();
  auto self_stride = self.strides();
  auto src_size = src.sizes();
  auto src_stride = src.strides();
  OpCommand cmd;
  if (c10::npu::NpuRunMode::IsGraphMode()) {
    // The accurate offset of tensor will be provided by scalar input.
    cmd.Name("ViewCopy")
        .InputWithoutContiguous(self)
        .Input(self_size)
        .Input(self_stride)
        .Input(Scalar(self.storage_offset()), at::kLong)
        .InputWithoutContiguous(src)
        .Input(src_size)
        .Input(src_stride)
        .Input(Scalar(src.storage_offset()), at::kLong)
        .Output(self)
        .Run();
  } else {
    /*
     * (Ascend) Fix multi-compilation of ViewCopy op by wrapping scalar
     * storage_offset as a NPU Tensor instead of GE Const node.
     * If GE Data node can pass vaule of storage_offset to op,
     * we can switch storage_offset to Data node finally.
     * The same problem occurs in operator AsStrided.
     */
    cmd.Name("ViewCopy")
        .InputWithoutContiguous(self)
        .Input(self_size, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Input(self_stride, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Input(at::Scalar(0), at::kLong)
        .InputWithoutContiguous(src)
        .Input(src_size, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Input(src_stride, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Input(at::Scalar(0), at::kLong)
        .Output(self)
        .Run();
  }

  return;
}

// the dst and src are same dtype
// the dst and src have same elemsize
// if exceptCopySize is not defined, we will copy dst storage size
// so: caller should make sure that the storage size of src and dst are reasonable.
void copy_d2d_by_memcpy(Tensor& dst, const Tensor& src, int64_t exceptSize) {
  int64_t size = exceptSize;
  auto dst_mem_size = StorageDescHelper::GetMemorySize(dst);
  if (exceptSize == 0) {
    size = dst_mem_size;
  }

  if (c10::npu::NpuRunMode::IsGraphMode()) {
    if (dst_mem_size != size ||
        dst_mem_size != StorageDescHelper::GetMemorySize(src)) {
      // In graph mode, using ViewCopy to copy part data of src.
      copy_kernel_npu(dst, src, true);
      return;
    }

    /*
    In single op mode, the current interface may copy tensors between different
    shapes. So in the graph mode, only Reshape can be used to complete the copy
    of the complete memory block, not Identity.

    Refer to the following case:
    a [3,4,5,6] [3,4,30]
    b [3,4,5,6] [3,4,5,6]
    a.copy_(b)

    We should ensure that after copying, the shape of a is still [3,4,5,6] [3,4,30].

    In single op mode, it is always satisfied. But in graph mode, it is
    only satisfied when doing Reshape operations based on base_sizes_ of dst.
    */

    // In graph mode, using Reshape to copy whole data of src.
    SmallVector<int64_t, 5> self_base_sizes_5 =
        dst.storage().get_npu_desc().base_sizes_;
    SmallVector<int64_t, 32> self_base_sizes_32;
    std::for_each(
        self_base_sizes_5.begin(), self_base_sizes_5.end(), [&](int64_t dim) {
          self_base_sizes_32.emplace_back(dim);
        });
    auto self_size = dst.sizes();
    OpCommand cmd;
    cmd.Name("Reshape")
        .InputWithoutContiguous(src)
        .Input(self_base_sizes_32)
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

  // The current logic is only used in single op mode.
  aclError error = c10::npu::queue::LaunchAsyncCopyTask(
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
} // namespace at
