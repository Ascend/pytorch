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
#include <c10/core/Scalar.h>
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
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {

// format are base format (the format of src and dst are all nchw now)
// dtype are same
// so the view_value and ReflushDescBySelf are base on the hypothesis above.
bool AicoreValid(at::Tensor& self, const at::Tensor& src) {
  const auto& dst_storage_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
  auto self_size = self.sizes();
  auto self_stride = self.strides();
  auto dst_storage_size_len = dst_storage_sizes.size();
  auto self_size_len = self_size.size();

  // count the difference between dst_storage and dst_size.
  int diff = dst_storage_size_len - self_size_len;
  if (diff == 1) {
    return true;
  }

  if (diff != 0) {
    return false;
  }

  // record the index of the difference.
  int diff_index = self_size_len;
  for (int64_t i = 0; i < self_size_len; i++) {
    if (dst_storage_sizes[i] != self_size[i]) {
      ++diff;
      if (diff > 1) {
        return false;
      }
      // differece should be 1.
      diff_index = i;
    }
  }

  // if diff or diff_index equals 0, no need viewcopy.
  if (diff == 0 || diff_index == 0) {
    return false;
  } 

  const auto& dst_base_stride = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.base_strides_;
  // dst_base_stride should be equal to dst_storage_stride except for diff_index
  if (self_stride.size() > dst_base_stride.size()) {
    return false;
  } 
  
  for (int64_t i = 0; i < self_stride.size(); i++) {
    if (dst_base_stride[i] != self_stride[i] && i != diff_index) {
      return false;
    } 
  }

  // dtype cannot be double and dst_size has to be equal with src_size.
  if (self.dtype() == at::ScalarType::Double || self_size != src.sizes()) {
    return false;
  } 
  
  return true;
}

void copy_kernel_npu(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  RECORD_FUNCTION("contiguous_d_ViewCopy", std::vector<c10::IValue>({src}));

  auto self_size = self.sizes();
  auto self_stride = self.strides();
  auto src_size = src.sizes();
  auto src_stride = src.strides();

  if (c10_npu::NpuRunMode::IsGraphMode()) {
    OpCommand cmd;
    cmd.Name("ViewCopy")
        .InputWithoutContiguous(self)
        .Input(self_size)
        .Input(self_stride)
        .Input(c10::Scalar(self.storage_offset()), at::kLong)
        .InputWithoutContiguous(src)
        .Input(src_size)
        .Input(src_stride)
        .Input(c10::Scalar(src.storage_offset()), at::kLong)
        .Output(self)
        .Run();

    return;
  };

  if (AicoreValid(self, src)) {
    at::Tensor contiguous_src(src);
    if (!NpuUtils::check_match(&contiguous_src)) {
      contiguous_src = NpuUtils::format_contiguous(contiguous_src);
    }
    src_stride = contiguous_src.strides();

    OpCommand cmd;
    cmd.Name("ViewCopy")
      .InputWithoutContiguous(self)
      .Input(self_size, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(self_stride, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(at::Scalar(0), at::kLong)
      .InputWithoutContiguous(contiguous_src)
      .Input(src_size, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(src_stride, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(at::Scalar(0), at::kLong)
      .Output(self)
      .Run();
  }
  else {
    OpCommand cmd;
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
      .Attr("_exclude_engines", (string)"AiCore")
      .Run();
  }
  
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
      // In graph mode, using Viewcopy to copy part data of src.
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
    c10::SmallVector<int64_t, 5> self_base_sizes_5 =
        torch_npu::NPUBridge::GetNpuStorageImpl(dst.storage().unsafeGetStorageImpl())->get_npu_desc().base_sizes_;
    c10::SmallVector<int64_t, 32> self_base_sizes_32(self_base_sizes_5.begin(), self_base_sizes_5.end());
    OpCommand cmd;
    cmd.Name("Reshape")
        .InputWithoutContiguous(src)
        .Input(self_base_sizes_32)
        .Output(dst)
        .Run();
    return;
  }

  if(!dst.data_ptr()) {
    TORCH_NPU_WARN("copy_d2d_by_memcpy, dst.data_ptr() is null.");
    return;
  }

  if(!src.data_ptr()) {
    TORCH_NPU_WARN("copy_d2d_by_memcpy, src.data_ptr() is null.");
    return;
  }

  if (dst.data_ptr() == src.data_ptr() && dst.element_size() == src.element_size()) {
    return;
  }

  // The current logic is only used in single op mode.
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
