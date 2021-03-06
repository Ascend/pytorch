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
namespace {
// view value : {numel, storage_offset, strides.size, strides}
c10::SmallVector<int64_t, N> get_view_value(
    const at::Tensor& t,
    const c10::IntArrayRef& strides) {
  static c10::SmallVector<int64_t, N> value;
  // It is determined by the definition of view attr
  value.resize(strides.size() + 3);
  value[0] = t.storage().nbytes() / t.element_size(); // storageImpl numel
  value[1] = t.storage_offset(); // default to 0
  value[2] = strides.size();
  for (size_t i = 0; i < strides.size(); i++) {
    value[3 + i] = strides[i];
  }
  return value;
}
} // namespace

// format are base format (the format of src and dst are all nchw now)
// dtype are same
// so the view_value and ReflushDescBySelf are base on the hypothesis above.
void copy_kernel_npu(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  RECORD_FUNCTION("d2dCopyWithPTCopy", std::vector<c10::IValue>({src}));
  // In graph mode, PTcopy will be supported in the future
  GraphModeGuard mode_guard(c10_npu::ModeKind::SINGLE_OP_MODE);
  const int64_t HEAD_FLAG = 0x6461656800000000;
  const int64_t FIXED_LEN =
      9; // head, len, version, two tensors' numel, offset and strides lens
  const int64_t VERSION = 0; // op version

  auto selfStrides = self.strides();
  auto srcStrides = src.strides();

  int64_t len = FIXED_LEN + selfStrides.size() + srcStrides.size();
  c10::SmallVector<int64_t, N> value;
  value.emplace_back(HEAD_FLAG); // head flag
  value.emplace_back(len); // value length
  value.emplace_back(VERSION);

  auto inputValue = get_view_value(src, srcStrides);
  auto outValue = get_view_value(self, selfStrides);

  value.insert(value.end(), inputValue.begin(), inputValue.end());
  value.insert(value.end(), outValue.begin(), outValue.end());

  at::Tensor attrTensor = CalcuOpUtil::copy_tensor_host_to_device(
      at::from_blob(value.data(), {value.size()}, dtype(at::ScalarType::Long)));

  auto src_desc_bp = torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc();
  auto self_desc_bp = torch_npu::NPUBridge::GetNpuStorageImpl(self)->get_npu_desc();

  // The action of PTcopy_ is defined by attrTensor, so the member of NPUStorageDesc
  // can not affect the result, but the PTcopy_ will check base_size and storage_size,
  // so we reflash them here, and recover them later.
  StorageDescHelper::ReflushDescBySelf(src);
  StorageDescHelper::ReflushDescBySelf(self);

  c10::SmallVector<NPUTensorDesc, N> inputs = {
      NPUTensorDesc(src), NPUTensorDesc(attrTensor)};
  c10::SmallVector<NPUTensorDesc, N> outputs = {NPUTensorDesc(self)};

  CalcuOpUtil::execute_npu_operate("PTcopy_", inputs, outputs, {});

  StorageDescHelper::CopyDesc(src, src_desc_bp);
  StorageDescHelper::CopyDesc(self, self_desc_bp);
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