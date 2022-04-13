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

namespace {
// view value : {numel, storage_offset, strides.size, strides}
SmallVector<int64_t, N> get_view_value(
    const Tensor& t,
    const IntArrayRef& strides) {
  static SmallVector<int64_t, N> value;
  // It is determined by the definition of view attr
  value.resize(strides.size() + 3);
  value[0] = t.storage().unsafeGetStorageImpl()->numel(); // storageImpl numel
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
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  RECORD_HOST_FUNCTION("d2dCopyWithPTCopy", std::vector<c10::IValue>({src}));
  // In single op mode, PTcopy will be replaced by ViewCopy in the future
  if (c10::npu::NpuRunMode::IsGraphMode()) {
    auto self_size = self.sizes();
    auto self_stride = self.strides();
    auto src_size = src.sizes();
    auto src_stride = src.strides();
    OpCommand cmd;
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
    return;
  };

  const int64_t HEAD_FLAG = 0x6461656800000000;
  const int64_t FIXED_LEN =
      9; // head, len, version, two tensors' numel, offset and strides lens
  const int64_t VERSION = 0; // op version

  auto selfStrides = self.strides();
  auto srcStrides = src.strides();

  int64_t len = FIXED_LEN + selfStrides.size() + srcStrides.size();
  SmallVector<int64_t, N> value;
  value.emplace_back(HEAD_FLAG); // head flag
  value.emplace_back(len); // value length
  value.emplace_back(VERSION);

  auto inputValue = get_view_value(src, srcStrides);
  auto outValue = get_view_value(self, selfStrides);

  value.insert(value.end(), inputValue.begin(), inputValue.end());
  value.insert(value.end(), outValue.begin(), outValue.end());

  Tensor attrTensor = CalcuOpUtil::copy_tensor_host_to_device(
      from_blob(value.data(), {value.size()}, dtype(ScalarType::Long)));

  auto src_desc_bp = src.storage().get_npu_desc();
  auto self_desc_bp = self.storage().get_npu_desc();

  // The action of PTcopy_ is defined by attrTensor, so the member of NPUStorageDesc
  // can not affect the result, but the PTcopy_ will check base_size and storage_size,
  // so we reflash them here, and recover them later.
  StorageDescHelper::ReflushDescBySelf(src);
  StorageDescHelper::ReflushDescBySelf(self);

  SmallVector<NPUTensorDesc, N> inputs = {
      NPUTensorDesc(src), NPUTensorDesc(attrTensor)};
  SmallVector<NPUTensorDesc, N> outputs = {NPUTensorDesc(self)};

  CalcuOpUtil::execute_npu_operate("PTcopy_", inputs, outputs, {});

  StorageDescHelper::CopyDesc(src, src_desc_bp);
  StorageDescHelper::CopyDesc(self, self_desc_bp);
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