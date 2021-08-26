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

#include "ATen/native/npu/frame/StorageDescHelper.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/frame/InferFormat.h"

namespace at {
namespace native {
namespace npu {

bool StorageDescHelper::MetaDataAreMatch(const Tensor* tensor) {
  auto& desc = tensor->storage().unsafeGetStorageImpl()->npu_desc_;
  return IsSameSize(desc.base_sizes_, tensor->sizes()) && IsSameSize(desc.base_strides_, tensor->strides());
}

bool StorageDescHelper::OffsetAreMatch(const Tensor* tensor) {
  return tensor->storage_offset() == 0;
}

// copy related
bool StorageDescHelper::IsSameDesc(const NPUStorageDesc& a, const NPUStorageDesc& b) {
  if ((a.origin_format_ != b.origin_format_) || (a.npu_format_ != b.npu_format_)) {
    if ((!FormatHelper::IsBaseFormatType(a.npu_format_)) || (!FormatHelper::IsBaseFormatType(b.npu_format_))) {
      return false;
    }
  }
  return (a.base_sizes_ == b.base_sizes_) && (a.base_strides_ == b.base_strides_) && (a.storage_sizes_ == b.storage_sizes_);
}

bool StorageDescHelper::IsSameDesc(const Tensor& a, const Tensor& b) {
  auto descA = a.storage().unsafeGetStorageImpl()->npu_desc_;
  auto descB = b.storage().unsafeGetStorageImpl()->npu_desc_;
  return IsSameDesc(descA, descB);
}

bool StorageDescHelper::IsSameSize(SmallVector<int64_t,5> a, IntArrayRef b) {
  if (a.size() == b.size()) {
    return std::equal(a.begin(), a.end(), b.begin());
  }
  return false;
}

void StorageDescHelper::UpdateDesc(NPUStorageDesc& npuDesc, IntArrayRef& new_size) {
  npuDesc.base_sizes_ = new_size;

  // 计算连续场景下size对应的stride值
  auto dim_ = new_size.size();
  SmallVector<int64_t, 5> new_stride(dim_);
  if (dim_ > 0) {
    int last_idx = dim_ - 1;
    new_stride[last_idx] = 1;
    for (auto i = last_idx - 1; i >= 0; --i) {
      new_stride[i] = new_stride[i + 1] * std::max<int64_t>(new_size[i + 1], 1);
    }
  }
  npuDesc.base_strides_ = new_stride;

  // 更新物理内存信息
  auto physical_size = FormatHelper::GetStorageSizes(npuDesc);
  npuDesc.storage_sizes_ = physical_size;
}

FormatShape StorageDescHelper::ComputeStrideFromShape(const FormatShape& shape) {
  FormatShape compute_stride = shape;
  compute_stride[shape.size() - 1] = 1;
  for (auto i = shape.size() - 1; i > 0; i--) {
    compute_stride[i-1] = shape[i] * compute_stride[i];
  }
  return compute_stride;
}

void StorageDescHelper::SetDesc(Tensor& dst) {
  dst.storage().unsafeGetStorageImpl()->npu_desc_ = SetDesc();
}

void StorageDescHelper::SetDesc(Tensor& dst, IntArrayRef size, IntArrayRef strides) {
  dst.storage().unsafeGetStorageImpl()->npu_desc_ = SetDesc(size, strides);
}

void StorageDescHelper::SetDesc(Tensor& dst, IntArrayRef size, IntArrayRef strides, aclFormat format) {
  dst.storage().unsafeGetStorageImpl()->npu_desc_ = SetDesc(size, strides, format);
}

void StorageDescHelper::CopyDesc(Tensor& dst, const Tensor& src) {
  CopyDesc(dst, src.storage());
}

void StorageDescHelper::CopyDesc(Tensor& dst, const Storage& src) {
  CopyDesc(dst, src.unsafeGetStorageImpl()->npu_desc_);
}

void StorageDescHelper::CopyDesc(const Tensor& dst, const NPUStorageDesc& src_desc) {
  auto& dstDesc = dst.storage().unsafeGetStorageImpl()->npu_desc_;
  dstDesc = src_desc;
}

void StorageDescHelper::ReflushDescBySelf(const Tensor& src) {
  auto& desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
  desc.base_sizes_ = src.sizes();
  desc.storage_sizes_ = src.sizes();
  desc.base_strides_ = src.strides();
}

NPUStorageDesc StorageDescHelper::SetDesc() {
  return SetDesc({0}, {});
}

NPUStorageDesc StorageDescHelper::SetDesc(IntArrayRef size, IntArrayRef strides) {
  return SetDesc(size, strides, InferFormat::GuessBaseFormat(size));
}

NPUStorageDesc StorageDescHelper::SetDesc(IntArrayRef size, IntArrayRef strides, aclFormat format) {
  struct NPUStorageDesc npu_desc;
  npu_desc.base_sizes_ = size;
  npu_desc.base_strides_ = strides;
  // guess ori format and npu format unit by size and dst format
  // eg: size: [2,3,4,5] format: nd
  // we will return [NCHW, NCHW] because 4 dim tensor must be nchw,
  // then the tensor used to be the input of conv2d will not make mistake
  aclFormat baseFormat, npuFormat;
  std::tie(baseFormat, npuFormat) = InferFormat::GuessFormatUnit(size, format);
  npu_desc.storage_sizes_ = FormatHelper::GetStorageSizes(npuFormat, size);
  npu_desc.origin_format_ = baseFormat;
  npu_desc.npu_format_ = npuFormat;
  return npu_desc;
}

int64_t StorageDescHelper::GetMemorySize(const NPUStorageDesc& desc) {
  auto physical_size = FormatHelper::GetStorageSizes(desc);
  return prod_intlist(physical_size);
}

int64_t StorageDescHelper::GetMemorySize(const Tensor& dst) {
  auto desc = dst.storage().unsafeGetStorageImpl()->npu_desc_;
  return GetMemorySize(desc);
}

int64_t StorageDescHelper::GetMemorySize(IntArrayRef size, aclFormat format) {
  auto physical_size = FormatHelper::GetStorageSizes(format, size);
  return prod_intlist(physical_size);
}

int64_t StorageDescHelper::GetValidMemorySize(const Tensor& tensor) {
  int64_t real_bytes = 0;
  for (int64_t i = tensor.dim() - 1; i >= 0; i--) {
    real_bytes += (tensor.size(i) - 1) * tensor.stride(i);
  }
  return real_bytes + 1;
}


} // namespace npu
} // namespace native
} // namespace at