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

#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include <c10/util/accumulate.h>

namespace at_npu
{
  namespace native
  {

    bool StorageDescHelper::MetaDataAreMatch(const at::Tensor *tensor)
    {
      auto &desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(*tensor);
      return IsSameSize(desc.base_sizes_, tensor->sizes()) && IsSameSize(desc.base_strides_, tensor->strides());
    }

    // copy related
    bool StorageDescHelper::IsSameDesc(const torch_npu::NPUStorageDesc &a, const torch_npu::NPUStorageDesc &b)
    {
      if ((a.origin_format_ != b.origin_format_) || (a.npu_format_ != b.npu_format_))
      {
        if ((!FormatHelper::IsBaseFormatType(a.npu_format_)) || (!FormatHelper::IsBaseFormatType(b.npu_format_)))
        {
          return false;
        }
      }
      return (a.base_sizes_ == b.base_sizes_) && (a.base_strides_ == b.base_strides_) && (a.storage_sizes_ == b.storage_sizes_);
    }

    bool StorageDescHelper::IsSameDesc(const at::Tensor &a, const at::Tensor &b)
    {
      const auto &descA = torch_npu::NPUBridge::GetNpuStorageImplDesc(a);
      const auto &descB = torch_npu::NPUBridge::GetNpuStorageImplDesc(b);
      return IsSameDesc(descA, descB);
    }

    bool StorageDescHelper::IsSameSize(const c10::SmallVector<int64_t, 5> &a, const c10::IntArrayRef &b)
    {
      if (a.size() == b.size())
      {
        return std::equal(a.begin(), a.end(), b.begin());
      }
      return false;
    }

    void StorageDescHelper::UpdateDesc(torch_npu::NPUStorageDesc &npuDesc, const c10::IntArrayRef &new_data_sizes,
                                       const c10::IntArrayRef &new_shape_sizes)
    {
      int64_t new_data_numel = c10::multiply_integers(new_data_sizes);
      int64_t new_shape_numel = c10::multiply_integers(new_shape_sizes);
      const c10::IntArrayRef &new_size = new_data_numel > new_shape_numel ? new_data_sizes : new_shape_sizes;

      npuDesc.base_sizes_ = new_size;

      // 计算连续场景下size对应的stride值
      int64_t dim_ = static_cast<int64_t>(new_size.size());
      c10::SmallVector<int64_t, 5> new_stride(dim_);
      if (dim_ > 0) {
        int64_t last_idx = dim_ - 1;
        new_stride[last_idx] = 1;
        for (auto i = last_idx - 1; i >= 0; --i) {
          new_stride[i] = new_stride[i + 1] * std::max<int64_t>(new_size[i + 1], 1);
        }
      }
      npuDesc.base_strides_ = new_stride;

      // 更新物理内存信息
      npuDesc.storage_sizes_ = FormatHelper::GetStorageSizes(npuDesc);
      if (new_data_numel > new_shape_numel) {
        // Refresh format to base format only when flattening storage data
        npuDesc.storage_sizes_ = new_size;
        npuDesc.npu_format_ = InferFormat::GuessStorageFormat(npuDesc.storage_sizes_, ACL_FORMAT_ND);
      }
    }

    FormatShape StorageDescHelper::ComputeStrideFromShape(const FormatShape &shape)
    {
      FormatShape compute_stride = shape;
      compute_stride[shape.size() - 1] = 1;
      for (auto i = shape.size() - 1; i > 0; i--)
      {
        compute_stride[i - 1] = shape[i] * compute_stride[i];
      }
      return compute_stride;
    }

    void StorageDescHelper::SetDesc(at::Tensor &dst)
    {
      torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_ = SetDesc(dst.dtype());
    }

    void StorageDescHelper::SetDesc(at::Tensor &dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides)
    {
      torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_ = SetDesc(dst.dtype(), size, strides);
    }

    void StorageDescHelper::SetDesc(at::Tensor &dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides, aclFormat format)
    {
      torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_ = SetDesc(dst.dtype(), size, strides, format);
    }

    void StorageDescHelper::CopyDesc(at::Tensor &dst, const at::Tensor &src)
    {
      CopyDesc(dst, src.storage());
    }

    void StorageDescHelper::CopyDesc(at::Tensor &dst, const c10::Storage &src)
    {
      CopyDesc(dst, torch_npu::NPUBridge::GetNpuStorageImpl(src.unsafeGetStorageImpl())->npu_desc_);
    }

    void StorageDescHelper::CopyDesc(const at::Tensor &dst, const torch_npu::NPUStorageDesc &src_desc)
    {
      auto &dstDesc = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_;
      dstDesc = src_desc;
    }

    void StorageDescHelper::ReflushDescBySelf(const at::Tensor &src)
    {
      auto &desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
      desc.base_sizes_ = src.sizes();
      desc.storage_sizes_ = src.sizes();
      desc.base_strides_ = src.strides();
    }

    torch_npu::NPUStorageDesc StorageDescHelper::SetDesc(const caffe2::TypeMeta &dtype)
    {
      return SetDesc(dtype, {0}, {});
    }

    torch_npu::NPUStorageDesc StorageDescHelper::SetDesc(const caffe2::TypeMeta &dtype, const c10::IntArrayRef& size,
                                                         const c10::IntArrayRef& strides)
    {
      return SetDesc(dtype, size, strides, InferFormat::GuessBaseFormat(size));
    }

    torch_npu::NPUStorageDesc StorageDescHelper::SetDesc(const caffe2::TypeMeta &dtype, const c10::IntArrayRef& size,
                                                         const c10::IntArrayRef& strides, aclFormat format)
    {
      struct torch_npu::NPUStorageDesc npu_desc;
      npu_desc.data_type_ = dtype;
      npu_desc.base_sizes_ = size;
      npu_desc.base_strides_ = strides;
      // guess ori format and npu format unit by size and dst format
      // eg: size: [2,3,4,5] format: nd
      // we will return [NCHW, NCHW] because 4 dim tensor must be nchw,
      // then the tensor used to be the input of conv2d will not make mistake
      aclFormat baseFormat;
      aclFormat npuFormat;
      std::tie(baseFormat, npuFormat) = InferFormat::GuessFormatUnit(size, format);
      npu_desc.storage_sizes_ = FormatHelper::GetStorageSizes(npuFormat, size, dtype);
      npu_desc.origin_format_ = baseFormat;
      npu_desc.npu_format_ = npuFormat;
      return npu_desc;
    }

    int64_t StorageDescHelper::GetMemorySize(const torch_npu::NPUStorageDesc &desc)
    {
      const auto &physical_size = FormatHelper::GetStorageSizes(desc);
      return c10::multiply_integers(physical_size);
    }

    int64_t StorageDescHelper::GetMemorySize(const at::Tensor &dst)
    {
      auto desc = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_;
      return GetMemorySize(desc);
    }

    int64_t StorageDescHelper::GetMemorySize(const c10::IntArrayRef& size, aclFormat format, caffe2::TypeMeta dtype)
    {
        const auto &physical_size = FormatHelper::GetStorageSizes(format, size, dtype);
        return c10::multiply_integers(physical_size);
    }

    int64_t StorageDescHelper::GetValidMemorySize(const at::Tensor &tensor)
    {
      int64_t real_bytes = 0;
      for (int64_t i = tensor.dim() - 1; i >= 0; i--)
      {
        real_bytes += (tensor.size(i) - 1) * tensor.stride(i);
      }
      return real_bytes + 1;
    }

  } // namespace native
} // namespace at_npu

