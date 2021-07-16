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

#include "FormatCastHelper.h"
#include "ATen/native/npu/frame/FormatHelper.h"

namespace at {
namespace native {
namespace npu {

bool FormatCastHelper::IsSameGroupType(const Tensor& src, const Tensor& dst) {
  auto src_format = src.storage().get_npu_desc().npu_format_;
  auto dst_format = dst.storage().get_npu_desc().npu_format_;
  return FormatHelper::GetBaseFormat(src_format) == FormatHelper::GetBaseFormat(dst_format);
}

void FormatCastHelper::base_format_cast_nocheck(const Tensor& dst, const Tensor& src) {
  dst.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());
  dst.copy_memory_(src, true);
}

void FormatCastHelper::format_cast_as_base_format(const Tensor& src, aclFormat format) {
  AT_ASSERT(FormatHelper::IsBaseFormatType(format), "dst format must be base format");
  AT_ASSERT(FormatHelper::IsBaseFormatType(src), "src format must be base format");
  
  auto& src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
  // due to CANN principle : if the ori format of a tensor is the
  // same as the npu format, then its base shape must be same as storage shape
  // so we should not change the storage shape when format cast between base format
  src_desc.origin_format_ = format;
  src_desc.npu_format_ = format;
  return;
}

bool FormatCastHelper::format_cast_between_group(Tensor& dst, const Tensor& src, FormatCastHelper::FormatCastFunc format_cast_inside_group) {
  if (FormatHelper::IsBaseFormatType(src)) {
    if (FormatHelper::IsBaseFormatType(dst)) {
      // src base format (src format) -> dst base format
      base_format_cast_nocheck(dst, src); // only need to copy memory
      return true;
    } else {
      // src base format (src format) -> dst base format
      // dst base format -> dst format
      auto src_base_format = FormatHelper::GetBaseFormat(src);
      format_cast_as_base_format(src, FormatHelper::GetBaseFormat(dst)); // prepare: covert src to dst base format
      format_cast_inside_group(dst, src); // src base format (src format) -> dst base format
      format_cast_as_base_format(src, src_base_format); // recover: dst base format -> dst format
      return true;
    }
  } else {
    if (FormatHelper::IsBaseFormatType(dst)) {
      // src format -> src base format
      // src base format -> dst base format (dst format)
      auto dst_base_format =FormatHelper::GetBaseFormat(dst);
      format_cast_as_base_format(dst, FormatHelper::GetBaseFormat(src)); // prepare: cover dst to src base format
      format_cast_inside_group(dst, src); // src format -> src base format
      format_cast_as_base_format(dst, dst_base_format); // recover: src base format -> dst format
      return true;
    }
  }
  return false;
}

Tensor FormatCastHelper::ApplyBaseFormatTensorBy(const Tensor& src) {
  auto format = FormatHelper::GetBaseFormat(src);
  return src.npu_format_cast(format);
}

Tensor& FormatCastHelper::CovertSelfToBaseFormat(Tensor& src) {
  auto format = FormatHelper::GetBaseFormat(src);
  return src.npu_format_cast_(format);
}


} // namespace npu
} // namespace native
} // namespace at