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

#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/common/FormatCastHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor format_cast_impl_out_npu(at::Tensor& dst, const at::Tensor& src) {
  string srcFormat = FormatHelper::GetFormatName(src);
  string dstFormat = FormatHelper::GetFormatName(dst);

  if (!FormatCastHelper::IsSameGroupType(src, dst)) {
    bool res = FormatCastHelper::format_cast_between_group(dst, src, format_cast_impl_out_npu);
    if (!res) {
      AT_ERROR("unsupport cast from ", srcFormat, " to ", dstFormat);
    }
    return dst;
  }

  TransDataOpCommand cmd;
  cmd.Name("TransData")
    .InputAndOutput(src, dst)
    .Attr("src_format", srcFormat)
    .Attr("dst_format", dstFormat)
    .Run();
  return dst;
}

// convert src from src_format to dst_format, write the result into dst
at::Tensor& NPUNativeFunctions::npu_format_cast_(at::Tensor& dst, const at::Tensor& src) {
  c10::NPUStorageDesc src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
  c10::NPUStorageDesc dst_desc = dst.storage().unsafeGetStorageImpl()->npu_desc_;
  if (src_desc.npu_format_ == dst_desc.npu_format_) {
    dst.copy_(src);
    return dst;
  }

  // calculate the output result of the NPU
  format_cast_impl_out_npu(dst, src);

  return dst;
}

// conver self to acl_format, write the result into new result tensor
at::Tensor NPUNativeFunctions::npu_format_cast(
    const at::Tensor& src,
    int64_t acl_format) {
  c10::NPUStorageDesc src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
  if (src_desc.npu_format_ == acl_format) {
    NPU_LOGD("no need to do format cast");
    return src;
  }
  if (FormatHelper::IsBaseFormatType(src) &&
      FormatHelper::IsBaseFormatType(static_cast<aclFormat>(acl_format))) {
    FormatCastHelper::format_cast_as_base_format(src, static_cast<aclFormat>(acl_format));
    return src;
  }
  // transdata only support float and half
  TORCH_CHECK(src.scalar_type() == at::ScalarType::Float || src.scalar_type() == at::ScalarType::Half,
      "can not cast format when src is not float32 or float16");

  at::Tensor dst = OpPreparation::ApplyTensorWithFormat(
      src_desc.base_sizes_, src.options(), acl_format);

  // calculate the output result of the NPU
  format_cast_impl_out_npu(dst, src);

  // format cast only change physical layout of base tensor and view tensor's
  // metadata remain unchanged
  dst.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());
  return dst;
}

// conver self to acl_format, write the result into self
at::Tensor& NPUNativeFunctions::npu_format_cast_(
    at::Tensor& src,
    int64_t acl_format) {
  c10::NPUStorageDesc src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
  if (src_desc.npu_format_ == acl_format) {
    return src;
  }
  if (FormatHelper::IsBaseFormatType(src) &&
      FormatHelper::IsBaseFormatType(static_cast<aclFormat>(acl_format))) {
    FormatCastHelper::format_cast_as_base_format(src, static_cast<aclFormat>(acl_format));
    return src;
  }
  // transdata only support float and half
  TORCH_CHECK(src.scalar_type() == at::ScalarType::Float || src.scalar_type() == at::ScalarType::Half,
      "can not cast format when src is not float32 or float16");

  at::Tensor dst = OpPreparation::ApplyTensorWithFormat(
      src_desc.base_sizes_, src.options(), acl_format);

  // calculate the output result of the NPU
  format_cast_impl_out_npu(dst, src);

  // format cast only change physical layout of base tensor and view tensor's
  // metadata remain unchanged
  src.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());

  return src;
}

} // namespace native
} // namespace at_npu
