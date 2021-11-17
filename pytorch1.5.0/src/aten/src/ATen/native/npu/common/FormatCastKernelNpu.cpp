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

#include "ATen/native/npu/common/FormatCastHelper.h"
#include "ATen/native/npu/frame/FormatHelper.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/NpuStorageOffsetGuard.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor format_cast_impl_out_npu(Tensor& dst, const Tensor& src) {
  string srcFormat = FormatHelper::GetFormatName(src);
  string dstFormat = FormatHelper::GetFormatName(dst);

  if (!FormatCastHelper::IsSameGroupType(src, dst)) {
    bool res = FormatCastHelper::format_cast_between_group(dst, src, format_cast_impl_out_npu);
    if (!res) {
      AT_ERROR("unsupport cast from ", srcFormat, " to ", dstFormat);
    }
    return dst;
  }

  /*
  In order to consider performance, The current adaptation uses the direct call
  format conversion operator `Transdata`, Unfortunately, Different from ordinary
  computing operators, operator `Transdata` belongs to a special memory movement
  operator, which leads to too much special treatment in the existing framework,
  reduced scalability and maintainability.

  So, to solve the problem, we use the `Identity` operator instead of the
  `Transdata` operator to meet the current memory move function. Then, it is
  determined by the FE framework to insert the transdata operator into the
  graph.

  The purpose is to control the format conversion operator in the underlying FE
  framework.
  */

  // offset guard with InputWithoutContiguous
  // view + transdata scene: we do transdata first, then we should set offset =
  // 0 to keep results correct.
  NpuStorageOffsetGuard guard_input(const_cast<Tensor &>(src));
  NpuStorageOffsetGuard guard_output(dst);
  OpCommand cmd;
  cmd.Name("Identity")
     .InputWithoutContiguous(src)
     .Output(dst)
     .Run();

  return dst;
}

// convert src from src_format to dst_format, write the result into dst
Tensor& format_cast_npu_(Tensor& dst, const Tensor& src) {
  NPUStorageDesc src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
  NPUStorageDesc dst_desc = dst.storage().unsafeGetStorageImpl()->npu_desc_;
  if (src_desc.npu_format_ == dst_desc.npu_format_) {
    dst.copy_(src);
    return dst;
  }

  // calculate the output result of the NPU
  format_cast_impl_out_npu(dst, src);

  return dst;
}

// conver self to acl_format, write the result into new result tensor
Tensor format_cast_npu(
    const Tensor& src,
    int64_t acl_format) {
  NPUStorageDesc src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
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
  TORCH_CHECK(src.scalar_type() == ScalarType::Float || src.scalar_type() == ScalarType::Half,
    "can not cast format when src is not float32 or float16");

  Tensor dst = at::empty_with_format(
      src_desc.base_sizes_, src.options(), acl_format);

  // calculate the output result of the NPU
  format_cast_impl_out_npu(dst, src);

  // format cast only change physical layout of base tensor and view tensor's
  // metadata remain unchanged
  dst.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());
  return dst;
}

// conver self to acl_format, write the result into self
Tensor& format_cast_npu_(
    Tensor& src,
    int64_t acl_format) {
  NPUStorageDesc src_desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
  if (src_desc.npu_format_ == acl_format) {
    return src;
  }
  if (FormatHelper::IsBaseFormatType(src) && 
      FormatHelper::IsBaseFormatType(static_cast<aclFormat>(acl_format))) {
    FormatCastHelper::format_cast_as_base_format(src, static_cast<aclFormat>(acl_format));
    return src;
  }
  // transdata only support float and half
  TORCH_CHECK(src.scalar_type() == ScalarType::Float || src.scalar_type() == ScalarType::Half,
    "can not cast format when src is not float32 or float16");

  Tensor dst = at::empty_with_format(
      src_desc.base_sizes_, src.options(), acl_format);

  // calculate the output result of the NPU
  format_cast_impl_out_npu(dst, src);

  // format cast only change physical layout of base tensor and view tensor's
  // metadata remain unchanged
  src.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());

  return src;
}

} // namespace native
} // namespace at
