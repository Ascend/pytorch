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

#include "InferFormat.h"
#include "FormatHelper.h"
#include "c10/npu/OptionsManager.h"

namespace at {
namespace native {
namespace npu {

aclFormat InferFormat::GuessFormatWhenContiguous(const Tensor& tensor) {
  auto desc = tensor.storage().unsafeGetStorageImpl()->npu_desc_; 
  // fix: NCDHW -> default format
  if ((desc.origin_format_ == ACL_FORMAT_NCDHW)) {
    if ((tensor.sizes().size() != desc.base_sizes_.size()) && (tensor.sizes().size() <= 4)) {
      return ACL_FORMAT_NCHW;
    }
  }
  return desc.origin_format_;
}

// NOTE: this method should cooperate with shape infer.
std::tuple<aclFormat, aclFormat> InferFormat::GuessFormatUnit(const IntArrayRef& size, aclFormat format) {
  if ((FormatHelper::GetBaseFormat(format) == ACL_FORMAT_NCDHW) && (size.size() > 4)) {
    return std::make_tuple(ACL_FORMAT_NCDHW, format);
  } else if (format == ACL_FORMAT_ND && size.size() == 4) {
    // 4 dim tensor must be NCHW, reflush base format
    return std::make_tuple(ACL_FORMAT_NCHW, ACL_FORMAT_NCHW);
  } else {
    if (FormatHelper::GetBaseFormat(format) == ACL_FORMAT_NCDHW) {
      // scence: Dimensionality reduction: NCDHW->NCHW, for example: max/min
      // NOTE(NPU Dimensionality reduction)
      if (size.size() == 4) {
        return std::make_tuple(ACL_FORMAT_NCHW, ACL_FORMAT_NCHW);
      }
    }
  }
  return std::make_tuple(FormatHelper::GetBaseFormat(format), format);
}

aclFormat InferFormat::GuessBaseFormat(const IntArrayRef& size) {
  if (size.size() == 5) {
    return ACL_FORMAT_NCDHW;
  } else if (size.size() == 4) {
    return ACL_FORMAT_NCHW;
  }
  return ACL_FORMAT_ND;
}

aclFormat InferFormat::GuessStorageFormat(const IntArrayRef& size, aclFormat format) {
  int64_t dim = size.size();
  aclFormat baseFormat = FormatHelper::GetBaseFormat(format);
  bool isBaseFormat = FormatHelper::IsBaseFormatType(format);
  // if base format and tensor size is not match, we should reflush them
  if ((isBaseFormat) && (baseFormat == ACL_FORMAT_NCDHW)) {
    // scence1: Dimensionality reduction: NCDHW->NCHW, for example: max/min
    // scence2: view, as_strided
    // NOTE(NPU Dimensionality reduction)
    if (dim == 4) {
      return ACL_FORMAT_NCHW;
    } else if (dim == 5) {
      return ACL_FORMAT_NCDHW;
    } else {
      return ACL_FORMAT_ND;
    }
  } else if (format == ACL_FORMAT_NCHW && dim != 4) {
      return ACL_FORMAT_ND;
  }
  return format;
}

FormatShape InferFormat::GuessStorageSizeWhenConvertFormat(const Tensor& tensor) {
  auto format = FormatHelper::GetFormat(tensor);
  auto size = tensor.storage().unsafeGetStorageImpl()->npu_desc_.base_sizes_;
  // TransData: ND->NZ, ND size < 2, we can expand dimension to 2, the storage have no effect.
  // now, only ND->NZ and NZ->ND will call transdata， so we no need to check other format.
  if ((size.size() < 2) && format == ACL_FORMAT_ND) {
    do {
        size.emplace_back(1);
    } while(size.size() < 2);
  }
  return FormatHelper::GetStorageSizes(format, size);
}

bool InferFormat::IsDefiniteTensorWhenMetaDataChanges(const Tensor& tensor, const IntArrayRef& size) {
  auto baseformat = FormatHelper::GetBaseFormat(tensor);
  if (baseformat == ACL_FORMAT_NCHW && size.size() >= 5) {
    return true;
  }
  if (baseformat == ACL_FORMAT_NCDHW && size.size() != 5) {
    return true;
  }
  return false;
}

} // namespace npu
} // namespace native
} // namespace at