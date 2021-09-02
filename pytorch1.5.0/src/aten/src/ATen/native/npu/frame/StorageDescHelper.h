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

#ifndef __NATIVE_NPU_UTILS_STORAGE_DESC_HELPER__
#define __NATIVE_NPU_UTILS_STORAGE_DESC_HELPER__

#include <ATen/ATen.h>
#include <ATen/native/npu/utils/NPUDefinition.h>

namespace at {
namespace native {
namespace npu {

class StorageDescHelper {
public:
  // Get Part
  // sizes, strides in StorageDesc are same as those in MetaData
  static bool MetaDataAreMatch(const Tensor* tensor);
  // storage offset are match, the npu only support offset == 0
  static bool OffsetAreMatch(const Tensor* tensor);

  // helper function of transdata op.
  static bool IsSameDesc(const NPUStorageDesc& a, const NPUStorageDesc& b);
  static bool IsSameDesc(const Tensor& a, const Tensor& b);

  // calculate storage size need by npu memory
  static int64_t GetMemorySize(const Tensor& dst);
  static int64_t GetMemorySize(IntArrayRef size, aclFormat format);
  // Calculate the valid memory size of the tensor, because of view operator and so on.
  static int64_t GetValidMemorySize(const Tensor& tensor);

  // Set Part
  // StorageDesc Init/Set
  static void SetDesc(Tensor& dst);
  static void SetDesc(Tensor& dst, IntArrayRef size, IntArrayRef strides);
  static void SetDesc(Tensor& dst, IntArrayRef size, IntArrayRef strides, aclFormat format);

  static void CopyDesc(Tensor& dst, const Tensor& src);
  static void CopyDesc(Tensor& dst, const Storage& src);
  static void CopyDesc(const Tensor& dst, const NPUStorageDesc& src_desc);
  // 
  static void UpdateDesc(NPUStorageDesc& npuDesc, IntArrayRef& new_size);

  static FormatShape ComputeStrideFromShape(const FormatShape& shape);

  // TODO(ascend): need to remove later
  static void ReflushDescBySelf(const Tensor& src);
private:
  // Get Part
  static bool IsSameSize(SmallVector<int64_t,5> a, IntArrayRef b);
  static int64_t GetMemorySize(const NPUStorageDesc& dst);
  // Set Part
  static NPUStorageDesc SetDesc();
  static NPUStorageDesc SetDesc(IntArrayRef size, IntArrayRef strides);
  static NPUStorageDesc SetDesc(IntArrayRef size, IntArrayRef strides, aclFormat format);
};

} // namespace npu
} // namespace native
} // namespace at

#endif