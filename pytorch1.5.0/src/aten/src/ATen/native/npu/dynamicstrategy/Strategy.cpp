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
#include <ATen/native/npu/dynamicstrategy/Strategy.h>

namespace at {
namespace native {
namespace npu {

void DescStrategyBase::CreateDefaultDescInfo(const aclTensorDesc** descs,
    int num,
    int64_t* storageDims,
    aclFormat* storageFormats,
    SmallVector<FormatShape, N>& inShape,
    SmallVector<FormatShape, N>& inStorageShape) {
   
  // create input shape
  for (int64_t i = 0; i < num; ++i) {
    aclTensorDesc* desc = const_cast<aclTensorDesc*>(descs[i]);
    int64_t dim = (int64_t)aclGetTensorDescNumDims(desc);
    dim = (dim == 0) ? 1 : dim;
    int64_t storageDim = (storageDims[i] == 0) ? 1 : storageDims[i];
   
    FormatShape shape(dim, -1);
    FormatShape storageShape(storageDim, -1);
    aclFormat storageFormat = storageFormats[i];
    if (storageFormat == ACL_FORMAT_NC1HWC0) {
      storageShape[4] = 16;
    }

    inShape.emplace_back(shape);
    inStorageShape.emplace_back(storageShape);
  }
}

} // namespace npu
} // namespace native
} // namespace at