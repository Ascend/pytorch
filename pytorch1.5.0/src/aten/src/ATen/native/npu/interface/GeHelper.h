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

#ifndef __NATIVE_NPU_INTERFACE_GE_HELPER__
#define __NATIVE_NPU_INTERFACE_GE_HELPER__

#include <third_party/acl/inc/graph/tensor.h> // TensorDesc
#include <third_party/acl/inc/graph/types.h> // Format
#include <third_party/acl/inc/acl/acl_base.h> // aclTensorDesc

namespace at {
namespace native {
namespace npu {

/**
  This class is used to transform acl interface to ge interface
  e.g aclTensorDesc vs ge::TensorDesc
  */
class GeHelper {
public:
  /**
    This API is used to transform aclTensorDesc to ge::TensorDesc
    */
  static ge::TensorDesc Convert(const aclTensorDesc* desc);
private:
  static ge::DataType Convert(aclDataType dataType);
  static ge::Format Convert(aclFormat format);
private:
  static std::vector<int64_t> GetTensorDescDims(const aclTensorDesc* desc);
};

} // namespace npu
} // namespace native
} // namespace at


#endif // __NATIVE_NPU_INTERFACE_GE_HELPER__