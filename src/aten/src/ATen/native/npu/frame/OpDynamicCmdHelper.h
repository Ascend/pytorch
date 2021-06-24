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

#ifndef __NATIVE_NPU_UTILS_OP_DYNAMIC_COMMAND_HELPER__
#define __NATIVE_NPU_UTILS_OP_DYNAMIC_COMMAND_HELPER__

#include <c10/util/SmallVector.h>
#include <ATen/ATen.h>
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/frame/NPUDefine.h"
#include <third_party/acl/inc/acl/acl.h>
#include <third_party/acl/inc/acl/acl_base.h>

namespace at {
namespace native {
namespace npu {

typedef enum {
    FIXED_NONE,
    FIXED_ALL,
    FIXED_C, // Only Support Format of NCHW and NC1HWC0
    FIXED_CONST_VALUE, // Op need Recompile When the Value of ConstInput Changed
    FIXED_CONST_DIM,  // Op need Recompile When the Dim of ConstInput Changed
} shapeStrage;

// covert pytorch tensor to acl tensor.
class OpDynamicCmdHelper {
public:  
  static aclTensorDesc* CovertToAclInputDynamicCompileDesc(Tensor tensor,
      c10::optional<Tensor> cpu_tensor,
      string& dynamicKey,
      string descName = "",
      string forceDataType = "",
      shapeStrage strage = FIXED_NONE);

  static aclTensorDesc* CovertToAclInputConstDynamicCompileDesc(Tensor tensor,
      c10::optional<Tensor> cpu_tensor,
      string& dynamicKey,
      string descName = "",
      string forceDataType = "",
      shapeStrage strage = FIXED_CONST_VALUE);

  static aclTensorDesc* CovertToAclOutputDynamicCompileDesc(const Tensor* tensorPtr, 
      string forceDataType,
      shapeStrage strage = FIXED_NONE, 
      bool isDimZeroToOne = true);

  static std::tuple<SmallVector<int64_t, N>, SmallVector<int64_t, N>, SmallVector<int64_t, N>>
  CreateDynamicCompilelDims(NPUStorageDesc npuDesc, shapeStrage strage, bool isDimZeroToOne);

  static void ShapeStrageMaker(
      NPUStorageDesc npuDesc,
      SmallVector<int64_t, N>& shape,
      SmallVector<int64_t, N>& storageShape,
      shapeStrage strage);

  static string CreateConstShapeKey(
    SmallVector<int64_t, N> dimList,
    shapeStrage strage);

  static string CreateShapeKey(
    SmallVector<int64_t, N> shape,
    SmallVector<int64_t, N> storageShape);

  static std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat, aclTensorDesc*>
  CovertNPUTensorWithZeroDimToDynamicAclInput(const Tensor& tensor, string descName);

  static std::tuple<aclTensorDesc*, aclDataBuffer*, int64_t, aclFormat, aclTensorDesc*>
  CovertTensorWithZeroDimToDynamicAclInput(const Tensor& tensor, ScalarType type);

  static const aclTensorDesc** ConvertTensorWithZeroDimToOneDim(const aclTensorDesc** descs, int num);

  static std::tuple<string, int, const aclTensorDesc**, int, const aclTensorDesc**, const aclopAttr*>
  CreateDynamicCompileParams(ExecuteParas& params);

  static std::tuple<string, int, const aclTensorDesc**, const aclDataBuffer**, int, const aclTensorDesc**, aclDataBuffer**, const aclopAttr*>
  CreateDynamicRunParams(ExecuteParas& params);
}; // class OpCommandImpl

} // npu
} // native
} // at

#endif