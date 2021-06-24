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

#ifndef __DYNAMIC_SHAPE_UTIL__
#define __DYNAMIC_SHAPE_UTIL__

#include <c10/core/thread_pool.h>
#include <c10/npu/NPUStream.h>
#include <c10/npu/npu_log.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include <third_party/acl/inc/acl/acl_op_compiler.h>
#include "ATen/native/npu/frame/InputInfoLib.h"
#include "ATen/native/npu/frame/LogUtil.h"
#include "ATen/native/npu/frame/DebugDynamic.h"
#include "ATen/native/npu/frame/NPUDefine.h"
#include <vector>
#include <unordered_set>

namespace at {
namespace native {
namespace npu {

class DynamicShapeUtil {
  friend void WaitDynamicCompileComplete();
  friend aclError DynamicRun(
      ExecuteParas& params,
      aclrtStream stream);
  friend void DynamicIncreaseSteps();
  
 private:
  DynamicShapeUtil();
  ~DynamicShapeUtil();

  static DynamicShapeUtil* GetInstance() {
    static DynamicShapeUtil instance;
    return &instance;
  };

  aclError Run(ExecuteParas& params, aclrtStream stream);

  aclError DynamicRun(ExecuteParas& params, aclrtStream stream);

  static void IncreaseSteps();
  bool CheckFirstStep();

  aclTensorDesc* CreatConstDesc(
      const size_t index, 
      const aclTensorDesc* desc,
      const int dims,
      const aclFormat format,
      const aclDataType dtype,
      const string compileName,
      CONST_PARAMS& constParams);

  void WaitThreadComplete();

  void CreateAclParamsDesc(
    const aclTensorDesc** inDescDynamic,
    int inNum,
    int64_t* inStorageDims,
    aclFormat* inStorageFormats,
    CONST_PARAMS& constParams,
    SmallVector<FormatShape, N> shape,
    SmallVector<FormatShape, N> storageShape,
    const aclTensorDesc** outDescDynamic);

  aclError CompileDynamic(ExecuteParas& cur_paras);

  ExecuteParas CreateCompileParams(
      ExecuteParas& params,
      DynamicCompileShape dynamicCompileShape);

  void StartThreadCompile(
      ExecuteParas params,
      const string key);

  int ExecuteDynamic(
      ExecuteParas& cur_paras,
      aclrtStream stream);

  void staticCompileAndExecute(
      ExecuteParas& cur_paras,
      const string key,
      aclrtStream stream);

  void DynamicCompileShapeMaker(ExecuteParas& params, 
    DynamicCompileShape& compileShape);

 private:
  bool isStaticType(const ExecuteParas& cur_paras);

private:
  static std::unordered_set<string> disableDynamicOp;
  std::shared_ptr<TaskThreadPool> thread_pool_;
  InputInfoLib dynamicMap;
  DynamicLogUtil logUtil;
  static long long int steps_;
  bool isDynamicOnly;
};

aclError DynamicRun(ExecuteParas& params, aclrtStream stream);

C10_API void WaitDynamicCompileComplete();

C10_API void DynamicIncreaseSteps();

} // namespace npu
} // namespace native
} // namespace at

#endif