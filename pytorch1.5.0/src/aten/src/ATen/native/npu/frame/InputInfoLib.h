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

#ifndef __NATIVE_NPU_UTILS_MAP__
#define __NATIVE_NPU_UTILS_MAP__

#include <c10/util/SmallVector.h>
#include <c10/npu/NPUStream.h>
#include <ATen/native/npu/utils/NpuUtils.h>
#include <ATen/native/npu/frame/NPUDefine.h>
#include <ATen/native/npu/frame/FormatHelper.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include <unordered_map>
#include <string>
#include <set>

namespace at {
namespace native {
namespace npu {

struct DynamicCompileShape {
  SmallVector<FormatShape, N> inputShape;
  SmallVector<FormatShape, N> inputStorageShape;
  SmallVector<FormatShape, N> outputShape;
  SmallVector<FormatShape, N> outputStorageShape;
};

using namespace std;

class InputInfoLib {

 public:
  InputInfoLib(){};

  ~InputInfoLib(){};

 public:
  // if key exists in the Lib, return true, else return false. 
  bool IsExistStaticKey(const string& key);
  
  // if key already exists in the Lib, return false, else insert key and return true.
  bool InsertStaticKey(const string& key);

  // if key already exists in the Lib, return false, else insert key and return true.
  bool InsertDynamicKey(const string& key, bool value);

  // if key not exists in the Lib, return false, else update key and return true.
  bool UpdateDynamicKey(const string& key, bool value);

  // retrun value of the key
  bool GetDynamicValue(const string& key);

  // if key exists in the Lib, return true, else return false
  bool IsExistDynamicKey(const string& key);

  bool CheckConstInput(int64_t index, CONST_PARAMS& constParams);
  
  // create a static key according to aclInputDesc.
  string CreateStaticKey(ExecuteParas& params, bool& hasScalar);

  // create a dynamic key according to aclInputDesc.
  string CreateDynamicKey(ExecuteParas& params, DynamicCompileShape dynamicCompileShape);

  string ShapeMaker(int64_t inputId, ExecuteParas& params, DynamicCompileShape dynamicCompileShape);

  string ConstShapeMaker(int64_t inputId, ExecuteParas& params);
  
  int64_t constSign = -1;
 private:
  set<string> staticLib;

  unordered_map<string, bool> dynamicLib;
};

} // namespace npu
} // namespace native
} // namespace at
#endif