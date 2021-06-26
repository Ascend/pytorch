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


#include "AclInterface.h"
#include "c10/npu/register/FunctionLoader.h"
#include "c10/util/Exception.h"
#include <iostream>

namespace c10 {
namespace npu {
namespace acl {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libascendcl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libascendcl, funcName)

REGISTER_LIBRARY(libascendcl)
LOAD_FUNCTION(aclGetRecentErrMsg)
LOAD_FUNCTION(aclrtCreateEventWithFlag)

const char *AclGetErrMsg()
{
  typedef const char *(*aclGetErrMsg)();
  static aclGetErrMsg func = nullptr;
  if (func == nullptr) {
    func = (aclGetErrMsg)GET_FUNC(aclGetRecentErrMsg);
  }
  if (func != nullptr) {
    return func();
  }
  return "";
}

aclError AclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) {
  typedef aclError(*AclrtCreateEventWithFlagFunc)(aclrtEvent*, uint32_t);
  static AclrtCreateEventWithFlagFunc func = nullptr;
  if (func == nullptr) {
    func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventWithFlag);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtCreateEventWithFlag");
  return func(event, flag);
}

} // namespace acl
} // namespace npu
} // namespace c10
