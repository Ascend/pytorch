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
LOAD_FUNCTION(aclrtQueryEventWaitStatus)
LOAD_FUNCTION(aclprofCreateStepInfo)
LOAD_FUNCTION(aclprofGetStepTimestamp)
LOAD_FUNCTION(aclprofDestroyStepInfo)

aclprofStepInfoPtr init_stepinfo(){
  typedef aclprofStepInfoPtr(*npdInitFunc)();
  static npdInitFunc func = nullptr;
  if(func == nullptr){
      func = (npdInitFunc)GET_FUNC(aclprofCreateStepInfo);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofCreateStepInfo");
  auto ret = func();
  return ret;
}

NpdStatus destroy_stepinfo(aclprofStepInfoPtr stepInfo){
  typedef NpdStatus(*npdDestroyFunc)(aclprofStepInfoPtr);
  static npdDestroyFunc func = nullptr;
  if(func == nullptr){
      func = (npdDestroyFunc)GET_FUNC(aclprofDestroyStepInfo);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofDestroyStepInfo");
  auto ret = func(stepInfo);
  return ret;
}

NpdStatus start_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream){
  typedef NpdStatus(*npdStartProfiling)(aclprofStepInfoPtr, aclprofStepTag, aclrtStream);
  static npdStartProfiling func = nullptr;
  if(func == nullptr){
      func = (npdStartProfiling)GET_FUNC(aclprofGetStepTimestamp);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofGetStepTimestamp");
  auto ret = func(stepInfo, stepTag, stream);
  return ret;
}

NpdStatus stop_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream){
  typedef NpdStatus(*npdStopProfiling)(aclprofStepInfoPtr, aclprofStepTag, aclrtStream);
  static npdStopProfiling func = nullptr;
  if(func == nullptr){
      func = (npdStopProfiling)GET_FUNC(aclprofGetStepTimestamp);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofGetStepTimestamp");
  auto ret = func(stepInfo, stepTag, stream);
  return ret;
}

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

aclError AclQueryEventStatus(aclrtEvent event, aclrtEventWaitStatus *waitStatus, aclrtEventStatus *recordStatus)
{
  typedef aclError (*aclQueryEventWaitStatus)(aclrtEvent event, aclrtEventWaitStatus *status);
  static aclQueryEventWaitStatus func = nullptr;
  if (func == nullptr) {
    func = (aclQueryEventWaitStatus)GET_FUNC(aclrtQueryEventWaitStatus);
  }
  if (func != nullptr) {
    return func(event, waitStatus);
  } else {
    return aclrtQueryEvent(event, recordStatus);
  }
}
} // namespace acl
} // namespace npu
} // namespace c10
