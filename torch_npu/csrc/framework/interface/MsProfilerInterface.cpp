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

#include <c10/npu/NPUException.h>
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/register/FunctionLoader.h"

namespace at_npu {
namespace native {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libmsprofiler, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libmsprofiler, funcName)


REGISTER_LIBRARY(libmsprofiler)
LOAD_FUNCTION(aclprofCreateStamp)
LOAD_FUNCTION(aclprofDestroyStamp)
LOAD_FUNCTION(aclprofSetCategoryName)
LOAD_FUNCTION(aclprofSetStampCategory)
LOAD_FUNCTION(aclprofSetStampPayload)
LOAD_FUNCTION(aclprofSetStampTraceMessage)
LOAD_FUNCTION(aclprofMsproftxSwitch)
LOAD_FUNCTION(aclprofMark)
LOAD_FUNCTION(aclprofPush)
LOAD_FUNCTION(aclprofPop)
LOAD_FUNCTION(aclprofRangeStart)
LOAD_FUNCTION(aclprofRangeStop)


void *AclprofCreateStamp() {
    typedef void*(*AclprofCreateStampFunc)();
    static AclprofCreateStampFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofCreateStampFunc)GET_FUNC(aclprofCreateStamp);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofCreateStamp");
    return  func();
}

void AclprofDestroyStamp(void *stamp) {
    typedef void(*AclprofDestroyStampFunc)(void *);
    static AclprofDestroyStampFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofDestroyStampFunc)GET_FUNC(aclprofDestroyStamp);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofDestroyStamp");
    func(stamp);
}
aclError AclprofSetCategoryName(uint32_t category, const char *categoryName) {
    typedef aclError(*AclprofSetCategoryNameFunc)(uint32_t, const char *);
    static AclprofSetCategoryNameFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetCategoryNameFunc)GET_FUNC(aclprofSetCategoryName);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetCategoryName");
    return func(category, categoryName);  
}

aclError AclprofSetStampCategory(void *stamp, uint32_t category) {
    typedef aclError(*AclprofSetStampCategoryFunc)(void *, uint32_t);
    static AclprofSetStampCategoryFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetStampCategoryFunc)GET_FUNC(aclprofSetStampCategory);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetStampCategory");
    return func(stamp, category);      
}

aclError AclprofSetStampPayload(void *stamp, const int32_t type, void *value) {
    typedef aclError(*AclprofSetStampPayloadFunc)(void *, const int32_t, void *);
    static AclprofSetStampPayloadFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetStampPayloadFunc)GET_FUNC(aclprofSetStampPayload);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetStampPayload");
    return func(stamp, type, value);      
}

aclError AclprofSetStampTraceMessage(void *stamp, const char *msg, uint32_t msgLen) {
    typedef aclError(*AclprofSetStampTraceMessageFunc)(void *, const char *, uint32_t);
    static AclprofSetStampTraceMessageFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetStampTraceMessageFunc)GET_FUNC(aclprofSetStampTraceMessage);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetStampTraceMessage");
    return func(stamp, msg, msgLen);  
}

aclError AclprofMsproftxSwitch(bool isOpen) {
    typedef aclError(*AclprofMsproftxSwitchFunc)(bool);
    static AclprofMsproftxSwitchFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofMsproftxSwitchFunc)GET_FUNC(aclprofMsproftxSwitch);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofMsproftxSwitch");
    return func(isOpen);    
}

aclError AclprofMark(void *stamp) {
    typedef aclError(*AclprofMarkFunc)(void *);
    static AclprofMarkFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofMarkFunc)GET_FUNC(aclprofMark);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofMark");
    return func(stamp);    
}

aclError AclprofPush(void *stamp) {
    typedef aclError(*AclprofPushFunc)(void *);
    static AclprofPushFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofPushFunc)GET_FUNC(aclprofPush);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofPush");
    return func(stamp);    
}

aclError AclprofPop() {
    typedef aclError(*AclprofPopFunc)();
    static AclprofPopFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofPopFunc)GET_FUNC(aclprofPop);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofPop");
    return func();    
}

aclError AclprofRangeStart(void *stamp, uint32_t *rangeId) {
    typedef aclError(*AclprofRangeStartFunc)(void *, uint32_t *);
    static AclprofRangeStartFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofRangeStartFunc)GET_FUNC(aclprofRangeStart);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofRangeStart");
    return func(stamp, rangeId);    
}

aclError AclprofRangeStop(uint32_t rangeId) {
    typedef aclError(*AclprofRangeStopFunc)(uint32_t);
    static AclprofRangeStopFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofRangeStopFunc)GET_FUNC(aclprofRangeStop);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofRangeStop");
    return func(rangeId);   
}

}
}