#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

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
LOAD_FUNCTION(aclprofReportStamp)
LOAD_FUNCTION(aclprofSetStampTagName)
LOAD_FUNCTION(aclprofSetCategoryName)
LOAD_FUNCTION(aclprofSetStampCategory)
LOAD_FUNCTION(aclprofSetStampPayload)
LOAD_FUNCTION(aclprofSetStampTraceMessage)
LOAD_FUNCTION(aclprofSetStampCallStack)
LOAD_FUNCTION(aclprofMsproftxSwitch)
LOAD_FUNCTION(aclprofMark)
LOAD_FUNCTION(aclprofPush)
LOAD_FUNCTION(aclprofPop)
LOAD_FUNCTION(aclprofRangeStart)
LOAD_FUNCTION(aclprofRangeStop)
LOAD_FUNCTION(aclprofSetConfig)


void *AclprofCreateStamp() {
    typedef void*(*AclprofCreateStampFunc)();
    static AclprofCreateStampFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofCreateStampFunc)GET_FUNC(aclprofCreateStamp);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofCreateStamp", PROF_ERROR(ErrCode::NOT_FOUND));
    return  func();
}

void AclprofDestroyStamp(void *stamp) {
    typedef void(*AclprofDestroyStampFunc)(void *);
    static AclprofDestroyStampFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofDestroyStampFunc)GET_FUNC(aclprofDestroyStamp);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofDestroyStamp", PROF_ERROR(ErrCode::NOT_FOUND));
    func(stamp);
}

aclError AclprofSetStampTagName(void *stamp, const char *tagName, uint16_t len) {
    typedef aclError(*AclprofSetStampTagNameFunc)(void *, const char *, uint16_t);
    static AclprofSetStampTagNameFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetStampTagNameFunc)GET_FUNC(aclprofSetStampTagName);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetStampTagName", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(stamp, tagName, len);
}

aclError AclprofSetCategoryName(uint32_t category, const char *categoryName) {
    typedef aclError(*AclprofSetCategoryNameFunc)(uint32_t, const char *);
    static AclprofSetCategoryNameFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetCategoryNameFunc)GET_FUNC(aclprofSetCategoryName);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetCategoryName", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(category, categoryName);
}

aclError AclprofSetStampCategory(void *stamp, uint32_t category) {
    typedef aclError(*AclprofSetStampCategoryFunc)(void *, uint32_t);
    static AclprofSetStampCategoryFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetStampCategoryFunc)GET_FUNC(aclprofSetStampCategory);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetStampCategory", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(stamp, category);
}

aclError AclprofSetStampPayload(void *stamp, const int32_t type, void *value) {
    typedef aclError(*AclprofSetStampPayloadFunc)(void *, const int32_t, void *);
    static AclprofSetStampPayloadFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetStampPayloadFunc)GET_FUNC(aclprofSetStampPayload);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetStampPayload", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(stamp, type, value);
}

aclError AclprofSetStampTraceMessage(void *stamp, const char *msg, uint32_t msgLen) {
    typedef aclError(*AclprofSetStampTraceMessageFunc)(void *, const char *, uint32_t);
    static AclprofSetStampTraceMessageFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetStampTraceMessageFunc)GET_FUNC(aclprofSetStampTraceMessage);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetStampTraceMessage", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(stamp, msg, msgLen);
}

aclError AclprofSetStampCallStack(void *stamp, const char *callStack, uint32_t len) {
    typedef aclError(*AclprofSetStampCallStackFunc)(void *, const char *, uint32_t);
    static AclprofSetStampCallStackFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetStampCallStackFunc)GET_FUNC(aclprofSetStampCallStack);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetStampCallStack", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(stamp, callStack, len);
}

aclError AclprofMsproftxSwitch(bool isOpen) {
    typedef aclError(*AclprofMsproftxSwitchFunc)(bool);
    static AclprofMsproftxSwitchFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofMsproftxSwitchFunc)GET_FUNC(aclprofMsproftxSwitch);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofMsproftxSwitch", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(isOpen);
}

aclError AclprofMark(void *stamp) {
    typedef aclError(*AclprofMarkFunc)(void *);
    static AclprofMarkFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofMarkFunc)GET_FUNC(aclprofMark);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofMark", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(stamp);
}

aclError AclprofPush(void *stamp) {
    typedef aclError(*AclprofPushFunc)(void *);
    static AclprofPushFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofPushFunc)GET_FUNC(aclprofPush);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofPush", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(stamp);
}

aclError AclprofPop() {
    typedef aclError(*AclprofPopFunc)();
    static AclprofPopFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofPopFunc)GET_FUNC(aclprofPop);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofPop", PROF_ERROR(ErrCode::NOT_FOUND));
    return func();
}

aclError AclprofRangeStart(void *stamp, uint32_t *rangeId) {
    typedef aclError(*AclprofRangeStartFunc)(void *, uint32_t *);
    static AclprofRangeStartFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofRangeStartFunc)GET_FUNC(aclprofRangeStart);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofRangeStart", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(stamp, rangeId);
}

aclError AclprofRangeStop(uint32_t rangeId) {
    typedef aclError(*AclprofRangeStopFunc)(uint32_t);
    static AclprofRangeStopFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofRangeStopFunc)GET_FUNC(aclprofRangeStop);
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofRangeStop", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(rangeId);
}

aclError AclprofReportStamp(const char *tag, unsigned int tagLen,
                            unsigned char *data, unsigned int dataLen) {
    typedef aclError(*AclprofReportStampFunc)(const char *, unsigned int, unsigned char *, unsigned int);
    static AclprofReportStampFunc func = (AclprofReportStampFunc)GET_FUNC(aclprofReportStamp);
    TORCH_CHECK(func, "Failed to find function ", "aclprofReportStamp", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(tag, tagLen, data, dataLen);
}

bool CheckInterfaceReportStamp() {
    typedef aclError(*AclprofReportStampFunc)(const char *, unsigned int, unsigned char *, unsigned int);
    static AclprofReportStampFunc func = (AclprofReportStampFunc)GET_FUNC(aclprofReportStamp);
    return (func == nullptr) ? false : true;
}

aclError AclprofSetConfig(aclprofConfigType configType, const char* config, size_t configLength) {
    typedef aclError(*AclprofSetConfigFunc)(aclprofConfigType, const char *, size_t);
    static AclprofSetConfigFunc func = nullptr;
    if (func == nullptr) {
        func = (AclprofSetConfigFunc)GET_FUNC(aclprofSetConfig);
        if (func == nullptr) {
            return ACL_ERROR_PROF_MODULES_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclprofSetConfig", PROF_ERROR(ErrCode::NOT_FOUND));
    return func(configType, config, configLength);
}
}
}
