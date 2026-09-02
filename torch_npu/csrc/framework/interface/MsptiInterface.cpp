#include "torch_npu/csrc/framework/interface/MsptiInterface.h"

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"

namespace at_npu {
namespace native {

#undef TORCH_NPU_LOAD_FUNC
#define TORCH_NPU_LOAD_FUNC(funcName) TORCH_NPU_REGISTER_FUNCTION(libmspti, funcName)

#undef TORCH_NPU_GET_FUNC

#define TORCH_NPU_GET_FUNC(funcName) TORCH_NPU_GET_FUNCTION(libmspti, funcName)

TORCH_NPU_REGISTER_LIBRARY(libmspti)
TORCH_NPU_LOAD_FUNC(msptiActivityIsEnabled)
TORCH_NPU_LOAD_FUNC(msptiActivityEnable)
TORCH_NPU_LOAD_FUNC(msptiActivityDisable)
TORCH_NPU_LOAD_FUNC(msptiActivityRegisterCallbacks)
TORCH_NPU_LOAD_FUNC(msptiActivityFlushAll)
TORCH_NPU_LOAD_FUNC(msptiActivityGetNextRecord)
TORCH_NPU_LOAD_FUNC(msptiActivityPushExternalCorrelationId)
TORCH_NPU_LOAD_FUNC(msptiActivityPopExternalCorrelationId)
TORCH_NPU_LOAD_FUNC(msptiSubscribe)
TORCH_NPU_LOAD_FUNC(msptiUnsubscribe)

static bool IsSupportMsptiFuncImpl() {
  static auto checkSupport = []() -> bool {
    char* path = std::getenv("ASCEND_HOME_PATH");
    if (path != nullptr) {
      std::string soPath = std::string(path) + "/lib64/libmspti.so";
      soPath = torch_npu::toolkit::profiler::Utils::RealPath(soPath);
      return !soPath.empty();
    }
    return false;
  };
  return checkSupport();
}

bool IsSupportMsptiFunc() {
  static bool isSupport = IsSupportMsptiFuncImpl();
  return isSupport;
}

bool MsptiActivityIsEnabled(msptiActivityKind kind) {
  using MsptiActivityIsEnabledFunc = bool (*)(msptiActivityKind);
  static MsptiActivityIsEnabledFunc func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (MsptiActivityIsEnabledFunc)TORCH_NPU_GET_FUNC(msptiActivityIsEnabled);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiActivityIsEnabled");
      noFuncFlag = true;
      return false;
    }
  }
  return func(kind);
}

bool MsptiActivityEnable(msptiActivityKind kind) {
  using Func = msptiResult (*)(msptiActivityKind);
  static Func func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (Func)TORCH_NPU_GET_FUNC(msptiActivityEnable);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiActivityEnable");
      noFuncFlag = true;
      return false;
    }
  }
  return func(kind) == MSPTI_SUCCESS;
}

bool MsptiActivityDisable(msptiActivityKind kind) {
  using Func = msptiResult (*)(msptiActivityKind);
  static Func func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (Func)TORCH_NPU_GET_FUNC(msptiActivityDisable);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiActivityDisable");
      noFuncFlag = true;
      return false;
    }
  }
  return func(kind) == MSPTI_SUCCESS;
}

bool MsptiActivityRegisterCallbacks(
    msptiBuffersCallbackRequestFunc funcBufferRequested,
    msptiBuffersCallbackCompleteFunc funcBufferCompleted) {
  using Func = msptiResult (*)(msptiBuffersCallbackRequestFunc, msptiBuffersCallbackCompleteFunc);
  static Func func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (Func)TORCH_NPU_GET_FUNC(msptiActivityRegisterCallbacks);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiActivityRegisterCallbacks");
      noFuncFlag = true;
      return false;
    }
  }
  return func(funcBufferRequested, funcBufferCompleted) == MSPTI_SUCCESS;
}

bool MsptiActivityFlushAll(uint32_t flag) {
  using Func = msptiResult (*)(uint32_t);
  static Func func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (Func)TORCH_NPU_GET_FUNC(msptiActivityFlushAll);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiActivityFlushAll");
      noFuncFlag = true;
      return false;
    }
  }
  return func(flag) == MSPTI_SUCCESS;
}

bool MsptiActivityGetNextRecord(uint8_t* buffer, size_t validBufferSizeBytes, msptiActivity** record) {
  using Func = msptiResult (*)(uint8_t*, size_t, msptiActivity**);
  static Func func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (Func)TORCH_NPU_GET_FUNC(msptiActivityGetNextRecord);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiActivityGetNextRecord");
      noFuncFlag = true;
      return false;
    }
  }
  return func(buffer, validBufferSizeBytes, record) == MSPTI_SUCCESS;
}

bool MsptiActivityPushExternalCorrelationId(msptiExternalCorrelationKind kind, uint64_t id) {
  using Func = msptiResult (*)(msptiExternalCorrelationKind, uint64_t);
  static Func func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (Func)TORCH_NPU_GET_FUNC(msptiActivityPushExternalCorrelationId);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiActivityPushExternalCorrelationId");
      noFuncFlag = true;
      return false;
    }
  }
  return func(kind, id) == MSPTI_SUCCESS;
}

bool MsptiActivityPopExternalCorrelationId(msptiExternalCorrelationKind kind, uint64_t* lastId) {
  using Func = msptiResult (*)(msptiExternalCorrelationKind, uint64_t*);
  static Func func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (Func)TORCH_NPU_GET_FUNC(msptiActivityPopExternalCorrelationId);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiActivityPopExternalCorrelationId");
      noFuncFlag = true;
      return false;
    }
  }
  return func(kind, lastId) == MSPTI_SUCCESS;
}

bool MsptiSubscribe(msptiSubscriberHandle* subscriber) {
  using Func = msptiResult (*)(msptiSubscriberHandle*, msptiCallbackFunc, void*);
  static Func func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (Func)TORCH_NPU_GET_FUNC(msptiSubscribe);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiSubscribe");
      noFuncFlag = true;
      return false;
    }
  }
  return func(subscriber, nullptr, nullptr) == MSPTI_SUCCESS;
}

bool MsptiUnsubscribe(msptiSubscriberHandle subscriber) {
  using Func = msptiResult (*)(msptiSubscriberHandle);
  static Func func = nullptr;
  static bool noFuncFlag = false;
  if (noFuncFlag) {
    return false;
  }
  if (func == nullptr) {
    func = (Func)TORCH_NPU_GET_FUNC(msptiUnsubscribe);
    if (func == nullptr) {
      ASCEND_LOGW("Failed to get func msptiUnsubscribe");
      noFuncFlag = true;
      return false;
    }
  }
  return func(subscriber) == MSPTI_SUCCESS;
}

} // namespace native
} // namespace at_npu
