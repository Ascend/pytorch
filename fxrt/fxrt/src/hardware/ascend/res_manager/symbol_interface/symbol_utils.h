#ifndef FXRT_SRC_HARDWARE_ASCEND_SYMBOL_UTILS_H_
#define FXRT_SRC_HARDWARE_ASCEND_SYMBOL_UTILS_H_
#include <string>
#include "common/common.h"
#include "acl/acl.h"
#include "common/visible.h"

#ifndef ACL_ERROR_RT_DEVICE_MEM_ERROR
#define ACL_ERROR_RT_DEVICE_MEM_ERROR 507053
#endif
#ifndef ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR
#define ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR 507054
#endif
#ifndef ACL_ERROR_RT_COMM_OP_RETRY_FAIL
#define ACL_ERROR_RT_COMM_OP_RETRY_FAIL 507904
#endif
#ifndef ACL_ERROR_RT_DEVICE_TASK_ABORT
#define ACL_ERROR_RT_DEVICE_TASK_ABORT 107022
#endif
const int thread_level = 0;

template <typename Function, typename... Args>
auto RunAscendApi(Function f, int line, const char* callF, const char* funcName, Args... args) {
  if (f == nullptr) {
    RT_GLOG(ERROR) << funcName << " is null.";
  }

  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f), Args...>, int>) {
    auto ret = f(args...);
    return ret;
  } else {
    return f(args...);
  }
}

template <typename Function>
auto RunAscendApi(Function f, int line, const char* callF, const char* funcName) {
  if (f == nullptr) {
    RT_GLOG(ERROR) << funcName << " is null.";
  }
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f)>, int>) {
    auto ret = f();
    return ret;
  } else {
    return f();
  }
}

template <typename Function>
bool HasAscendApi(Function f) {
  return f != nullptr;
}

namespace fxrt::device::ascend {

#define CALL_ASCEND_API(funcName, ...) \
  RunAscendApi(fxrt::device::ascend::funcName##_, __LINE__, __FUNCTION__, #funcName, ##__VA_ARGS__)

#define HAS_ASCEND_API(funcName) HasAscendApi(fxrt::device::ascend::funcName##_)

DA_API std::string GetAscendPath();
DA_API const char* GetAscendSocVersion();
void* GetLibHandler(const std::string& libPath, bool ifGlobal = false);
void LoadAscendApiSymbols();
void LoadSimulationApiSymbols();
} // namespace fxrt::device::ascend

#endif // FXRT_SRC_HARDWARE_ASCEND_SYMBOL_UTILS_H_
