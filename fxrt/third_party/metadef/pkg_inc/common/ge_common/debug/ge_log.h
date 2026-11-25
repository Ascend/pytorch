/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_GE_COMMON_DEBUG_GE_LOG_H_
#define INC_COMMON_GE_COMMON_DEBUG_GE_LOG_H_

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <sstream>

#include "base/err_msg.h"
#include "dlog_pub.h"
#include "common/ge_common/error_codes_define.h"

#ifdef __GNUC__
#include <unistd.h>
#include <sys/syscall.h>
#else
#include "mmpa/mmpa_api.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GE_MODULE_NAME static_cast<int32_t>(GE)
#define GE_MODULE_NAME_U16 static_cast<uint16_t>(GE)

// trace status of log
enum TraceStatus { TRACE_INIT = 0, TRACE_RUNNING, TRACE_WAITING, TRACE_STOP };

class GE_FUNC_VISIBILITY GeLog{public : static uint64_t GetTid(){
#ifdef __GNUC__
    const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
return tid;
}
}
;

inline bool IsLogEnable(const int32_t module_name, const int32_t log_level) {
  const int32_t enable = CheckLogLevel(module_name, log_level);
  // 1:enable, 0:disable
  return (enable == 1);
}

inline bool IsLogPrintStdout() {
  static int32_t stdout_flag = -1;
  if (stdout_flag == -1) {
    const char* env_ret = getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    const bool print_stdout = ((env_ret != nullptr) && (strcmp(env_ret, "1") == 0));
    stdout_flag = print_stdout ? 1 : 0;
  }
  return (stdout_flag == 1) ? true : false;
}

#define GELOGE(ERROR_CODE, fmt, ...)                        \
  do {                                                      \
    dlog_error(                                             \
        GE_MODULE_NAME,                                     \
        "%" PRIu64 " %s: ErrorNo: %" PRIuLEAST8 "(%s)" fmt, \
        GeLog::GetTid(),                                    \
        &__FUNCTION__[0U],                                  \
        (ERROR_CODE),                                       \
        ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()),         \
        ##__VA_ARGS__);                                     \
  } while (false)

#define GELOGW(fmt, ...)                                                                                 \
  do {                                                                                                   \
    dlog_warn(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GELOGI(fmt, ...)                                                                                 \
  do {                                                                                                   \
    dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GELOGD(fmt, ...)                                                                                  \
  do {                                                                                                    \
    dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GEEVENT(fmt, ...)                                                                                  \
  do {                                                                                                     \
    dlog_info(                                                                                             \
        static_cast<int32_t>(static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(GE_MODULE_NAME)), \
        "%" PRIu64 " %s:" fmt,                                                                             \
        GeLog::GetTid(),                                                                                   \
        &__FUNCTION__[0U],                                                                                 \
        ##__VA_ARGS__);                                                                                    \
    if (!IsLogPrintStdout()) {                                                                             \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
    }                                                                                                      \
  } while (false)

#define GERUNINFO(fmt, ...)                                                                                \
  do {                                                                                                     \
    dlog_info(                                                                                             \
        static_cast<int32_t>(static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(GE_MODULE_NAME)), \
        "%" PRIu64 " %s:" fmt,                                                                             \
        GeLog::GetTid(),                                                                                   \
        &__FUNCTION__[0U],                                                                                 \
        ##__VA_ARGS__);                                                                                    \
    if (!IsLogPrintStdout()) {                                                                             \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
    }                                                                                                      \
  } while (false)

#define GELOGT(VALUE, fmt, ...)                                                        \
  do {                                                                                 \
    constexpr const char_t* TraceStatStr[] = {"INIT", "RUNNING", "WAITING", "STOP"};   \
    constexpr int32_t idx = static_cast<int32_t>(VALUE);                               \
    auto v = TraceStatStr[idx];                                                        \
    dlog_info(                                                                         \
        (static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(GE_MODULE_NAME)), \
        "[status:%s]%" PRIu64 " %s:" fmt,                                              \
        v,                                                                             \
        GeLog::GetTid(),                                                               \
        &__FUNCTION__[0U],                                                             \
        ##__VA_ARGS__);                                                                \
  } while (false)

#define GE_LOG_ERROR(MOD_NAME, ERROR_CODE, fmt, ...)        \
  do {                                                      \
    dlog_error(                                             \
        (MOD_NAME),                                         \
        "%" PRIu64 " %s: ErrorNo: %" PRIuLEAST8 "(%s)" fmt, \
        GeLog::GetTid(),                                    \
        &__FUNCTION__[0U],                                  \
        (ERROR_CODE),                                       \
        ((GE_GET_ERRORNO_STR(ERROR_CODE)).c_str()),         \
        ##__VA_ARGS__);                                     \
  } while (false)

// print memory when it is greater than 1KB.
#define GE_PRINT_DYNAMIC_MEMORY(FUNC, PURPOSE, SIZE)                                                        \
  do {                                                                                                      \
    if (static_cast<size_t>(SIZE) > 1024UL) {                                                               \
      GELOGI("MallocMemory, func=%s, size=%zu, purpose=%s", (#FUNC), static_cast<size_t>(SIZE), (PURPOSE)); \
    }                                                                                                       \
  } while (false)

#define GELOG_DEPRECATED(option)                                         \
  do {                                                                   \
    std::cout << "[WARNING][GE] Option " << (option)                     \
              << " is deprecated and will be removed in future version." \
                 " Please do not configure this option in the future."   \
              << std::endl;                                              \
  } while (false)

#ifdef __cplusplus
}
#endif

#if !defined(__ANDROID__) && !defined(ANDROID)
#define DOMI_LOGE(fmt, ...) GE_LOG_ERROR(GE_MODULE_NAME, (ge::FAILED), fmt, ##__VA_ARGS__)
#else
#include <android/log.h>
#if defined(BUILD_VERSION_PERF)
#define DOMI_LOGE(fmt, ...)
#else
// The Android system has strict log control. Do not modify the log.
#define DOMI_LOGE(fmt, ...) \
  __android_log_print(ANDROID_LOG_ERROR, "NPU_FMK", "%s %s(%d)::" #fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif
#endif

// ge marco
#define GE_LOGI_IF(condition, ...) \
  if ((condition)) {               \
    GELOGI(__VA_ARGS__);           \
  }

#define GE_LOGW_IF(condition, ...) \
  if ((condition)) {               \
    GELOGW(__VA_ARGS__);           \
  }

#define GE_LOGE_IF(condition, ...)     \
  if ((condition)) {                   \
    GELOGE((ge::FAILED), __VA_ARGS__); \
  }

// If expr is not SUCCESS, print the log and return the same value
#define GE_CHK_STATUS_RET(expr, ...)       \
  do {                                     \
    const ge::Status _chk_status = (expr); \
    if (_chk_status != ge::SUCCESS) {      \
      GELOGE((ge::FAILED), __VA_ARGS__);   \
      return _chk_status;                  \
    }                                      \
  } while (false)

// If expr is not SUCCESS, print the log and do not execute return
#define GE_CHK_STATUS(expr, ...)           \
  do {                                     \
    const ge::Status _chk_status = (expr); \
    if (_chk_status != ge::SUCCESS) {      \
      GELOGE(_chk_status, __VA_ARGS__);    \
    }                                      \
  } while (false)

// If expr is not SUCCESS, return the same value
#define GE_CHK_STATUS_RET_NOLOG(expr)      \
  do {                                     \
    const ge::Status _chk_status = (expr); \
    if (_chk_status != ge::SUCCESS) {      \
      return _chk_status;                  \
    }                                      \
  } while (false)

// If expr is not GRAPH_SUCCESS, print the log and return FAILED
#define GE_CHK_GRAPH_STATUS_RET(expr, ...)                     \
  do {                                                         \
    if ((expr) != ge::GRAPH_SUCCESS) {                         \
      REPORT_INNER_ERR_MSG("E19999", "Operator graph failed"); \
      GELOGE(ge::FAILED, __VA_ARGS__);                         \
      return (ge::FAILED);                                     \
    }                                                          \
  } while (false)

// If expr is not SUCCESS, print the log and execute a custom statement
#define GE_CHK_STATUS_EXEC(expr, exec_expr, ...)                      \
  do {                                                                \
    const ge::Status _chk_status = (expr);                            \
    GE_CHK_BOOL_EXEC(_chk_status == SUCCESS, exec_expr, __VA_ARGS__); \
  } while (false)

// If expr is not true, print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS(expr, _status, ...) \
  do {                                             \
    const bool b = (expr);                         \
    if (!b) {                                      \
      REPORT_INNER_ERR_MSG("E19999", __VA_ARGS__); \
      GELOGE((_status), __VA_ARGS__);              \
      return (_status);                            \
    }                                              \
  } while (false)

// If expr is true, print info log and return the specified status
#define GE_CHK_BOOL_RET_SPECIAL_STATUS(expr, _status, ...) \
  do {                                                     \
    const bool b = (expr);                                 \
    if (b) {                                               \
      GELOGI(__VA_ARGS__);                                 \
      return (_status);                                    \
    }                                                      \
  } while (false)

// If expr is not true, print the log and return the specified status
#define GE_CHK_BOOL_RET_STATUS_NOLOG(expr, _status, ...) \
  do {                                                   \
    const bool b = (expr);                               \
    if (!b) {                                            \
      return (_status);                                  \
    }                                                    \
  } while (false)

// If expr is not true, print the log and execute a custom statement
#define GE_CHK_BOOL_EXEC(expr, exec_expr, ...) \
  {                                            \
    const bool b = (expr);                     \
    if (!b) {                                  \
      GELOGE(ge::FAILED, __VA_ARGS__);         \
      exec_expr;                               \
    }                                          \
  }

// -----------------runtime related macro definitions-------------------------------
// If expr is not RT_ERROR_NONE, print the log
#define GE_CHK_RT(expr)                                                \
  do {                                                                 \
    const rtError_t _rt_err = (expr);                                  \
    if (_rt_err != RT_ERROR_NONE) {                                    \
      GELOGE(ge::RT_FAILED, "Call rt api failed, ret: 0x%X", _rt_err); \
    }                                                                  \
  } while (false)

// If expr is not RT_ERROR_NONE, print the log and execute the exec_expr expression
#define GE_CHK_RT_EXEC(expr, exec_expr)                                \
  {                                                                    \
    const rtError_t _rt_ret = (expr);                                  \
    if (_rt_ret != RT_ERROR_NONE) {                                    \
      GELOGE(ge::RT_FAILED, "Call rt api failed, ret: 0x%X", _rt_ret); \
      exec_expr;                                                       \
    }                                                                  \
  }

// If expr is not RT_ERROR_NONE, print the log and return
#define GE_CHK_RT_RET(expr)                                                                             \
  do {                                                                                                  \
    const rtError_t _rt_ret = (expr);                                                                   \
    if (_rt_ret != RT_ERROR_NONE) {                                                                     \
      REPORT_INNER_ERR_MSG("E19999", "Call %s fail, ret: 0x%X", #expr, static_cast<uint32_t>(_rt_ret)); \
      GELOGE(ge::RT_FAILED, "Call rt api failed, ret: 0x%X", static_cast<uint32_t>(_rt_ret));           \
      return RT_ERROR_TO_GE_STATUS(_rt_ret);                                                            \
    }                                                                                                   \
  } while (false)

// If expr is true, execute exec_expr without printing logs
#define GE_IF_BOOL_EXEC(expr, exec_expr) \
  {                                      \
    if (expr) {                          \
      exec_expr;                         \
    }                                    \
  }

// If make_shared is abnormal, print the log and execute the statement
#define GE_MAKE_SHARED(exec_expr0, exec_expr1) \
  try {                                        \
    exec_expr0;                                \
  } catch (const std::bad_alloc&) {            \
    GELOGE(ge::FAILED, "Make shared failed");  \
    exec_expr1;                                \
  }

#define GE_ERRORLOG_AND_ERRORMSG(_status, errormsg)        \
  do {                                                     \
    GELOGE((_status), "[Check][InnerData]%s", (errormsg)); \
    REPORT_INNER_ERR_MSG("E19999", "%s", (errormsg));      \
  } while (false)

#define GE_WARNINGLOG_AND_ERRORMSG(errormsg)                                                         \
  do {                                                                                               \
    GELOGW("%s", (errormsg));                                                                        \
    REPORT_PREDEFINED_ERR_MSG(                                                                       \
        "E10052", std::vector<const char_t*>({"reason"}), std::vector<const char_t*>({(errormsg)})); \
  } while (false)

#define GE_CHK_LOG_AND_ERRORMSG(expr, _status, errormsg)                                               \
  do {                                                                                                 \
    const bool b = (expr);                                                                             \
    if (!b) {                                                                                          \
      GELOGE((_status), "%s", (errormsg));                                                             \
      REPORT_PREDEFINED_ERR_MSG(                                                                       \
          "E10052", std::vector<const char_t*>({"reason"}), std::vector<const char_t*>({(errormsg)})); \
      return (_status);                                                                                \
    }                                                                                                  \
  } while (false)

namespace ge {
template <typename T>
GE_FUNC_VISIBILITY std::string FmtToStr(const T& t) {
  std::string fmt;
  std::stringstream st;
  st << "[" << t << "]";
  fmt = st.str();
  return fmt;
}
} // namespace ge

#define GE_LOGE(fmt, ...) GE_LOG_ERROR(GE_MODULE_NAME, ge::FAILED, fmt, ##__VA_ARGS__)

// Only check error log
#define GE_CHK_BOOL_ONLY_LOG(expr, ...) \
  do {                                  \
    const bool b = (expr);              \
    if (!b) {                           \
      GELOGI(__VA_ARGS__);              \
    }                                   \
  } while (false)

#endif // INC_COMMON_GE_COMMON_DEBUG_GE_LOG_H_
