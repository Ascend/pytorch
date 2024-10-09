#pragma once

#include <ctime>
#include <mutex>
#include <cstdarg>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/NPUErrorCodes.h"
#include "torch_npu/csrc/core/npu/npu_log.h"


#define C10_NPU_SHOW_ERR_MSG()                                           \
    do {                                                                 \
        TORCH_NPU_WARN(c10_npu::c10_npu_get_error_message());            \
    } while (0)

#define NPU_CHECK_WARN(err_code)                                             \
    do {                                                                     \
        auto Error = err_code;                                               \
        static c10_npu::acl::AclErrorCode err_map;                           \
        if ((Error) != ACL_ERROR_NONE) {                                     \
            TORCH_NPU_WARN("NPU warning, error code is ", Error,             \
                "[Error]: ",                                                 \
                (err_map.error_code_map.find(Error) !=                       \
                err_map.error_code_map.end() ?                               \
                "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
                "\n", c10_npu::c10_npu_get_error_message());                 \
        }                                                                    \
    } while (0)

void warn_(const ::c10::Warning& warning);

#define TORCH_NPU_WARN(...)                                      \
    warn_(::c10::Warning(                                        \
          ::c10::UserWarning(),                                  \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          ::c10::str(__VA_ARGS__),                               \
          false));

#define TORCH_NPU_WARN_ONCE(...)                                                  \
    C10_UNUSED static const auto C10_ANONYMOUS_VARIABLE(TORCH_NPU_WARN_ONCE_) =   \
        [&] {                                                                     \
            TORCH_NPU_WARN(__VA_ARGS__);                                          \
            return true;                                                          \
        }()

enum class SubModule {
    PTA = 0,
    OPS = 1,
    DIST = 2,
    GRAPH = 3,
    PROF = 4
};

enum class ErrCode {
    SUC = 0,
    PARAM = 1,
    TYPE = 2,
    VALUE = 3,
    PTR = 4,
    INTERNAL = 5,
    MEMORY = 6,
    NOT_SUPPORT = 7,
    NOT_FOUND = 8,
    UNAVAIL = 9,
    SYSCALL = 10,
    TIMEOUT = 11,
    PERMISSION = 12,
    ACL = 100,
    HCCL = 200,
    GE = 300
};

static std::string getCurrentTimestamp();
std::string formatErrorCode(SubModule submodule, ErrCode errorCode);

#define PTA_ERROR(error) formatErrorCode(SubModule::PTA, error)
#define OPS_ERROR(error) formatErrorCode(SubModule::OPS, error)
#define DIST_ERROR(error) formatErrorCode(SubModule::DIST, error)
#define GRAPH_ERROR(error) formatErrorCode(SubModule::GRAPH, error)
#define PROF_ERROR(error) formatErrorCode(SubModule::PROF, error)

#define DEVICE_TASK_ABORT "reason=[device task abort]"
#define DEVICE_MEM_ERROR "reason=[device mem error]"

inline const char* getErrorFunction(const char* msg)
{
    return msg;
}

// If there is just 1 provided C-string argument, use it.
inline const char* getErrorFunction(const char* /* msg */, const char* args)
{
    return args;
}

#define CHECK_AND_THROW_FORCE_STOP(err_code)                                 \
    auto Error_stop = (int)(err_code);                                       \
    auto stop_error = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);                 \
    if ((stop_error) != ACL_ERROR_NONE) {                                    \
        Error_stop = stop_error;                                             \
    }                                                                        \
    if ((Error_stop) == ACL_ERROR_RT_DEVICE_TASK_ABORT) {                    \
        ASCEND_LOGE("getRepoStopFlag in Run, throw FORCE STOP.");            \
        TORCH_CHECK(                                                         \
            false,                                                           \
            __func__,                                                        \
            ":",                                                             \
            __FILE__,                                                        \
            ":",                                                             \
            __LINE__,                                                        \
            " NPU function error: FORCE STOP.",                              \
            ", error code is ", Error_stop,                                  \
            PTA_ERROR(ErrCode::ACL));                                        \
    }                                                                        \

#define CHECK_AND_THROW_UCE_ERROR(err_code)                                  \
    auto Error_uce = (int)(err_code);                                        \
    auto uce_error = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);                  \
    if ((uce_error) != ACL_ERROR_NONE) {                                     \
        Error_uce = uce_error;                                               \
    }                                                                        \
    if ((Error_uce) == ACL_ERROR_RT_DEVICE_MEM_ERROR && c10_npu::checkUceErrAndRepair()) {     \
        TORCH_CHECK(                                                         \
            false,                                                           \
            __func__,                                                        \
            ":",                                                             \
            __FILE__,                                                        \
            ":",                                                             \
            __LINE__,                                                        \
            " NPU function error: UCE ERROR.",                               \
            ", error code is ", Error_uce,                                   \
            PTA_ERROR(ErrCode::ACL));                                        \
    }                                                                        \

#define NPU_CHECK_ERROR_CHECK_UCE(err_code, check_uce, ...)                  \
    do {                                                                     \
        auto Error = err_code;                                               \
        static c10_npu::acl::AclErrorCode err_map;                           \
        if ((Error) != ACL_ERROR_NONE) {                                     \
            if (check_uce) {                                                 \
                CHECK_AND_THROW_FORCE_STOP(Error);                           \
                CHECK_AND_THROW_UCE_ERROR(Error);                            \
            }                                                                \
            TORCH_CHECK(                                                     \
                false,                                                       \
                __func__,                                                    \
                ":",                                                         \
                __FILE__,                                                    \
                ":",                                                         \
                __LINE__,                                                    \
                " NPU function error: ", getErrorFunction(#err_code, ##__VA_ARGS__),    \
                ", error code is ", Error,                                   \
                PTA_ERROR(ErrCode::ACL),                                     \
                (err_map.error_code_map.find(Error) !=                       \
                err_map.error_code_map.end() ?                               \
                "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
                "\n", c10_npu::c10_npu_get_error_message());                 \
        }                                                                    \
    } while (0)

#define NPU_CHECK_ERROR_WITHOUT_UCE(err_code, ...) NPU_CHECK_ERROR_CHECK_UCE(err_code, false, ##__VA_ARGS__)
#define NPU_CHECK_ERROR(err_code, ...) NPU_CHECK_ERROR_CHECK_UCE(err_code, true, ##__VA_ARGS__)

#define OPS_CHECK_ERROR(err_code, ...)                                       \
    do {                                                                     \
        auto Error = err_code;                                               \
        static c10_npu::acl::AclErrorCode err_map;                           \
        if ((Error) != ACL_ERROR_NONE) {                                     \
            CHECK_AND_THROW_FORCE_STOP(Error);                               \
            CHECK_AND_THROW_UCE_ERROR(Error);                                \
            TORCH_CHECK(                                                     \
                false,                                                       \
                __func__,                                                    \
                ":",                                                         \
                __FILE__,                                                    \
                ":",                                                         \
                __LINE__,                                                    \
                " OPS function error: ", getErrorFunction(#err_code, ##__VA_ARGS__),    \
                ", error code is ", Error,                                   \
                OPS_ERROR(ErrCode::ACL),                                     \
                (err_map.error_code_map.find(Error) !=                       \
                err_map.error_code_map.end() ?                               \
                "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
                "\n", c10_npu::c10_npu_get_error_message());                 \
        }                                                                    \
    } while (0)

#define NPU_CHECK_SUPPORTED_OR_ERROR(err_code, ...)                              \
    do {                                                                         \
        auto Error = err_code;                                                   \
        static c10_npu::acl::AclErrorCode err_map;                               \
        if ((Error) != ACL_ERROR_NONE) {                                         \
            CHECK_AND_THROW_FORCE_STOP(Error);                                   \
            if ((Error) == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {                   \
                static auto feature_not_support_warn_once = []() {               \
                    printf("[WARN]%s,%s:%u:%s\n",                                \
                           __FUNCTION__, __FILENAME__, __LINE__,                 \
                           "Feature is not supportted and the possible cause is" \
                           " that driver and firmware packages do not match.");  \
                    return true;                                                 \
                }();                                                             \
            } else {                                                             \
                CHECK_AND_THROW_UCE_ERROR(Error);                                \
                TORCH_CHECK(                                                     \
                    false,                                                       \
                    __func__,                                                    \
                    ":",                                                         \
                    __FILE__,                                                    \
                    ":",                                                         \
                    __LINE__,                                                    \
                    " NPU function error: ", getErrorFunction(#err_code, ##__VA_ARGS__),    \
                    ", error code is ", Error,                                   \
                    (err_map.error_code_map.find(Error) !=                       \
                    err_map.error_code_map.end() ?                               \
                    "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
                    "\n", c10_npu::c10_npu_get_error_message());                 \
            }                                                                    \
        }                                                                        \
    } while (0)

namespace c10_npu {

struct MemUceInfo {
    int device;
    aclrtMemUceInfo info[MAX_MEM_UCE_INFO_ARRAY_SIZE];
    size_t retSize;
    int mem_type;

    MemUceInfo() : device(-1), retSize(0), mem_type(0)
    {
        std::memset(info, 0, sizeof(info));
    }

    void clear()
    {
        device = -1;
        std::memset(info, 0, sizeof(info));
        retSize = 0;
        mem_type = 0;
    }
};

C10_NPU_API const char *c10_npu_get_error_message();

bool checkUceErrAndRepair();

void set_mem_uce_info(MemUceInfo info);

MemUceInfo get_mem_uce_info();

void clear_mem_uce_info();

} // namespace c10_npu
