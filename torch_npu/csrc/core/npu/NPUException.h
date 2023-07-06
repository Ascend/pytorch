#pragma once

#include <iostream>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include"torch_npu/csrc/core/npu/interface/AclInterface.h"
#include"torch_npu/csrc/core/npu/NPUErrorCodes.h"

#define C10_NPU_SHOW_ERR_MSG()                            \
do {                                                      \
  std::cout<<c10_npu::acl::AclGetErrMsg()<<std::endl;    \
} while (0)

#define NPU_CHECK_ERROR(err_code)                                    \
  do {                                                               \
    auto Error = err_code;                                           \
    static c10_npu::acl::AclErrorCode err_map;                       \
    if ((Error) != ACL_ERROR_NONE) {                                 \
      TORCH_CHECK(                                                   \
        false,                                                       \
        __func__,                                                    \
        ":",                                                         \
        __FILE__,                                                    \
        ":",                                                         \
        __LINE__,                                                    \
        " NPU error, error code is ", Error,                         \
        (err_map.error_code_map.find(Error) !=                       \
        err_map.error_code_map.end() ?                               \
        "\n[Error]: " + err_map.error_code_map[Error] : ".") ,       \
        "\n", c10_npu::acl::AclGetErrMsg());                         \
    }                                                                \
  } while (0)

#define NPU_CHECK_SUPPORTED_OR_ERROR(err_code)                         \
  do {                                                                 \
    auto Error = err_code;                                             \
    static c10_npu::acl::AclErrorCode err_map;                         \
    if ((Error) != ACL_ERROR_NONE) {                                   \
      if ((Error) == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {               \
        static auto feature_not_support_warn_once = []() {             \
          printf("[WARN]%s,%s:%u:%s\n",                                \
                 __FUNCTION__, __FILENAME__, __LINE__,                 \
                 "Feature is not supportted and the possible cause is" \
                 " that driver and firmware packages do not match.");  \
          return true;                                                 \
        }();                                                           \
      } else {                                                         \
        TORCH_CHECK(                                                   \
          false,                                                       \
          __func__,                                                    \
          ":",                                                         \
          __FILE__,                                                    \
          ":",                                                         \
          __LINE__,                                                    \
          " NPU error, error code is ", Error,                         \
          (err_map.error_code_map.find(Error) !=                       \
          err_map.error_code_map.end() ?                               \
          "\n[Error]: " + err_map.error_code_map[Error] : ".") ,       \
          "\n", c10_npu::acl::AclGetErrMsg());                         \
      }                                                                \
    }                                                                  \
  } while (0)

#define NPU_CHECK_WARN(err_code)                                     \
  do {                                                               \
    auto Error = err_code;                                           \
    static c10_npu::acl::AclErrorCode err_map;                       \
    if ((Error) != ACL_ERROR_NONE) {                                 \
      TORCH_NPU_WARN("NPU warning, error code is ", Error,               \
        "[Error]: ",                                                 \
        (err_map.error_code_map.find(Error) !=                       \
        err_map.error_code_map.end() ?                               \
        "\n[Error]: " + err_map.error_code_map[Error] : ".") ,       \
        "\n", c10_npu::acl::AclGetErrMsg());                         \
    }                                                                \
  } while (0)

void warn_(const ::c10::Warning& warning);

#define TORCH_NPU_WARN(...)                                  \
  warn_(::c10::Warning(                                       \
      ::c10::UserWarning(),                                  \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
      ::c10::str(__VA_ARGS__),                               \
      false));

#define TORCH_NPU_WARN_ONCE(...)                                          \
  C10_UNUSED static const auto C10_ANONYMOUS_VARIABLE(TORCH_NPU_WARN_ONCE_) = \
      [&] {                                                               \
        TORCH_NPU_WARN(__VA_ARGS__);                                      \
        return true;                                                      \
      }()
