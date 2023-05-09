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

#define C10_NPU_CHECK(Error)                           \
  do {                                                 \
    if ((Error) != ACL_ERROR_NONE) {                   \
      TORCH_CHECK(                                     \
          false,                                       \
          __func__,                                    \
          ":",                                         \
          __FILE__,                                    \
          ":",                                         \
          __LINE__,                                    \
          " NPU error, error code is ", Error,         \
          ": ",                                       \
          (c10_npu::acl::error_code_map[Error]),      \
          "\n", c10_npu::acl::AclGetErrMsg());        \
    }                                                  \
  } while (0)

#define NPU_CHECK_SUPPORTED_OR_ERROR(Error)               \
  do {                                                    \
    if ((Error) != ACL_ERROR_NONE                         \
        && (Error) != ACL_ERROR_RT_FEATURE_NOT_SUPPORT) { \
      TORCH_CHECK(                                        \
          false,                                          \
          __func__,                                       \
          ":",                                            \
          __FILE__,                                       \
          ":",                                            \
          __LINE__,                                       \
          " NPU error, error code is ", Error,            \
          ": ",                                           \
          (c10_npu::acl::error_code_map[Error]),          \
          "\n", c10_npu::acl::AclGetErrMsg());            \
    }                                                     \
  } while (0)

#define C10_NPU_CHECK_WARN(Error)                        \
  do {                                                   \
    if ((Error) != ACL_ERROR_NONE) {                     \
      TORCH_WARN("NPU warning, error code is ", Error,   \
      ": ",                                             \
      (c10_npu::acl::error_code_map[Error]),            \
      "\n", c10_npu::acl::AclGetErrMsg());              \
    }                                                    \
  } while (0)
