#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include "third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

#if defined(__GNUC__) && __GNUC__ >= 12
// __FILE_NAME__ is only available in GCC 12 and above
#define FILE_NAME __FILE_NAME__
#else
// use __FILE__ instead in GCC 11 and below, only get filename without path
#define FILE_NAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define NPUStatus std::string
#define NPU_STATUS_SUCCESS "SUCCESS"
#define NPU_STATUS_INTERNAL_ERROR "INTERNAL_ERROR"
#define NPU_STATUS_PARAM_ERROR "PARAM_ERROR"
#define NPU_STATUS_ALLOC_ERROR "ALLOC_ERROR"
#define NPU_STATUS_FAILED "FAILED"

#define ASCEND_LOGE(fmt, ...)                                                                           \
    do {                                                                                                \
        if (c10_npu::option::OptionsManager::isACLGlobalLogOn(ACL_ERROR)) {                             \
            aclAppLog(ACL_ERROR, __FUNCTION__, FILE_NAME, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__);   \
        }                                                                                               \
    } while (0);
#define ASCEND_LOGW(fmt, ...)                                                                           \
    do {                                                                                                \
        if (c10_npu::option::OptionsManager::isACLGlobalLogOn(ACL_WARNING)) {                           \
            aclAppLog(ACL_WARNING, __FUNCTION__, FILE_NAME, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__); \
        }                                                                                               \
    } while (0);
#define ASCEND_LOGI(fmt, ...)                                                                           \
    do {                                                                                                \
        if (c10_npu::option::OptionsManager::isACLGlobalLogOn(ACL_INFO)) {                              \
            aclAppLog(ACL_INFO, __FUNCTION__, FILE_NAME, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__);    \
        }                                                                                               \
    } while (0);
#define ASCEND_LOGD(fmt, ...)                                                                           \
    do {                                                                                                \
        if (c10_npu::option::OptionsManager::isACLGlobalLogOn(ACL_DEBUG)) {                             \
            aclAppLog(ACL_DEBUG, __FUNCTION__, FILE_NAME, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__);   \
        }                                                                                               \
    } while (0);
