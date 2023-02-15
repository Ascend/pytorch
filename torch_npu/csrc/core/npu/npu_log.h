#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include "third_party/acl/inc/acl/acl_base.h"


#define NPUStatus std::string
#define SUCCESS "SUCCESS"
#define INTERNEL_ERROR "INTERNEL_ERROR"
#define PARAM_ERROR "PARAM_ERROR"
#define ALLOC_ERROR "ALLOC_ERROR"
#define FAILED "FAILED"

#define ASCEND_LOGE(fmt, ...) \
  aclAppLog(ACL_ERROR, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__)
#define ASCEND_LOGW(fmt, ...) \
  aclAppLog(ACL_WARNING, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__)
#define ASCEND_LOGI(fmt, ...) \
  aclAppLog(ACL_INFO, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__)
#define ASCEND_LOGD(fmt, ...) \
  aclAppLog(ACL_DEBUG, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__)

#define NPU_LOGE(fmt, ...)          \
  printf(                           \
      "[ERROR]%s,%s:%u:" #fmt "\n", \
      __FUNCTION__,                 \
      __FILE__,                     \
      __LINE__,                     \
      ##__VA_ARGS__)
#define NPU_LOGW(fmt, ...)         \
  printf(                          \
      "[WARN]%s,%s:%u:" #fmt "\n", \
      __FUNCTION__,                \
      __FILE__,                    \
      __LINE__,                    \
      ##__VA_ARGS__)
#define NPU_LOGI(fmt, ...)         \
  printf(                          \
      "[INFO]:" #fmt "\n",         \
      ##__VA_ARGS__)

#ifdef USE_NPU_LOG
#define NPU_LOGD(fmt, ...)         \
  printf(                          \
      "[INFO]%s,%s:%u:" #fmt "\n", \
      __FUNCTION__,                \
      __FILE__,                    \
      __LINE__,                    \
      ##__VA_ARGS__)
#else
#define NPU_LOGD(fmt, ...)
#endif