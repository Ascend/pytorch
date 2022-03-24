// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
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

#ifndef __C10_NPU_NPU_LOG___
#define __C10_NPU_NPU_LOG___
#include <iostream>
#include <string>

#define NPUStatus std::string
#define SUCCESS "SUCCESS"
#define INTERNEL_ERROR "INTERNEL_ERROR"
#define PARAM_ERROR "PARAM_ERROR"
#define ALLOC_ERROR "ALLOC_ERROR"
#define FAILED "FAILED"

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
#endif

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