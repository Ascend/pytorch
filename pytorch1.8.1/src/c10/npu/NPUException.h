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

#pragma once

#include <iostream>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include <c10/npu/interface/AclInterface.h>

#define C10_NPU_SHOW_ERR_MSG()                            \
do {                                                      \
  std::cout<<c10::npu::acl::AclGetErrMsg()<<std::endl;    \
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
          "\n", c10::npu::acl::AclGetErrMsg());        \
    }                                                  \
  } while (0)

#define C10_NPU_CHECK_WARN(Error)                        \
  do {                                                   \
    if ((Error) != ACL_ERROR_NONE) {                     \
      TORCH_WARN("NPU warning, error code is ", Error,   \
      "\n", c10::npu::acl::AclGetErrMsg());              \
    }                                                    \
  } while (0)
