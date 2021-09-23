// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#ifndef __NATIVE_NPU_INTERFACE_ENVVARIABLES__
#define __NATIVE_NPU_INTERFACE_ENVVARIABLES__

namespace at {
namespace native {
namespace npu {
namespace env {

/**
  check if the autotuen is enabled, return true or false.
  */
bool AutoTuneEnabled();
bool CheckBmmV2Enable();
bool CheckFuzzyEnable();
bool CheckProfilingEnable();
bool CheckMmBmmNDEnable();

} // namespace env
} // namespace npu
} // namespace native
} // namespace at

#endif //__NATIVE_NPU_INTERFACE_ENVVARIABLES__