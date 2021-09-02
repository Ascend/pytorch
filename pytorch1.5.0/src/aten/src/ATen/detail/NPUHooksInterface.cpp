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

#include <ATen/detail/NPUHooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

static NPUHooksInterface* npu_hooks = nullptr;

const NPUHooksInterface& getNPUHooks() {
    static std::once_flag once;
    std::call_once(once, [] {
        npu_hooks = NPUHooksRegistry()->Create("NPUHooks", NPUHooksArgs{}).release();
        if (!npu_hooks) {
            npu_hooks = new(std::nothrow) NPUHooksInterface();
            if (!npu_hooks) {
                AT_ERROR("create NPUHooksInterface failed.");
            }
        }
    });
    return *npu_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(NPUHooksRegistry, NPUHooksInterface, NPUHooksArgs)

} // namespace at