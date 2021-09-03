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

#include <c10/core/Allocator.h>
#include <ATen/core/Generator.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

constexpr const char* NPU_HELP =
        "This error has occurred because you are trying to use some NPU functionality, "
        "but do not build library with options USE_NPU = 1";

// The CUDAHooksInterface is an omnibus interface for any NPU functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of NPU code).  How do I
// decide if a function should live in this class?  There are two tests:
//
//  1. Does the *implementation* of this function require linking against
//     NPU libraries?
//
//  2. Is this function *called* from non-NPU ATen code?
//
// (2) should filter out many ostensible use-cases, since many times a NPU
// function provided by ATen is only really ever used by actual NPU code.
//
// TODO: Consider putting the stub definitions in another class, so that one
// never forgets to implement each virtual function in the real implementation
// in NPUHooks.  This probably doesn't buy us much though.
struct CAFFE2_API NPUHooksInterface {
    // This should never actually be implemented, but it is used to
    // squelch -Werror=non-virtual-dtor
    virtual ~NPUHooksInterface() {}

    // Initialize THCState and, transitively, the CUDA state
    virtual void initNPU() const {
      TORCH_CHECK(false, "Cannot initialize NPU without building library with options USE_NPU = 1.", NPU_HELP);
    }

    virtual Generator* getDefaultNPUGenerator(DeviceIndex device_index = -1) const {
      TORCH_CHECK(false, "Cannot get default NPU generator without ATen_cuda library. ", NPU_HELP);
    }

    virtual bool hasNPU() const {
      return false;
    }

    virtual int64_t current_device() const {
      return -1;
    }

    virtual Allocator* getPinnedMemoryAllocator() const {
      TORCH_CHECK(false, "Pinned memory requires NPU. ", NPU_HELP);
    }

    virtual int getNumNPUs() const {
      return 0;
    }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct CAFFE2_API NPUHooksArgs {};

C10_DECLARE_REGISTRY(NPUHooksRegistry, NPUHooksInterface, NPUHooksArgs);
#define REGISTER_NPU_HOOKS(clsname) \
C10_REGISTER_CLASS(NPUHooksRegistry, clsname, clsname)

namespace detail {
  CAFFE2_API const NPUHooksInterface& getNPUHooks();
} // namespace detail
} // namespace at