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

#ifndef __NATIVE_NPU_TOOLS_E2EPROFILER__
#define __NATIVE_NPU_TOOLS_E2EPROFILER__

#include <third_party/acl/inc/acl/acl.h>
#include <c10/npu/NPUException.h>
#include <chrono>
#include <sstream>
#include <thread>
#include <functional>

namespace at { 
namespace native {
namespace npu {
namespace profiler {


class TORCH_NPU_API E2ERecordFunction {
  public:
  // Default constructor is used with before function called afterwards
  E2ERecordFunction() { }

  E2ERecordFunction(const E2ERecordFunction&) = delete;
  E2ERecordFunction& operator=(const E2ERecordFunction&) = delete;

  // push and pop need to be paired, rangeStart and rangeStop need to be paired,
  void push();
  void pop();

  void rangeStart();
  void rangeStop();
  // before function initializes RecordFunction members and calls
  // start callbacks
  void before(const char* name);
  void before(std::string name);

  template<typename F>
  void before(F fn) {
    before(fn);
  }

  // Destructor calls end callbacks
  virtual ~E2ERecordFunction();

  bool active() const {
    return initialized_;
  }

  // Executes end callbacks
  void end();

  // Retrieves the thread_id that this RecordFunction ran start callbacks with.
  // Useful for writing thread safe end callbacks that may be potentially
  // executed in a different thread (async ops)
  inline uint16_t getStartCallbacksThreadId() const {
    return threadId_;
  }

  // Get logical thread_id for the current thread
  static uint16_t getCurrentThreadId();

 private:
  void processCallbacks();

  void checkProfilerRet(aclError ret, const std::string message);
  void checkProfilerRet(aclError ret, const char* message);

  std::string name_ = "";
  bool initialized_ = false;

  // The logical thread_id that this RecordFunction was created with.
  uint16_t threadId_ = 0;
  void * local_stamp = nullptr;
  uint32_t rangeId = -1;
};

TORCH_NPU_API bool hasCallbacks();

TORCH_NPU_API void popCallback();

using E2ERecordFunctionCallback = std::function<void(E2ERecordFunction&)>;
void pushCallback(
    E2ERecordFunctionCallback start,
    E2ERecordFunctionCallback end = [](E2ERecordFunction&){}
    );

// optional argument - function's seq_no
#define E2E_RECORD_FUNCTION(fn, ...) \
  at::native::npu::profiler::E2ERecordFunction e2e_guard; \
  if (at::native::npu::profiler::hasCallbacks()) { \
    e2e_guard.before(fn, ##__VA_ARGS__); \
  }

TORCH_NPU_API void init_e2e_profiler(const std::string dump_path,  uint64_t npu_event, uint64_t aicore_metrics);

TORCH_NPU_API void finalize_e2e_profiler();

} // namespace profiler
} // namespace npu
} // namespace native
} // namespace at

#endif // __NATIVE_NPU_TOOLS_E2EPROFILER__