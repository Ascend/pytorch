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

#include <torch/csrc/autograd/profiler.h>
#include <c10/npu/NPUStream.h>
#include <c10/npu/NPUGuard.h>
#include <c10/npu/interface/AclInterface.h>
#include <third_party/acl/inc/acl/acl_rt.h>
#include <sstream>

namespace torch { namespace autograd { namespace profiler {

namespace {

static inline void npuCheck(aclError result, const char * file, int line) {
  if(result != ACL_ERROR_NONE) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << ", aclError id:" << result << ".";
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_NPU_CHECK(result) npuCheck(result,__FILE__,__LINE__);

struct NPUMethods : public CUDAStubs {
  void npu_destropy_event(aclrtEvent event) const override {
    aclrtEventStatus status = ACL_EVENT_STATUS_RESERVED;
    TORCH_NPU_CHECK(aclrtQueryEvent(event, &status));
    if (status == ACL_EVENT_STATUS_COMPLETE) {
        TORCH_NPU_CHECK(aclrtDestroyEvent(event));
    } else {
        std::cout << "Warning! NPU destroy event error, status is not completed." << std::endl;
    }
  }
  void record(int* device, aclrtEvent* event1, int64_t* cpu_ns) const override {
    TORCH_NPU_CHECK(aclrtGetDevice(device));
    TORCH_NPU_CHECK(c10::npu::acl::AclrtCreateEventWithFlag(event1, ACL_EVENT_TIME_LINE));
    auto stream = c10::npu::getCurrentNPUStream();
    *cpu_ns = getTime();
    TORCH_NPU_CHECK(aclrtRecordEvent(*event1, stream));
  }
  float elapsed(const aclrtEvent& event1, const aclrtEvent& event2) const override {
    TORCH_NPU_CHECK(aclrtSynchronizeEvent(event1));
    TORCH_NPU_CHECK(aclrtSynchronizeEvent(event2));
    float ms;
    TORCH_NPU_CHECK(aclrtEventElapsedTime(&ms, event1, event2));
    return ms*1000.0;
  }
  void onEachDevice(std::function<void(int)> op) const override {
    c10::npu::OptionalNPUGuard device_guard;
    int dev = -1;
    auto ret = aclrtGetDevice(&dev);
    if (ret != ACL_ERROR_NONE) {
        dev = 0;
    }
    device_guard.set_index(dev);
    op(dev);
  }

  void synchronize() const override {
    c10::npu::npuSynchronizeDevice();
  }
  bool enabled() const override {
    return true;
  }

};

struct RegisterNPUMethods {
  RegisterNPUMethods() {
    static NPUMethods methods;
    registerCUDAMethods(&methods);
  }
};
RegisterNPUMethods reg;

} // namespaces
} // namespace profiler
} // namespace autograd
} // namespace torch
