#include <sstream>

#include <c10/util/ApproximateClock.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace torch_npu {
namespace profiler {
namespace {

using torch::profiler::impl::ProfilerStubs;
using torch::profiler::impl::ProfilerVoidEventStub;

static inline void npuCheck(aclError result, const char *file, int line)
{
    if (result != ACL_ERROR_NONE) {
        std::stringstream ss;
        ss << file << ":" << line << ": "
           << ", aclError id:" << result << "." << PROF_ERROR(ErrCode::ACL);
        throw std::runtime_error(ss.str());
    }
}
#define TORCH_NPU_CHECK(result) npuCheck(result, __FILE__, __LINE__);

struct NPUMethods : public ProfilerStubs {
    void record(
        c10::DeviceIndex* device,
        ProfilerVoidEventStub* event,
        int64_t* cpu_ns) const override
    {
        static int local_device = -1;
        static bool init_flag = false;
        if (!init_flag) {
            TORCH_NPU_CHECK(c10_npu::GetDevice(&local_device));
            init_flag = true;
        }
        if (device) {
            *device = local_device;
        }
        aclrtEvent npu_event = nullptr;
        TORCH_NPU_CHECK(c10_npu::acl::AclrtCreateEventWithFlag(&npu_event, ACL_EVENT_TIME_LINE));
        *event = std::shared_ptr<void>(npu_event, [](aclrtEvent ptr) {
            TORCH_NPU_CHECK(aclrtDestroyEvent(ptr));
        });
        static auto stream = c10_npu::getCurrentNPUStream();
        if (cpu_ns) {
            *cpu_ns = c10::getTime();
        }
        TORCH_NPU_CHECK(aclrtRecordEvent(npu_event, stream));
        ASCEND_LOGI("Event: aclrtRecordEvent is successfully executed.");
    }

    float elapsed(
        const ProfilerVoidEventStub* event1_,
        const ProfilerVoidEventStub* event2_) const override
    {
        auto event1 = event1_->get();
        auto event2 = event2_->get();
        TORCH_NPU_CHECK(aclrtSynchronizeEvent(event1));
        ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed for event1.");
        TORCH_NPU_CHECK(aclrtSynchronizeEvent(event2));
        ASCEND_LOGI("Event: aclrtSynchronizeEvent is successfully executed for event2.");
        float ms;
        TORCH_NPU_CHECK(aclrtEventElapsedTime(&ms, event1, event2));
        return ms * 1000.0;
    }

    void onEachDevice(std::function<void(int)> op) const override
    {
        c10_npu::OptionalNPUGuard device_guard;
        int dev = -1;
        auto ret = c10_npu::GetDevice(&dev);
        if (ret != ACL_ERROR_NONE) {
            dev = 0;
        }
        device_guard.set_index(dev);
        op(dev);
    }

    void synchronize() const override
    {
        c10_npu::npuSynchronizeDevice();
    }

    bool enabled() const override
    {
        return true;
    }

    void mark(const char*) const override {}

    void rangePush(const char*) const override {}

    void rangePop() const override {}
};

struct RegisterNPUMethods {
    RegisterNPUMethods()
    {
        static NPUMethods methods;
        torch::profiler::impl::registerPrivateUse1Methods(&methods);
    }
};
RegisterNPUMethods reg;

} // namespaces
} // namespace profiler
} // namespace torch_npu
