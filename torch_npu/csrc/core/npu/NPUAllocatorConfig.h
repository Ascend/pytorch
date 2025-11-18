#pragma once
#include <atomic>

#include <c10/core/AllocatorConfig.h>
#include <c10/util/Deprecated.h>
#include <c10/util/env.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
namespace NPUCachingAllocator {

class C10_NPU_API NPUAllocatorConfig {
public:
    static size_t pinned_reserve_segment_size_mb()
    {
        return instance().m_pinned_reserve_segment_size_mb;
    }

    static NPUAllocatorConfig& instance() {
        static NPUAllocatorConfig *s_instance = ([]() {
            auto inst = new NPUAllocatorConfig();
            auto env = c10::utils::get_env("PYTORCH_NPU_ALLOC_CONF"); // optional<string>
            if (env.has_value()) {
                inst->parseArgs(env.value());
            }
            return inst;
        })();
        return *s_instance;
    }

    void parseArgs(const std::string& env);

private:
    std::atomic<size_t> m_pinned_reserve_segment_size_mb{0};

    NPUAllocatorConfig() = default;

    size_t parsePinnedReserveSegmentSize(
        const c10::CachingAllocator::ConfigTokenizer& tokenizer,
        size_t i);
};

} // namespace NPUCachingAllocator
} // namespace c10_npu
