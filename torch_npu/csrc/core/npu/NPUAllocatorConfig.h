#pragma once
#include <set>
#include <unordered_set>

#include <c10/core/AllocatorConfig.h>
#include <c10/util/Deprecated.h>
#include <c10/util/env.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
namespace NPUCachingAllocator {

// Constants used by NPUAllocatorConfig
constexpr size_t kAlignRoundLarge = 16384;            // round up large allocs to 16 KB
constexpr size_t kSmallBuffer = 2097152;              // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;             // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760;           // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kMB = 1024 * 1024;                   // 1 MB
constexpr size_t k20MB = 20;                          // 20 MB for segment_size
constexpr size_t k512MB = 512;                        // 512 MB for segment_size
constexpr size_t kRoundUpPowerOfTwoStart = 1ULL << 20; // 1 MB
constexpr size_t kRoundUpPowerOfTwoEnd = 1ULL << 36;   // 64 GB
constexpr size_t kRoundUpPowerOfTwoIntervals = 16;


/*
 *  Considering that users may configure only via the PYTORCH_NPU_ALLOC_CONF environment variable, the
    AcceleratorAllocatorConfig class cannot parse PYTORCH_NPU_ALLOC_CONF. Therefore, torch_npu must uniformly obtain
    configuration values from the NPUAllocatorConfig class instance, rather than from the AcceleratorAllocatorConfig
    class instance. Otherwise, some configurations in the PYTORCH_NPU_ALLOC_CONF environment variable may not take
    effect, leading to errors.
*/
class NPUAllocatorConfig {
public:
    static size_t max_split_size()
    {
        // Make sure the NPUAllocatorConfig instance is initialized first
        instance();
        return c10::CachingAllocator::AcceleratorAllocatorConfig::max_split_size();
    }

    static double garbage_collection_threshold()
    {
        instance();
        return c10::CachingAllocator::AcceleratorAllocatorConfig::garbage_collection_threshold();
    }

    static bool expandable_segments()
    {
        instance();
        return c10::CachingAllocator::AcceleratorAllocatorConfig::use_expandable_segments();
    }

    static size_t large_segment_size()
    {
        instance();
        return c10::CachingAllocator::AcceleratorAllocatorConfig::large_segment_size();
    }

    static bool pin_memory_expandable_segments()
    {
        return instance().m_pin_memory_expandable_segments;
    }

    static bool pinned_mem_register()
    {
        return instance().m_pinned_mem_register;
    }

    static size_t base_addr_aligned_size()
    {
        return instance().m_base_addr_aligned_size;
    }

    static bool page_size_1g_enable()
    {
        return instance().m_page_size_1g;
    }
    static bool multi_stream_lazy_reclaim()
    {
        return instance().m_multi_stream_lazy_reclaim;
    }

    static size_t segment_size_mb()
    {
        return instance().m_segment_size_mb;
    }

    static double per_process_memory_fraction()
    {
        return instance().m_per_process_memory_fraction;
    }

    static size_t roundup_power2_divisions(size_t size)
    {
        instance();
        return c10::CachingAllocator::AcceleratorAllocatorConfig::roundup_power2_divisions(size);
    }

    static bool pinned_use_background_threads()
    {
        instance();
        return c10::CachingAllocator::AcceleratorAllocatorConfig::pinned_use_background_threads();
    }

    static size_t pinned_reserve_segment_size_mb()
    {
        return instance().m_pinned_reserve_segment_size_mb;
    }

    static bool release_lock_on_npumalloc() {
        return instance().m_release_lock_on_npumalloc;
    }

    // When enabled, preemptively reject allocations exceeding
    // per_process_memory_fraction * total_device_memory, throwing
    // OutOfMemoryError without attempting the driver allocation.
    // This prevents fatal device OOM crashes in serving scenarios.
    static bool throw_on_npumalloc_oom() {
        return instance().m_throw_on_npumalloc_oom;
    }

    static size_t max_non_split_rounding_size()
    {
        instance();
        return c10::CachingAllocator::AcceleratorAllocatorConfig::max_non_split_rounding_size();
    }

    // Pinned memory allocator thresholds for rounding and caching control.
    // Delegate to AcceleratorAllocatorConfig which handles env var parsing.
    static size_t pinned_max_round_threshold()
    {
        instance();
        return c10::CachingAllocator::AcceleratorAllocatorConfig::pinned_max_round_threshold();
    }

    static size_t pinned_max_cached_size()
    {
        instance();
        return c10::CachingAllocator::AcceleratorAllocatorConfig::pinned_max_cached_size();
    }

    static NPUAllocatorConfig &instance();

    // Required by REGISTER_ALLOCATOR_CONFIG_PARSE_HOOK macro
    static const std::unordered_set<std::string>& getKeys() {
        static std::unordered_set<std::string> keys{
            "pin_memory_expandable_segments",
            "pinned_mem_register",
            "base_addr_aligned_kb",
            "page_size",
            "segment_size_mb",
            "multi_stream_lazy_reclaim",
            "pinned_reserve_segment_size_mb",
            "per_process_memory_fraction",
            "release_lock_on_npumalloc",
            "throw_on_npumalloc_oom"
        };
        return keys;
    }

    void parseArgs(const std::string& env, std::set<std::string> supported_settings = {});

private:
    bool m_pin_memory_expandable_segments = false;
    bool m_pinned_mem_register = false;
    size_t m_base_addr_aligned_size = kAlignRoundLarge;
    bool m_page_size_1g = false; // 新增1G页配置标志
    size_t m_segment_size_mb = 0;
    bool m_multi_stream_lazy_reclaim = false;
    double m_per_process_memory_fraction = 1.0;
    size_t m_pinned_reserve_segment_size_mb = 0;
    bool m_release_lock_on_npumalloc = false;
    bool m_throw_on_npumalloc_oom = false;

    NPUAllocatorConfig() = default;

    size_t parsePinMemoryExpandableSegments(const c10::CachingAllocator::ConfigTokenizer& config, size_t i);
    size_t parsePinnedMemRegister(const c10::CachingAllocator::ConfigTokenizer& config, size_t i);
    size_t parseAddrAlignSize(const c10::CachingAllocator::ConfigTokenizer& config, size_t i);
    size_t parsePageSize(const c10::CachingAllocator::ConfigTokenizer& config, size_t i);
    size_t parseSegmentSizeMb(const c10::CachingAllocator::ConfigTokenizer& config, size_t i);
    size_t parseMultiStreamLazyReclaim(const c10::CachingAllocator::ConfigTokenizer& config, size_t i);
    size_t parsePerProcessMemoryFraction(const c10::CachingAllocator::ConfigTokenizer& config, size_t i);
    size_t parseReleaseLockOnNpuMalloc(const c10::CachingAllocator::ConfigTokenizer& config, size_t i);
    size_t parseThrowOnNpuMallocOom(const c10::CachingAllocator::ConfigTokenizer& config, size_t i);
};

} // namespace NPUCachingAllocator
} // namespace c10_npu
