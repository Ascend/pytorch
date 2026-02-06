#pragma once

#include <c10/util/Registry.h>
#include <set>

namespace c10_npu {
    namespace NPUCachingAllocator {
        constexpr size_t kAlignRoundLarge = 16384;            // round up large allocs to 16 KB
        constexpr size_t kMB = 1024 * 1024;                   // 1 MB
        constexpr size_t k20MB = 20;                          // 20 MB for segmemt_size
        constexpr size_t k512MB = 512;                        // 512 MB for segmemt_size
        constexpr size_t kRoundUpPowerOfTwoStart = 1ULL << 20; // 1 MB
        constexpr size_t kRoundUpPowerOfTwoEnd = 1ULL << 36;   // 64 GB
        constexpr size_t kRoundUpPowerOfTwoIntervals = 16;
        constexpr size_t kLargeBuffer = 20971520;             // "large" allocations may be packed in 20 MiB blocks

        class CachingAllocatorConfig {
        public:
            static size_t max_split_size()
            {
                return instance().m_max_split_size;
            }

            static double garbage_collection_threshold()
            {
                return instance().m_garbage_collection_threshold;
            }

            static bool expandable_segments()
            {
                return instance().m_expandable_segments;
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

            static size_t segment_size_mb()
            {
                return instance().m_segment_size_mb;
            }

            static size_t roundup_power2_divisions(size_t size);

            static CachingAllocatorConfig &instance()
            {
                static CachingAllocatorConfig *s_instance = ([]() {
                    auto inst = new CachingAllocatorConfig();
                    const char *env = getenv("PYTORCH_NPU_ALLOC_CONF");
                    inst->parseArgs(env);
                    return inst;
                })();
                return *s_instance;
            }

            void parseArgs(const char *env, std::set<std::string> supported_settings = {});

        private:
            size_t m_max_split_size;
            double m_garbage_collection_threshold;
            bool m_expandable_segments;
            bool m_pinned_mem_register;
            bool set_expandable_segments_flag = false;
            size_t m_base_addr_aligned_size = kAlignRoundLarge;
            bool m_page_size_1g = false; // 新增1G页配置标志
            size_t m_segment_size_mb;
            std::vector<size_t> m_roundup_power2_divisions;

            CachingAllocatorConfig()
                : m_max_split_size(std::numeric_limits<size_t>::max()),
                  m_garbage_collection_threshold(0),
                  m_expandable_segments(false),
                  m_pinned_mem_register(false),
                  m_base_addr_aligned_size(kAlignRoundLarge),
                  m_segment_size_mb(0),
                  m_roundup_power2_divisions(kRoundUpPowerOfTwoIntervals, 0)
            {}

            void lexArgs(const char *env, std::vector<std::string> &config);
            void consumeToken(const std::vector<std::string> &config, size_t i, const char c);
            size_t parseMaxSplitSize(const std::vector<std::string> &config, size_t i);
            size_t parseGarbageCollectionThreshold(const std::vector<std::string> &config, size_t i);
            size_t parseExpandableSegments(const std::vector<std::string> &config, size_t i);
            size_t parsePinnedMemRegister(const std::vector<std::string> &config, size_t i);
            size_t parseAddrAlignSize(const std::vector<std::string> &config, size_t i);
            size_t parsePageSize(const std::vector<std::string> &config, size_t i);
            size_t parseSegmentSizeMb(const std::vector<std::string> &config, size_t i);
            size_t parseRoundUpPower2Divisions(const std::vector<std::string> &config, size_t i);
        };
    }
}
