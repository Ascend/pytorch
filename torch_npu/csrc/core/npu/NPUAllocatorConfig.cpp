#include <set>
#include <string>

#include <c10/util/llvmMathExtras.h>
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUAllocatorConfig.h"

namespace c10_npu {
    namespace NPUCachingAllocator {
        bool isDigit(std::string str)
        {
            if (str.empty()) {
                return false;
            }
            return std::all_of(str.begin(), str.end(), [](unsigned char c) {
                return std::isdigit(c);
            });
        }

        void CachingAllocatorConfig::lexArgs(const char *env, std::vector<std::string> &config)
        {
            std::vector<char> buf;

            size_t env_length = strlen(env);
            for (size_t i = 0; i < env_length; i++) {
                if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
                    if (!buf.empty()) {
                        config.emplace_back(buf.begin(), buf.end());
                        buf.clear();
                    }
                    config.emplace_back(1, env[i]);
                } else if (env[i] != ' ') {
                    buf.emplace_back(static_cast<char>(env[i]));
                }
            }
            if (!buf.empty()) {
                config.emplace_back(buf.begin(), buf.end());
            }
        }

        void CachingAllocatorConfig::consumeToken(const std::vector<std::string> &config, size_t i, const char c)
        {
            TORCH_CHECK(i < config.size() && config[i].compare(std::string(1, c)) == 0,
                        "Error parsing CachingAllocator settings, expected ", c, PTA_ERROR(ErrCode::PARAM));
        }

        size_t CachingAllocatorConfig::parseMaxSplitSize(const std::vector<std::string> &config, size_t i)
        {
            consumeToken(config, ++i, ':');
            if (++i < config.size()) {
                TORCH_CHECK(isDigit(config[i]), "CachingAllocator option max_split_size_mb is invalid.");
                size_t val1 = static_cast<size_t>(stoi(config[i]));
                TORCH_CHECK(val1 > kLargeBuffer / (1024 * 1024),
                            "CachingAllocator option max_split_size_mb too small, must be > ", kLargeBuffer / (1024 * 1024),
                            PTA_ERROR(ErrCode::VALUE));
                val1 = std::max(val1, kLargeBuffer / (1024 * 1024));
                val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
                m_max_split_size = val1 * 1024 * 1024;
            } else {
                TORCH_CHECK(false, "Error, expecting max_split_size_mb value", PTA_ERROR(ErrCode::PARAM));
            }
            return i;
        }

        size_t CachingAllocatorConfig::parseGarbageCollectionThreshold(const std::vector<std::string> &config, size_t i)
        {
            consumeToken(config, ++i, ':');
            if (++i < config.size()) {
                double val1 = stod(config[i]);
                TORCH_CHECK(val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", PTA_ERROR(ErrCode::VALUE));
                TORCH_CHECK(val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", PTA_ERROR(ErrCode::VALUE));
                m_garbage_collection_threshold = val1;
            } else {
                TORCH_CHECK(false, "Error, expecting garbage_collection_threshold value", PTA_ERROR(ErrCode::VALUE));
            }
            return i;
        }

        size_t CachingAllocatorConfig::parseExpandableSegments(const std::vector<std::string> &config, size_t i)
        {
            consumeToken(config, ++i, ':');
            if (++i < config.size()) {
                TORCH_CHECK(i < config.size() && (config[i] == "True" || config[i] == "False"),
                            "Expected a single True/False argument for expandable_segments", PTA_ERROR(ErrCode::PARAM));
                m_expandable_segments = (config[i] == "True");
                if (m_expandable_segments) {
                    void *ptr = nullptr;
                    auto status = c10_npu::acl::AclrtReserveMemAddress(&ptr, 512, 0, nullptr, 1);
                    if (status == ACL_ERROR_NONE && ptr != nullptr) {
                        NPU_CHECK_ERROR(c10_npu::acl::AclrtReleaseMemAddress(ptr));
                    } else {
                        NPU_CHECK_ERROR(status, "aclrtReserveMemAddress");
                        TORCH_NPU_WARN_ONCE("expandable_segments setting failure, now change to `False`.");
                        m_expandable_segments = false;
                    }
                }
            } else {
                TORCH_CHECK(false, "Error, expecting expandable_segments value", PTA_ERROR(ErrCode::PARAM));
            }
            return i;
        }

        size_t CachingAllocatorConfig::parsePinnedMemRegister(const std::vector<std::string> &config, size_t i)
        {
            consumeToken(config, ++i, ':');
            if (++i < config.size()) {
                TORCH_CHECK(i < config.size() && (config[i] == "True" || config[i] == "False"),
                            "Expected a single True/False argument for pinned_mem_register", OPS_ERROR(ErrCode::PARAM));
                m_pinned_mem_register = (config[i] == "True");
                if (m_pinned_mem_register) {
                    if (!c10_npu::acl::AclrtMallocHostWithCfgExist()) {
                        TORCH_NPU_WARN_ONCE("pinned_mem_register setting failure, the current cann version does not support this feature, now change to `False`."
                            "To use this feature, you need to upgrade to version 8.5.0 or higher");
                        m_pinned_mem_register = false;
                        return i;
                    }
                }
            } else {
                TORCH_CHECK(false, "Error, expecting m_pinned_mem_register value", OPS_ERROR(ErrCode::VALUE));
            }
            return i;
        }

        size_t CachingAllocatorConfig::parseAddrAlignSize(const std::vector<std::string> &config, size_t i)
        {
            consumeToken(config, ++i, ':');
            if (++i < config.size()) {
                TORCH_CHECK(isDigit(config[i]), "CachingAllocator option base_addr_aligned_kb is invalid.");
                size_t val = static_cast<size_t>(stoi(config[i]));
                TORCH_CHECK(config[i].length() == std::to_string(val).length(),
                            "CachingAllocator option base_addr_aligned_kb error, must be [0~16], dtype is int",
                            OPS_ERROR(ErrCode::VALUE));
                TORCH_CHECK(val >= 0, "CachingAllocator option base_addr_aligned_kb error, must be [0~16], dtype is int",
                            OPS_ERROR(ErrCode::VALUE));
                TORCH_CHECK(val <= kAlignRoundLarge / 1024,
                            "CachingAllocator option base_addr_aligned_kb error, must be [0~16], dtype is int",
                            OPS_ERROR(ErrCode::VALUE));
                m_base_addr_aligned_size = val * 1024;
            } else {
                TORCH_CHECK(false, "Error, expecting base_addr_aligned_kb value", OPS_ERROR(ErrCode::VALUE));
            }
            return i;
        }

        size_t CachingAllocatorConfig::parsePageSize(const std::vector<std::string> &config, size_t i)
        {
            TORCH_CHECK(i + 2 < config.size(), "page_size requires format 'page_size:1g'", OPS_ERROR(ErrCode::VALUE));
            TORCH_CHECK(config[i + 1] == ":", "Expected ':' after page_size", OPS_ERROR(ErrCode::VALUE));

            if (config[i + 2] == "1g") {
                m_page_size_1g = true;
            } else {
                TORCH_CHECK(false, "Unsupported page_size value: ", config[i + 2], OPS_ERROR(ErrCode::VALUE));
            }
            return i + 2; // 返回最后处理的索引位置
        }

        size_t CachingAllocatorConfig::parseSegmentSizeMb(const std::vector<std::string> &config, size_t i)
        {
            consumeToken(config, ++i, ':');
            if (++i < config.size()) {
                TORCH_CHECK(isDigit(config[i]), "CachingAllocator option segment_size_mb is invalid.");
                size_t val = static_cast<size_t>(stoi(config[i]));
                TORCH_CHECK(val >= k20MB && val <= k512MB,
                            "CachingAllocator option segment_size_mb error, must be [20, 512], dtype is int",
                            OPS_ERROR(ErrCode::VALUE));
                m_segment_size_mb = val * kMB;
            } else {
                TORCH_CHECK(false, "Error, expecting segment_size_mb value", OPS_ERROR(ErrCode::VALUE));
            }
            return i;
        }

        size_t CachingAllocatorConfig::parseRoundUpPower2Divisions(const std::vector<std::string> &config, size_t i)
        {
            consumeToken(config, ++i, ':');
            TORCH_CHECK(i + 1 < config.size(), "Error, expecting roundup_power2_divisions value",
                        OPS_ERROR(ErrCode::VALUE));
            if (config[++i] == "[") {
                bool first_value = true;
                size_t last_index = 0;
                // NOLINTNEXTLINE(bugprone-inc-dec-in-conditions)
                while (++i < config.size()) {
                    if (config[i] == "]") {
                        break;
                    }

                    size_t value_index = i;
                    consumeToken(config, ++i, ':');
                    TORCH_CHECK(++i < config.size(), "Expected a value for roundup_power2_divisions entry",
                                OPS_ERROR(ErrCode::VALUE));
                    size_t value = static_cast<size_t>(stoull(config[i]));
                    TORCH_CHECK(value == 0 || c10::llvm::isPowerOf2_64(value),
                        "For roundups, the divisions has to be power of 2 or 0 to disable roundup ",
                        OPS_ERROR(ErrCode::VALUE));
                    if (config[value_index] == ">") {
                        std::fill(
                            m_roundup_power2_divisions.begin() +
                                static_cast<std::vector<size_t>::difference_type>(last_index + 1),
                            m_roundup_power2_divisions.end(),
                            value);
                    } else {
                        size_t boundary = static_cast<size_t>(stoull(config[value_index]));
                        TORCH_CHECK(c10::llvm::isPowerOf2_64(boundary),
                                    "For roundups, the intervals have to be power of 2 ",
                                    OPS_ERROR(ErrCode::VALUE));
                        size_t index = 63 - c10::llvm::countLeadingZeros(boundary);
                        index = std::clamp(index, size_t{0}, m_roundup_power2_divisions.size() - 1);

                        if (first_value) {
                            std::fill(
                                m_roundup_power2_divisions.begin(),
                                m_roundup_power2_divisions.begin() +
                                    static_cast<std::vector<size_t>::difference_type>(index),
                                value);
                            first_value = false;
                        }
                        m_roundup_power2_divisions[index] = value;
                        last_index = index;
                    }

                    if (i + 1 < config.size() && config[i + 1] != "]") {
                        consumeToken(config, ++i, ',');
                    }
                }
                TORCH_INTERNAL_ASSERT(
                    i < config.size(),
                    "Expected closing bracket ']' while parsing roundup_power2_divisions");
            } else {
                size_t value = static_cast<size_t>(stoull(config[i]));
                TORCH_CHECK(value == 0 || c10::llvm::isPowerOf2_64(value),
                            "For roundups, the divisions has to be power of 2 or 0 to disable roundup ",
                            OPS_ERROR(ErrCode::VALUE));
                std::fill(
                    m_roundup_power2_divisions.begin(),
                    m_roundup_power2_divisions.end(),
                    value);
            }
            return i;
        }

        size_t CachingAllocatorConfig::roundup_power2_divisions(size_t size)
        {
            if (size == 0 || instance().m_roundup_power2_divisions.empty()) {
                return 0;
            }
        
            size_t log_size = 63 - c10::llvm::countLeadingZeros(size);
        
            // Our intervals start at 1MB and end at 64GB
            const size_t interval_start = 63 - c10::llvm::countLeadingZeros(kRoundUpPowerOfTwoStart);
            const size_t interval_end = 63 - c10::llvm::countLeadingZeros(kRoundUpPowerOfTwoEnd);
        
            TORCH_INTERNAL_ASSERT(
                interval_end - interval_start == kRoundUpPowerOfTwoIntervals,
                "kRoundUpPowerOfTwoIntervals mismatch");
        
            size_t index = (log_size > interval_start) ? (log_size - interval_start) : 0ul;
            index = std::min(index, kRoundUpPowerOfTwoIntervals - 1);
            return instance().m_roundup_power2_divisions[index];
        }

        void CachingAllocatorConfig::parseArgs(const char *env, std::set<std::string> supported_settings)
        {
            // If empty, set the default values
            m_max_split_size = std::numeric_limits<size_t>::max();
            m_garbage_collection_threshold = 0;
            m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
        
            if (env == nullptr) {
                return;
            }
        
            std::vector<std::string> config;
            lexArgs(env, config);
        
            for (size_t i = 0; i < config.size(); i++) {
                // If supported_settings is not empty,
                // check if the setting is supported by torch_npu.npu.memory._set_allocator_settings().
                if (!supported_settings.empty() && supported_settings.count(config[i]) == 0) {
                    TORCH_CHECK(false, "torch_npu.npu.memory._set_allocator_settings() unsupported setting: ", config[i],
                        OPS_ERROR(ErrCode::VALUE));
                }
                if (config[i].compare("max_split_size_mb") == 0) {
                    i = parseMaxSplitSize(config, i);
                } else if (config[i].compare("garbage_collection_threshold") == 0) {
                    i = parseGarbageCollectionThreshold(config, i);
                } else if (config[i] == "expandable_segments") {
                    set_expandable_segments_flag = true;
                    i = parseExpandableSegments(config, i);
                } else if (config[i] == "pinned_mem_register") {
                    i = parsePinnedMemRegister(config, i);
                } else if (config[i] == "base_addr_aligned_kb") {
                    i = parseAddrAlignSize(config, i);
                } else if (config[i] == "page_size") {
                    i = parsePageSize(config, i);
                } else if (config[i] == "segment_size_mb") {
                    i = parseSegmentSizeMb(config, i);
                } else if (config[i] == "roundup_power2_divisions") {
                    i = parseRoundUpPower2Divisions(config, i);
                } else {
                    TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i], PTA_ERROR(ErrCode::PARAM));
                }
        
                if (i + 1 < config.size()) {
                    consumeToken(config, ++i, ',');
                }
            }
        
            if (m_expandable_segments) {
                if (set_expandable_segments_flag) {
                    TORCH_CHECK(m_max_split_size == std::numeric_limits<size_t>::max() && m_garbage_collection_threshold == 0,
                        "`max_split_size_mb` or `garbage_collection_threshold`, cannot be enabled with "
                        "`expandable_segments`, please set `expandable_segments` to `False`.",
                        OPS_ERROR(ErrCode::PARAM));
                } else if (m_max_split_size != std::numeric_limits<size_t>::max() || m_garbage_collection_threshold != 0) {
                    m_expandable_segments = false;
                    TORCH_NPU_WARN_ONCE("`max_split_size_mb` or `garbage_collection_threshold` is enabled, and the "
                        "`expandable_segments` is changed to `False` by default.");
                }
            }
        }

    } // namespace NPUCachingAllocator
} // namespace c10_npu
