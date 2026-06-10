#include <mutex>
#include <set>
#include <string>

#include <c10/core/AllocatorConfig.h>
#include <c10/util/Deprecated.h>
#include <c10/util/env.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUAllocatorConfig.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace c10_npu {
namespace NPUCachingAllocator {
const std::string pinMemoryExpandableMinCannVersion = "8.5.0";
const std::string pinMemoryExpandableMinDriverVersion = "25.5.0";
const std::string cannModule = "CANN";

bool isDigit(std::string str)
{
    if (str.empty()) {
        return false;
    }
    return std::all_of(str.begin(), str.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
}

NPUAllocatorConfig& NPUAllocatorConfig::instance()
{
    static NPUAllocatorConfig s_instance;
    static std::once_flag s_init_flag;

    std::call_once(s_init_flag, []() {
        auto env = c10::utils::get_env("PYTORCH_NPU_ALLOC_CONF");
        if (!env.has_value()) {
            env = c10::utils::get_env("PYTORCH_ALLOC_CONF");
        } else {
            // If PYTORCH_NPU_ALLOC_CONF is set, assign its value to PYTORCH_ALLOC_CONF,
            // because the AcceleratorAllocatorConfig parses PYTORCH_ALLOC_CONF.
            TORCH_NPU_MEMORY_LOGI("Set PYTORCH_ALLOC_CONF to PYTORCH_NPU_ALLOC_CONF value.");
            c10::utils::set_env("PYTORCH_ALLOC_CONF", env.value().c_str(), true);
        }
        if (!env.has_value()) {
            TORCH_NPU_MEMORY_LOGI("PYTORCH_NPU_ALLOC_CONF and PYTORCH_ALLOC_CONF not setted, use default configuration.");
            return;
        }
        TORCH_NPU_MEMORY_LOGI("Get alloc conf env: %s", env.value().c_str());
        auto& accAllocConfIns = c10::CachingAllocator::AcceleratorAllocatorConfig::instance();
        // Updating status of the AcceleratorAllocatorConfig instance is very important
        accAllocConfIns.parseArgs(env.value());
        // Check if the environment variable is valid
        if (accAllocConfIns.use_expandable_segments()) {
            TORCH_CHECK(accAllocConfIns.max_split_size() == std::numeric_limits<size_t>::max() &&
                accAllocConfIns.garbage_collection_threshold() == 0,
                "`max_split_size_mb` or `garbage_collection_threshold`, cannot be enabled with "
                "`expandable_segments`, please set `expandable_segments` to `False`.", OPS_ERROR(ErrCode::PARAM));
            void *ptr = nullptr;
            auto status = c10_npu::acl::AclrtReserveMemAddress(&ptr, 512, 0, nullptr, 1);
            if (status == ACL_ERROR_NONE && ptr != nullptr) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtReleaseMemAddress(ptr), "aclrtReleaseMemAddress failed.");
            } else {
                NPU_CHECK_ERROR(status, "aclrtReserveMemAddress failed.");
                const char* const report_msg = "expandable_segments setting failure, now change to `False`.";
                TORCH_NPU_WARN_ONCE(report_msg);
                TORCH_NPU_MEMORY_LOGW("%s", report_msg);
                accAllocConfIns.parseArgs("expandable_segments:False");
            }
        }
        s_instance.parseArgs(env.value());
        // log all the configuration
        TORCH_NPU_MEMORY_LOGI(
            "[npu alloc config] "
            "pin_memory_expandable_segments: %d, "
            "pinned_mem_register: %d, "
            "base_addr_aligned_kb: %zu, "
            "page_size_1g: %d, "
            "segment_size_mb: %zu, "
            "multi_stream_lazy_reclaim: %d, "
            "pinned_reserve_segment_size_mb: %zu, "
            "per_process_memory_fraction: %f.",
            s_instance.m_pin_memory_expandable_segments,
            s_instance.m_pinned_mem_register,
            s_instance.m_base_addr_aligned_size,
            s_instance.m_page_size_1g,
            s_instance.m_segment_size_mb,
            s_instance.m_multi_stream_lazy_reclaim,
            s_instance.m_pinned_reserve_segment_size_mb,
            s_instance.m_per_process_memory_fraction);
        TORCH_NPU_MEMORY_LOGI(
            "[common alloc config] "
            "max_split_size_mb: %zu, "
            "garbage_collection_threshold: %f, "
            "roundup_power2_divisions: %zu, "
            "expandable_segments: %d, "
            "pinned_use_background_threads: %d, "
            "large_segment_size: %zu.",
            accAllocConfIns.max_split_size(),
            accAllocConfIns.garbage_collection_threshold(),
            accAllocConfIns.roundup_power2_divisions(),
            accAllocConfIns.use_expandable_segments(),
            accAllocConfIns.pinned_use_background_threads());
    });

    return s_instance;
}

size_t NPUAllocatorConfig::parsePinMemoryExpandableSegments(const c10::CachingAllocator::ConfigTokenizer& tokenizer, size_t i)
{
    tokenizer.checkToken(++i, ":");
    if (++i < tokenizer.size()) {
        TORCH_CHECK(i < tokenizer.size() && (tokenizer[i] == "True" || tokenizer[i] == "False"),
            "Expected a single True/False argument for pin_memory_expandable_segments", OPS_ERROR(ErrCode::PARAM));
        m_pin_memory_expandable_segments = (tokenizer[i] == "True");
        if (m_pin_memory_expandable_segments) {
            if (!IsGteCANNVersion(pinMemoryExpandableMinCannVersion, cannModule)) {
                TORCH_NPU_WARN_ONCE("m_pin_memory_expandable_segments setting failure, the current cann version does not support this feature, now change to `False`."
                "To use this feature, you need to upgrade to version " + pinMemoryExpandableMinCannVersion + " or higher");
                m_pin_memory_expandable_segments = false;
                return i;
            }
            if (!IsGteDriverVersion(pinMemoryExpandableMinDriverVersion)) {
                TORCH_NPU_WARN_ONCE("m_pin_memory_expandable_segments setting failure, the current driver version does not support this feature, now change to `False`."
                "To use this feature, you need to upgrade to version " + pinMemoryExpandableMinDriverVersion + " or higher");
                m_pin_memory_expandable_segments = false;
                return i;
            }
        }
    } else {
        TORCH_CHECK(false, "Error, expecting m_pin_memory_expandable_segments value", OPS_ERROR(ErrCode::VALUE));
    }
    return i;
}

size_t NPUAllocatorConfig::parsePinnedMemRegister(const c10::CachingAllocator::ConfigTokenizer& tokenizer, size_t i)
{
    tokenizer.checkToken(++i, ":");
    if (++i < tokenizer.size()) {
        TORCH_CHECK(i < tokenizer.size() && (tokenizer[i] == "True" || tokenizer[i] == "False"),
                    "Expected a single True/False argument for pinned_mem_register", OPS_ERROR(ErrCode::PARAM));
        m_pinned_mem_register = (tokenizer[i] == "True");
    } else {
        TORCH_CHECK(false, "Error, expecting m_pinned_mem_register value", OPS_ERROR(ErrCode::VALUE));
    }
    return i;
}

size_t NPUAllocatorConfig::parseAddrAlignSize(const c10::CachingAllocator::ConfigTokenizer& tokenizer, size_t i)
{
    tokenizer.checkToken(++i, ":");
    if (++i < tokenizer.size()) {
        TORCH_CHECK(isDigit(tokenizer[i]), "CachingAllocator option base_addr_aligned_kb is invalid.");
        size_t val = static_cast<size_t>(stoi(tokenizer[i]));
        TORCH_CHECK(tokenizer[i].length() == std::to_string(val).length(),
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

size_t NPUAllocatorConfig::parseMultiStreamLazyReclaim(const c10::CachingAllocator::ConfigTokenizer& tokenizer, size_t i)
{
    tokenizer.checkToken(++i, ":");
    if (++i < tokenizer.size()) {
        TORCH_CHECK(i < tokenizer.size() && (tokenizer[i] == "True" || tokenizer[i] == "False"),
            "Expected a single True/False argument for multi_stream_lazy_reclaim", PTA_ERROR(ErrCode::PARAM));
        m_multi_stream_lazy_reclaim = (tokenizer[i] == "True");
    } else {
        TORCH_CHECK(false, "Error, expecting multi_stream_lazy_reclaim value", PTA_ERROR(ErrCode::PARAM));
    }
    return i;
}

size_t NPUAllocatorConfig::parsePerProcessMemoryFraction(const c10::CachingAllocator::ConfigTokenizer& tokenizer, size_t i)
{
    tokenizer.checkToken(++i, ":");
    if (++i < tokenizer.size()) {
        double val = stod(tokenizer[i]);
        TORCH_CHECK(val >= 0.0 && val <= 1.0,
            "per_process_memory_fraction is invalid, set it in [0.0, 1.0]", PTA_ERROR(ErrCode::VALUE));
        m_per_process_memory_fraction = val;
    } else {
        TORCH_CHECK(false, "Error, expecting per_process_memory_fraction value", PTA_ERROR(ErrCode::VALUE));
    }
    return i;
}

size_t NPUAllocatorConfig::parsePageSize(const c10::CachingAllocator::ConfigTokenizer& tokenizer, size_t i)
{
    TORCH_CHECK(i + 2 < tokenizer.size(), "page_size requires format 'page_size:1g'", OPS_ERROR(ErrCode::VALUE));
    tokenizer.checkToken(++i, ":");

    if (tokenizer[++i] == "1g") {
        m_page_size_1g = true;
    } else {
        TORCH_CHECK(false, "Unsupported page_size value: ", tokenizer[i], OPS_ERROR(ErrCode::VALUE));
    }
    return i;
}

size_t NPUAllocatorConfig::parseSegmentSizeMb(const c10::CachingAllocator::ConfigTokenizer& tokenizer, size_t i)
{
    TORCH_NPU_WARN_ONCE("`segment_size_mb` is deprecated, please use `large_segment_size_mb` instead.");
    tokenizer.checkToken(++i, ":");
    if (++i < tokenizer.size()) {
        TORCH_CHECK(isDigit(tokenizer[i]), "CachingAllocator option segment_size_mb is invalid.");
        size_t val = static_cast<size_t>(stoi(tokenizer[i]));
        TORCH_CHECK(val >= k20MB && val <= k512MB,
                    "CachingAllocator option segment_size_mb error, must be [20, 512], dtype is int",
                    OPS_ERROR(ErrCode::VALUE));
        m_segment_size_mb = val * kMB;
    } else {
        TORCH_CHECK(false, "Error, expecting segment_size_mb value", OPS_ERROR(ErrCode::VALUE));
    }
    return i;
}

void NPUAllocatorConfig::parseArgs(const std::string& env, std::set<std::string> supported_settings)
{
    if (env.empty()) {
        return;
    }
    c10::CachingAllocator::ConfigTokenizer tokenizer(env);
    for (size_t i = 0; i < tokenizer.size(); i++) {
        const auto& key = tokenizer[i];
        // If supported_settings is not empty, check if the setting is supported by torch_npu.npu.memory._set_allocator_settings().
        TORCH_CHECK(supported_settings.empty() || supported_settings.count(key) != 0,
            "torch_npu.npu.memory._set_allocator_settings() unsupported setting: ", key,
            OPS_ERROR(ErrCode::VALUE));
        if (key == "pin_memory_expandable_segments") {
            i = parsePinMemoryExpandableSegments(tokenizer, i);
        } else if (key == "pinned_mem_register") {
            i = parsePinnedMemRegister(tokenizer, i);
        } else if (key == "base_addr_aligned_kb") {
            i = parseAddrAlignSize(tokenizer, i);
        } else if (key == "page_size") {
            i = parsePageSize(tokenizer, i);
        } else if (key == "segment_size_mb") {
            i = parseSegmentSizeMb(tokenizer, i);
        } else if (key == "multi_stream_lazy_reclaim") {
            i = parseMultiStreamLazyReclaim(tokenizer, i);
        } else if (key == "per_process_memory_fraction") {
            i = parsePerProcessMemoryFraction(tokenizer, i);
        } else if (key == "pinned_reserve_segment_size_mb") {
            tokenizer.checkToken(++i, ":");
            m_pinned_reserve_segment_size_mb = tokenizer.toSizeT(++i);
        } else {
            const auto& accelerator_keys = c10::CachingAllocator::AcceleratorAllocatorConfig::getKeys();
            const auto& npu_support_keys = getSupportedPubilcKeys();
            // Check if it's a common key handled by AcceleratorAllocatorConfig
            if (accelerator_keys.find(key) != accelerator_keys.end()) {
                // Check if it's a supported key by torch_npu
                if (npu_support_keys.find(key) == npu_support_keys.end()) {
                    TORCH_NPU_WARN_ONCE("torch_npu not support key '", key, "' in NPU allocator config.");
                }
            } else {
                TORCH_CHECK_VALUE(false, "Unrecognized key '", key, "' in NPU allocator config.");
            }
            if (!supported_settings.empty()) {
                // supported_settings is not empty in torch_npu.npu.memory._set_allocator_settings()
                c10::CachingAllocator::AcceleratorAllocatorConfig::instance().parseArgs(env);
            }
            i = tokenizer.skipKey(i);
        }
        if (i + 1 < tokenizer.size()) {
            tokenizer.checkToken(++i, ",");
        }
    }
    if (m_pinned_mem_register) {
        if (!c10_npu::acl::AclrtMallocHostWithCfgExist()) {
            TORCH_NPU_WARN_ONCE("pinned_mem_register setting failure, the current cann version or driver version does not support this feature, now change to `False`."
                                "To use this feature, you need to upgrade to cann version 8.5.0 or higher and driver version 26.0.rc1 or higher.");
            m_pinned_mem_register = false;
        }

        if (m_pinned_mem_register && m_pin_memory_expandable_segments) {
            m_pinned_mem_register = false;
            TORCH_NPU_WARN_ONCE("pinned_mem_register setting failure, this feature is not supported when pin_memory_expandable_segments is set to `True`,"
                " now change to `False`.");
        }
    }
}
REGISTER_ALLOCATOR_CONFIG_PARSE_HOOK(NPUAllocatorConfig)
} // namespace NPUCachingAllocator
} // namespace c10_npu
