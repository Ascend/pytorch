#include <string>

#include <c10/core/AllocatorConfig.h>
#include <c10/util/Deprecated.h>

#include "torch_npu/csrc/core/npu/NPUAllocatorConfig.h"

namespace c10_npu {
namespace NPUCachingAllocator {

void NPUAllocatorConfig::parseArgs(const std::string& env)
{
    c10::CachingAllocator::ConfigTokenizer tokenizer(env);
    for (size_t i = 0; i < tokenizer.size(); ++i) {
        const auto& key = tokenizer[i];
        if (key == "pinned_reserve_segment_size_mb") {
            i = parsePinnedReserveSegmentSize(tokenizer, i);
        } else {
            // unrecognized token, we currently do not warn here until all NPUCachingAllocaotr config move here
            i = tokenizer.skipKey(i);
        }

        if (i+1 < tokenizer.size()) {
            tokenizer.checkToken(++i, ",");
        }
    }
}

size_t NPUAllocatorConfig::parsePinnedReserveSegmentSize(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i)
{
    tokenizer.checkToken(++i, ":");
    size_t val = tokenizer.toSizeT(++i);
    m_pinned_reserve_segment_size_mb = val;
    return i;
}

} // namespace NPUCachingAllocator
} // namespace c10_npu
