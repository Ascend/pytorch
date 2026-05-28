#include "kernel_cache.h"
#include "triton_runtime.h"
#include <fstream>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <c10/util/Logging.h>

namespace triton_runtime {

KernelCache& KernelCache::instance() {
    static KernelCache inst;
    return inst;
}

uint64_t KernelCache::compute_cache_key(
        const KernelDescriptor& kernel_desc,
        const std::vector<ArgInfo>& args) {
    std::ostringstream oss;
    oss << kernel_desc.function_name << "|";
    oss << kernel_desc.signature_str << "|";

    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& arg = args[i];
        bool is_ce = (i < kernel_desc.is_constexpr.size()) && kernel_desc.is_constexpr[i];

        if (arg.is_pointer()) {
            oss << "T:" << c10::toString(arg.tensor().scalar_type()) << ":";
            const auto& shape = arg.shape();
            for (size_t d = 0; d < shape.size(); ++d) {
                if (d > 0) oss << "x";
                oss << shape[d];
            }
        } else if (is_ce) {
            oss << "CE:" << arg.scalar_value();
        } else {
            oss << "S:" << arg.scalar_size();
        }
    }
    TRT_DEBUG("Cache key str: %s", oss.str().c_str());
    return std::hash<std::string>{}(oss.str());
}

std::shared_ptr<CompiledKernelEntry> KernelCache::query(
        const std::string& kernel_name,
        uint64_t cache_key) const {
    std::shared_lock lock(mutex_);
    auto kit = cache_.find(kernel_name);
    if (kit == cache_.end()) return nullptr;
    auto cit = kit->second.find(cache_key);
    if (cit == kit->second.end()) return nullptr;
    return cit->second;
}

void KernelCache::store(const std::string& kernel_name,
                         uint64_t cache_key,
                         std::shared_ptr<CompiledKernelEntry> entry) {
    std::unique_lock lock(mutex_);
    cache_[kernel_name][cache_key] = std::move(entry);
}

void KernelCache::invalidate(const std::string& kernel_name) {
    std::unique_lock lock(mutex_);
    auto it = cache_.find(kernel_name);
    if (it != cache_.end()) {
        cache_.erase(it);
    }
}

void KernelCache::clear() {
    std::unique_lock lock(mutex_);
    cache_.clear();
}

void KernelCache::shutdown() {
    clear();
}

} // namespace triton_runtime
