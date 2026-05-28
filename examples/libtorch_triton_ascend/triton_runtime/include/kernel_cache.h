#pragma once
#include <string>
#include <vector>
#include <memory>

#include <unordered_map>
#include <shared_mutex>
#include <sstream>
#include <dlfcn.h>
#include <cstdint>
#include "grid.h"
#include "arg_info.h"
#include "kernel_registry.h"

namespace triton_runtime {

// dlopen-based launcher: the generated launcher .so exposes this extern "C" entry point
typedef void (*TritonLaunchKernelFn)(
    const char* kernelName, const void* func, void* stream,
    int gridX, int gridY, int gridZ,
    const int64_t* shapes_data, const int* shape_dims, int num_tensors,
    const int* tensor_kinds,
    void* const* kernel_args, const size_t* arg_sizes, int num_args);

struct CompiledKernelEntry {
    std::string kernel_name;
    void* kernel_func_ptr = nullptr;
    std::string launcher_so_path;
    void* launcher_so_handle = nullptr;
    TritonLaunchKernelFn launch_fn = nullptr;
    std::vector<int> tensor_kinds;        // 0=input, 1=output, 2=input_output
    std::vector<bool> is_constexpr;

    std::string toString() const {
        std::ostringstream oss;
        oss << "CompiledKernelEntry{kernel_name=" << kernel_name
            << ", kernel_func_ptr=" << kernel_func_ptr
            << ", launcher_so=" << launcher_so_path
            << ", launch_fn=" << reinterpret_cast<const void*>(launch_fn)
            << ", num_tensors=" << tensor_kinds.size()
            << ", num_constexpr=";
        int ce_count = 0;
        for (bool b : is_constexpr) if (b) ce_count++;
        oss << ce_count << "/" << is_constexpr.size();
        oss << "}";
        return oss.str();
    }
};

class KernelCache {
public:
    static KernelCache& instance();

    static uint64_t compute_cache_key(
        const KernelDescriptor& kernel_desc,
        const std::vector<ArgInfo>& args);

    std::shared_ptr<CompiledKernelEntry> query(
        const std::string& kernel_name,
        uint64_t cache_key) const;

    void store(const std::string& kernel_name,
               uint64_t cache_key,
               std::shared_ptr<CompiledKernelEntry> entry);

    void invalidate(const std::string& kernel_name);
    void clear();
    void shutdown();

private:
    KernelCache() = default;
    std::unordered_map<std::string,
        std::unordered_map<uint64_t, std::shared_ptr<CompiledKernelEntry>>> cache_;
    mutable std::shared_mutex mutex_;
};

} // namespace triton_runtime
