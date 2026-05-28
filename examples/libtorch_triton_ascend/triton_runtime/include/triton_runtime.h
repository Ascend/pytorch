#pragma once

#include "logging.h"
#include "grid.h"
#include "arg_info.h"
#include "status.h"

#include <string>
#include <vector>
#include <torch/torch.h>

namespace triton_runtime {

struct KernelDescriptor;

class TritonRuntime {
public:
    static TritonRuntime& instance();

    // ──── Registration Interface ────

    Status register_kernel(const std::string& python_file,
                           const std::string& function_name,
                           const std::string& kernel_name = "");

    Status register_kernel_dir(const std::string& dir_path);

    Status register_kernel_from_dir(const std::string& dir_path);

    // ──── Execution Interface ────

    template<typename... Args>
    Status run(const std::string& kernel_name,
               const Grid& grid_spec,
               Args&&... args) {
        return run_impl(kernel_name, grid_spec,
            {ArgInfo(std::forward<Args>(args))...});
    }

    // ──── Query Interface ────

    bool has_kernel(const std::string& kernel_name) const;
    std::vector<std::string> list_kernels() const;
    void print_kernel_signature(const std::string& kernel_name) const;

    // ──── Lifecycle ────

    // Actively clean up all resources (safe to call while Python/CANN is still alive).
    // If not called, resources will be automatically reclaimed by OS at process exit;
    // no cleanup during static destruction.
    void shutdown();

private:
    TritonRuntime();
    // ~TritonRuntime() {
    //     shutdown();
    // }

    // Build a C++ BoundArgs map {param_name: &arg_info, ...} using the
    // kernel's param_names metadata. Pure C++, no Python.
    BoundArgs build_bound_args(
        const KernelDescriptor& desc,
        const std::vector<ArgInfo>& arg_infos);

    Status run_impl(const std::string& kernel_name,
                    const Grid& grid_spec,
                    std::vector<ArgInfo>&& arg_infos);
};

} // namespace triton_runtime
