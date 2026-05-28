#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <shared_mutex>
#include <sstream>
#include <pybind11/pybind11.h>
#include "status.h"

namespace triton_runtime {

struct KernelDescriptor {
    std::string kernel_name;
    std::string python_file;
    std::string function_name;
    std::string signature_json;
    std::string signature_str;  // formatted: "def fn(arg: [type], ...)"
    pybind11::object jit_function;
    std::vector<bool> is_constexpr;
    std::vector<std::string> param_names;  // params[i].name, for C++ bound_args
    bool is_autotuned = false;

    std::string toString() const {
        std::ostringstream oss;
        oss << "KernelDescriptor{kernel_name=" << kernel_name
            << ", python_file=" << python_file
            << ", function_name=" << function_name
            << ", signature_str=" << signature_str
            << ", num_params=" << param_names.size()
            << ", is_autotuned=" << (is_autotuned ? "true" : "false")
            << "}";
        return oss.str();
    }
};

class KernelRegistry {
public:
    static KernelRegistry& instance();

    Status register_kernel(const std::string& python_file,
                           const std::string& function_name,
                           const std::string& kernel_name);

    Status register_kernel_from_dir(const std::string& dir_path);

    std::shared_ptr<KernelDescriptor> lookup(const std::string& kernel_name) const;
    bool has_kernel(const std::string& kernel_name) const;
    std::vector<std::string> list_kernels() const;
    void print_kernel_signature(const std::string& kernel_name) const;
    Status unregister(const std::string& kernel_name);
    void shutdown();

private:
    KernelRegistry() = default;
    Status register_kernel_impl(const std::string& python_file,
                                const std::string& function_name,
                                const std::string& kernel_name,
                                const pybind11::object& inner_jit_fn,
                                bool is_autotuned);

    std::unordered_map<std::string, std::shared_ptr<KernelDescriptor>> registry_;
    mutable std::shared_mutex mutex_;
};

} // namespace triton_runtime
