#pragma once
#include "kernel_cache.h"
#include "kernel_registry.h"
#include "arg_info.h"
#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <functional>
#include <utility>

namespace triton_runtime {

class KernelCompiler {
public:
    static KernelCompiler& instance();

    std::shared_ptr<CompiledKernelEntry> compile(
        const KernelDescriptor& kernel_desc,
        const std::vector<ArgInfo>& args,
        const Grid& grid);

    void shutdown();

private:
    KernelCompiler() = default;

    pybind11::object do_python_compile(
        const KernelDescriptor& kernel_desc,
        const std::vector<ArgInfo>& args,
        const Grid& grid);
};

} // namespace triton_runtime
