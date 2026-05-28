#include "kernel_compiler.h"
#include "triton_runtime.h"
#include "python_env.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eval.h>
#include <torch/csrc/autograd/python_variable.h>
#include <fstream>

// CANN Runtime headers
extern "C" {
#include <acl/acl.h>
#include <acl/acl_rt.h>
}

namespace py = pybind11;
using namespace pybind11::literals;

namespace triton_runtime {

namespace {

struct ToPyScalar {
    py::object operator()(int32_t v) const { return py::int_(v); }
    py::object operator()(float v)   const { return py::float_(v); }
    py::object operator()(bool v)    const { return py::bool_(v); }
};

constexpr ToPyScalar kToPyScalar;

py::object scalar_to_py_arg(const ArgInfo& arg) {
    return std::visit(kToPyScalar, arg.scalar());
}

py::list build_kernel_args(const std::vector<ArgInfo>& args) {
    py::list kernel_args;

    for (int i = 0; i < (int)args.size(); i++) {
        const auto& arg = args[i];
        if (arg.is_pointer()) {
            kernel_args.append(py::reinterpret_steal<py::object>(
                THPVariable_Wrap(arg.tensor())));
        } else {
            kernel_args.append(scalar_to_py_arg(arg));
        }
    }
    return kernel_args;
}

static void fill_compiled_result(const KernelDescriptor& kernel_desc, const py::object compiled_kernel, CompiledKernelEntry& compile_result) {
    compiled_kernel.attr("_init_handles")();

    // filled from packed_metadata
    py::dict packed_metadata = compiled_kernel.attr("packed_metadata").cast<py::dict>();

    // filled from compiled_kernel
    compile_result.kernel_name = packed_metadata["kernel_name"].cast<std::string>();
    if (packed_metadata.contains("tensor_kinds")) {
        compile_result.tensor_kinds = packed_metadata["tensor_kinds"].cast<std::vector<int>>();
    }
    compile_result.is_constexpr = kernel_desc.is_constexpr;

    // fill kernel fn
    py::object kernel_function = compiled_kernel.attr("function");
    compile_result.kernel_func_ptr = reinterpret_cast<void*>(kernel_function.cast<uintptr_t>());

    // fill launch fn — get so_launcher_path from NPULauncher via CompiledKernel.run
    if (py::hasattr(compiled_kernel, "run")) {
        py::object launcher = compiled_kernel.attr("run");
        if (py::hasattr(launcher, "so_launcher_path")) {
            py::object so_path_obj = launcher.attr("so_launcher_path");
            if (!so_path_obj.is_none()) {
                compile_result.launcher_so_path = so_path_obj.cast<std::string>();
            }
        }
    }
    if (!compile_result.launcher_so_path.empty() && !compile_result.launcher_so_handle) {
        compile_result.launcher_so_handle = dlopen(compile_result.launcher_so_path.c_str(),
                                            RTLD_NOW | RTLD_GLOBAL);
        if (!compile_result.launcher_so_handle) {
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror()
                                     + " path=" + compile_result.launcher_so_path);
        }
        TRT_DEBUG("dlopen OK: %s", compile_result.launcher_so_path.c_str());

        compile_result.launch_fn = reinterpret_cast<TritonLaunchKernelFn>(
            dlsym(compile_result.launcher_so_handle, "triton_launch_kernel"));
        if (!compile_result.launch_fn) {
            throw std::runtime_error(std::string("dlsym triton_launch_kernel failed: ") + dlerror());
        }
        TRT_DEBUG("dlsym triton_launch_kernel OK");
    }
}

static std::vector<std::string> mangle_arg_types(const py::list& kernel_args) {
    std::vector<std::string> types;
    try {
        auto mangle_type = py::module::import("triton.runtime.jit").attr("mangle_type");
        for (int i = 0; i < (int)kernel_args.size(); i++) {
            types.push_back(mangle_type(kernel_args[i], false).cast<std::string>());
        }
    } catch (const py::error_already_set& e) {
        TRT_DEBUG("warning: mangle_arg_types failed: %s", e.what());
    }
    return types;
}

} // anonymous namespace

KernelCompiler& KernelCompiler::instance() {
    static KernelCompiler inst;
    return inst;
}

std::shared_ptr<CompiledKernelEntry> KernelCompiler::compile(
        const KernelDescriptor& kernel_desc,
        const std::vector<ArgInfo>& args,
        const Grid& grid) {

    // Pre-check cache with C++ computed key, skip Python compile if hit
    uint64_t cache_key = KernelCache::compute_cache_key(kernel_desc, args);
    TRT_DEBUG("Cache key hash: %lu", cache_key);
    auto cached = KernelCache::instance().query(kernel_desc.kernel_name, cache_key);
    if (cached) {
        TRT_DEBUG("Cache HIT: %s key=0x%lx", kernel_desc.kernel_name.c_str(), cache_key);
        return cached;
    } else {
        TRT_DEBUG("Cache MISS");
    }

    PythonEnv::instance().ensure_initialized();
    py::gil_scoped_acquire gil;

    py::object compiled_kernel = do_python_compile(kernel_desc, args, grid);

    CompiledKernelEntry compile_result;
    fill_compiled_result(kernel_desc, compiled_kernel, compile_result);

    auto entry = std::make_shared<CompiledKernelEntry>(std::move(compile_result));
    KernelCache::instance().store(kernel_desc.kernel_name, cache_key, entry);
    return entry;
}

py::object KernelCompiler::do_python_compile(
        const KernelDescriptor& kernel_desc,
        const std::vector<ArgInfo>& args,
        const Grid& grid) {
    try {
        py::object jit_fn = kernel_desc.jit_function;
        py::list kernel_args = build_kernel_args(args);
        py::tuple grid_tuple = py::make_tuple(grid.x, grid.y, grid.z);

        return jit_fn.attr("warmup")(
            *kernel_args,
            **py::dict("grid"_a = grid_tuple)
        );
    } catch (const py::error_already_set& e) {
        TRT_DEBUG("Python compile error: %s", e.what());
        throw std::runtime_error(std::string("Python compile error: ") + e.what());
    } catch (const std::exception& e) {
        TRT_DEBUG("Compile error: %s", e.what());
        throw;
    }
}

void KernelCompiler::shutdown() {
    // KernelCompiler has no Python member variables, no special cleanup needed
}

} // namespace triton_runtime
