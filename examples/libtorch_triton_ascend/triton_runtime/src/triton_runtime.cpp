#include "triton_runtime.h"
#include "kernel_registry.h"
#include "kernel_cache.h"
#include "kernel_compiler.h"

#include "python_env.h"

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <runtime/runtime/rt.h>

#include <filesystem>
#include <unordered_set>
#include <pybind11/pybind11.h>
#include <dlfcn.h>

namespace py = pybind11;
using namespace pybind11::literals;

// Tensor kinds for profiling
#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2

namespace triton_runtime {

extern "C" int aclInit(const char* config);

namespace {

Status lookup_kernel(
        const std::string& kernel_name,
        KernelDescriptor& out_desc) {
    auto desc = KernelRegistry::instance().lookup(kernel_name);
    if (!desc) {
        return Status::Error("Kernel not registered: " + kernel_name);
    }
    out_desc = *desc;
    return Status::OK();
}

Status get_compiled_kernel_entry(
        const KernelDescriptor& desc,
        const std::vector<ArgInfo>& arg_infos,
        const Grid& grid,
        std::shared_ptr<CompiledKernelEntry>& out_cached) {
    try {
        out_cached = KernelCompiler::instance().compile(
            desc, arg_infos, grid);
    } catch (const std::exception& e) {
        return Status::Error(std::string("Compile failed: ") + e.what());
    }
    return Status::OK();
}

Status prepare_and_launch_kernel(
        const std::shared_ptr<CompiledKernelEntry>& cached,
        const Grid& grid,
        const std::vector<ArgInfo>& arg_infos) {
    const char* kernelName = cached->kernel_name.c_str();
    try {
        std::vector<std::vector<int64_t>> tensorShapes;
        std::vector<void*> arg_ptrs;
        std::vector<size_t> arg_sizes;

        // calc total_storage
        size_t total_storage = 0;
        for (int i = 0; i < (int)arg_infos.size(); i++) {
            if (cached->is_constexpr[i]) continue;
            const auto& arg = arg_infos[i];
            if (arg.is_pointer()) {
                total_storage += sizeof(void*);
            } else {
                total_storage += arg.scalar_size();
            }
        }
        std::vector<char> args_storage(total_storage);
        size_t off = 0;

        // fill arg_ptrs and arg_sizes
        int tensor_idx = 0;
        for (int i = 0; i < (int)arg_infos.size(); i++) {
            if (cached->is_constexpr[i]) {
                continue;
            }
            const auto& arg = arg_infos[i];
            if (arg.is_pointer()) {
                void* ptr = arg.device_ptr();
                memcpy(args_storage.data() + off, &ptr, sizeof(void*));
                arg_ptrs.push_back(args_storage.data() + off);
                arg_sizes.push_back(sizeof(void*));
                off += sizeof(void*);
                tensorShapes.push_back(arg.shape());
                tensor_idx++;
            } else {
                size_t sz = arg.scalar_size();
                std::visit([&](auto&& val) {
                    memcpy(args_storage.data() + off, &val, sz);
                }, arg.scalar());
                arg_ptrs.push_back(args_storage.data() + off);
                arg_sizes.push_back(sz);
                off += sz;
            }
        }

        // Flatten shapes for C interface
        std::vector<int64_t> shapes_data;
        std::vector<int> shape_dims;
        for (auto& s : tensorShapes) {
            shape_dims.push_back((int)s.size());
            shapes_data.insert(shapes_data.end(), s.begin(), s.end());
        }

        // launch kernel
        if (!cached->launch_fn) {
            return Status::Error("launch_fn is null, kernel not properly compiled");
        }
        if (cached->kernel_func_ptr == nullptr) {
            return Status::Error("Function pointer is null, kernel not properly compiled");
        }

        c10_npu::NPUStream npu_stream = c10_npu::getCurrentNPUStream();
        void* stream = reinterpret_cast<void*>(npu_stream.stream());

        cached->launch_fn(kernelName, reinterpret_cast<const void*>(cached->kernel_func_ptr),
                  stream, grid.x, grid.y, grid.z,
                  shapes_data.data(), shape_dims.data(), (int)tensorShapes.size(),
                  cached->tensor_kinds.data(),
                  arg_ptrs.data(), arg_sizes.data(), (int)arg_ptrs.size());

        return Status::OK();

    } catch (const std::exception& e) {
        return Status::Error(std::string("Launch failed: ") + e.what());
    }
}

} // anonymous namespace

TritonRuntime::TritonRuntime() {
    // Pre-init ACL before Python interpreter to prevent DT_NEEDED-loaded
    // libascendcl.so's ELF constructors from interfering with aclInit later.
    int acl_ret = aclInit(nullptr);


    PythonEnv::instance().ensure_initialized();

    // init torch, torch_npu
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("append")(".");


    // by-pass "npu accelerator already initialized" problem, fixed in Ascend/pytorch v2.x.x-26.1.0
    py::module torch_module = py::module::import("torch");

    py::object orig_get_acc = torch_module.attr("_C").attr("_get_accelerator");
    py::object cpu_device = torch_module.attr("device")("cpu");
    torch_module.attr("_C").attr("_get_accelerator") = py::cpp_function([&]() -> py::object {
        return cpu_device;
    });

    py::module torch_npu_module = py::module_::import("torch_npu");
    py::object torch_npu_path = torch_npu_module.attr("__file__");
    torch_module.attr("_C").attr("_get_accelerator") = orig_get_acc;
    torch_module.attr("npu").attr("_lazy_init")();
}

void TritonRuntime::shutdown() {
    KernelCompiler::instance().shutdown();
    KernelCache::instance().shutdown();
    KernelRegistry::instance().shutdown();
}

TritonRuntime& TritonRuntime::instance() {
    static TritonRuntime inst;
    return inst;
}

Status TritonRuntime::register_kernel(const std::string& python_file,
                                       const std::string& function_name,
                                       const std::string& kernel_name) {
    return KernelRegistry::instance().register_kernel(
        python_file, function_name, kernel_name);
}


Status TritonRuntime::register_kernel_dir(const std::string& dir_path) {
    if (!std::filesystem::exists(dir_path) ||
        !std::filesystem::is_directory(dir_path)) {
        return Status::Error("Directory not found: " + dir_path);
    }

    Status overall = Status::OK();
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".py") {
            // Use filename (without extension) as kernel name
            std::string stem = entry.path().stem().string();
            Status s = register_kernel(entry.path().string(), stem, stem);
            if (!s.ok()) {
                overall = Status::Error(overall.error_message() +
                    "; " + s.error_message());
            }
        }
    }
    return overall;
}

Status TritonRuntime::register_kernel_from_dir(const std::string& dir_path) {
    return KernelRegistry::instance().register_kernel_from_dir(dir_path);
}

BoundArgs TritonRuntime::build_bound_args(
        const KernelDescriptor& desc,
        const std::vector<ArgInfo>& arg_infos) {
    BoundArgs bound;
    for (size_t i = 0; i < desc.param_names.size() && i < arg_infos.size(); i++) {
        const auto& arg = arg_infos[i];
        if (!arg.is_pointer())
            bound[desc.param_names[i]] = arg.scalar();
    }
    return bound;
}

Status TritonRuntime::run_impl(const std::string& kernel_name,
                               const Grid& grid,
                               std::vector<ArgInfo>&& arg_infos) {

    // Step 1: Look up kernel
    KernelDescriptor desc;
    Status s = lookup_kernel(kernel_name, desc);
    if (!s.ok()) return s;

    TRT_DEBUG("%s", desc.toString().c_str());

    if (desc.is_autotuned) {
        return Status::Error(
            "Kernel '" + kernel_name + "' is autotuned. "
            "Autotune is not supported in triton_runtime.");
    }

    // Resolve grid before compilation — callable grids need bound_args,
    // fixed grids are returned as-is.
    BoundArgs bound = build_bound_args(desc, arg_infos);
    Grid launch_grid = grid.resolve(bound);

    // Step 2-3: Compile and query cache
    std::shared_ptr<CompiledKernelEntry> cached;
    s = get_compiled_kernel_entry(desc, arg_infos, launch_grid, cached);
    if (!s.ok()) return s;

    TRT_DEBUG("%s \ngrid=(%d,%d,%d)", cached->toString().c_str(), launch_grid.x, launch_grid.y, launch_grid.z);

    return prepare_and_launch_kernel(cached, launch_grid, arg_infos);
}

bool TritonRuntime::has_kernel(const std::string& kernel_name) const {
    return KernelRegistry::instance().has_kernel(kernel_name);
}

std::vector<std::string> TritonRuntime::list_kernels() const {
    return KernelRegistry::instance().list_kernels();
}

void TritonRuntime::print_kernel_signature(const std::string& kernel_name) const {
    KernelRegistry::instance().print_kernel_signature(kernel_name);
}

} // namespace triton_runtime
