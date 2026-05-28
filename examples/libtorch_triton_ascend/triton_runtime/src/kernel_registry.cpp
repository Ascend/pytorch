#include "kernel_registry.h"
#include "triton_runtime.h"
#include "python_env.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_set>

namespace py = pybind11;

namespace triton_runtime {

namespace {

// Build formatted signature string from JITFunction params.
// Reads params[i].annotation (triton normalizes tl.tensor → "tensor",
// tl.constexpr → "constexpr", no annotation → "").
std::string format_signature_str(const std::string& function_name,
                                 const py::object& jit_fn) {
    std::ostringstream oss;
    oss << "def " << function_name << "(";

    try {
        py::list params = jit_fn.attr("params");

        for (size_t i = 0; i < params.size(); i++) {
            if (i > 0) oss << ", ";
            py::object p = params[i];
            std::string param_name = py::str(p.attr("name"));
            std::string annotation = py::str(p.attr("annotation"));

            oss << param_name;
            if (!annotation.empty()) {
                oss << ": [" << annotation << "]";
            }
        }
    } catch (...) {
        oss << "...";
    }

    oss << ")";
    return oss.str();
}

// Cross-validate: params[i].annotation == "constexpr" must match
// is_constexpr[i] derived from params[i].is_constexpr.
void validate_constexpr_consistency(const std::string& function_name,
                                    const py::object& jit_fn,
                                    const std::vector<bool>& is_constexpr) {
    try {
        py::list params = jit_fn.attr("params");

        for (size_t i = 0; i < params.size(); i++) {
            py::object p = params[i];
            std::string annotation = py::str(p.attr("annotation"));
            bool is_ce = i < is_constexpr.size() && is_constexpr[i];

            if ((annotation == "constexpr") != is_ce) {
                std::string param_name = py::str(p.attr("name"));
                TRT_DEBUG("WARNING: '%s' param '%s': annotation=%s but is_constexpr=%s",
                          function_name.c_str(),
                          param_name.c_str(),
                          annotation.empty() ? "''" : ("'" + annotation + "'").c_str(),
                          is_ce ? "true" : "false");
            }
        }
    } catch (...) {
        // can't validate, silently skip
    }
}

// Import a .py file and return the module object. Returns none() on failure.
py::object import_module_from_file(const std::string& python_file,
                                   const std::string& prefix) {
    auto file_path = std::filesystem::path(python_file);
    auto sys = py::module::import("sys");
    py::list sys_path = sys.attr("path");
    sys_path.append(file_path.parent_path().string());

    std::string unique_name = prefix + file_path.stem().string() + "_" +
        std::to_string(std::hash<std::string>{}(python_file));

    // Check if module already exists in sys.modules to avoid overwriting
    py::dict modules = sys.attr("modules");
    if (modules.contains(unique_name.c_str())) {
        return modules[unique_name.c_str()];
    }

    auto importlib = py::module::import("importlib.util");
    auto spec = importlib.attr("spec_from_file_location")(unique_name, python_file);
    if (!spec) return py::object();

    auto module = importlib.attr("module_from_spec")(spec);
    modules.attr("__setitem__")(unique_name, module);
    spec.attr("loader").attr("exec_module")(module);
    return module;
}

// Check if obj is an Autotuner or JITFunction, unwrap inner JITFunction.
// Returns {inner_fn, is_autotuned}. inner_fn is empty if neither.
std::pair<py::object, bool> unwrap_jit_function(const py::object& obj,
                                                 const std::string& func_name) {
    auto triton_mod = py::module::import("triton.runtime");
    py::object autotuner_class = triton_mod.attr("Autotuner");

    if (py::isinstance(obj, autotuner_class)) {
        TRT_DEBUG("WARNING: '%s' is autotuned, which is not supported in triton_runtime", func_name.c_str());
        return {obj.attr("fn"), true};
    }
    py::object jit_class = triton_mod.attr("JITFunction");
    if (py::isinstance(obj, jit_class)) {
        return {obj, false};
    }
    return {py::object(), false};
}

} // anonymous namespace

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry inst;
    return inst;
}

Status KernelRegistry::register_kernel_impl(
    const std::string& python_file,
    const std::string& function_name,
    const std::string& kernel_name,
    const py::object& inner_jit_fn,
    bool is_autotuned) {

    std::string name = kernel_name.empty() ? function_name : kernel_name;

    std::string sig_json;
    try {
        sig_json = py::str(inner_jit_fn.attr("__annotations__"));
    } catch (...) {
        sig_json = "{}";
    }

    std::vector<bool> is_constexpr;
    std::vector<std::string> param_names;
    try {
        py::list params = inner_jit_fn.attr("params");
        for (size_t i = 0; i < params.size(); i++) {
            py::object p = params[i];
            param_names.push_back(py::str(p.attr("name")).cast<std::string>());
            is_constexpr.push_back(p.attr("is_constexpr").cast<bool>());
        }
    } catch (...) {
        // params not available, leave empty
    }

    validate_constexpr_consistency(function_name, inner_jit_fn, is_constexpr);

    auto desc = std::make_shared<KernelDescriptor>();
    desc->kernel_name = name;
    desc->python_file = python_file;
    desc->function_name = function_name;
    desc->signature_json = sig_json;
    desc->signature_str = format_signature_str(function_name, inner_jit_fn);
    desc->jit_function = inner_jit_fn;
    desc->is_constexpr = std::move(is_constexpr);
    desc->param_names = std::move(param_names);
    desc->is_autotuned = is_autotuned;

    {
        std::unique_lock lock(mutex_);
        registry_[name] = desc;
    }

    return Status::OK();
}

Status KernelRegistry::register_kernel(const std::string& python_file,
                                         const std::string& function_name,
                                         const std::string& kernel_name) {
    if (!std::filesystem::exists(python_file)) {
        return Status::Error("Python file not found: " + python_file);
    }

    try {
        PythonEnv::instance().ensure_initialized();

        py::gil_scoped_acquire gil;

        auto module = import_module_from_file(python_file, "triton_kernel_");
        if (module.is_none()) {
            return Status::Error("Cannot create module spec for: " + python_file);
        }

        if (!py::hasattr(module, function_name.c_str())) {
            return Status::Error("Function '" + function_name +
                                 "' not found in " + python_file);
        }

        py::object jit_fn = module.attr(function_name.c_str());
        auto [inner_jit_fn, is_autotuned] = unwrap_jit_function(jit_fn, function_name);

        return register_kernel_impl(python_file, function_name, kernel_name,
                                    inner_jit_fn, is_autotuned);
    } catch (const py::error_already_set& e) {
        return Status::Error(std::string("Python error in register_kernel: ") + e.what());
    } catch (const std::exception& e) {
        return Status::Error(std::string("register_kernel failed: ") + e.what());
    }
}

std::shared_ptr<KernelDescriptor> KernelRegistry::lookup(const std::string& kernel_name) const {
    std::shared_lock lock(mutex_);
    auto it = registry_.find(kernel_name);
    return it != registry_.end() ? it->second : nullptr;
}

bool KernelRegistry::has_kernel(const std::string& kernel_name) const {
    std::shared_lock lock(mutex_);
    return registry_.count(kernel_name) > 0;
}

std::vector<std::string> KernelRegistry::list_kernels() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> names;
    names.reserve(registry_.size());
    for (const auto& [name, _] : registry_) {
        names.push_back(name);
    }
    return names;
}

Status KernelRegistry::register_kernel_from_dir(const std::string& dir_path) {
    if (!std::filesystem::exists(dir_path) ||
        !std::filesystem::is_directory(dir_path)) {
        return Status::Error("Directory not found: " + dir_path);
    }

    int registered = 0;
    std::string errors;

    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.path().extension() != ".py") continue;

        std::string py_file = entry.path().string();

        try {
            PythonEnv::instance().ensure_initialized();
            py::gil_scoped_acquire gil;

            auto module = import_module_from_file(py_file, "triton_kernel_");
            if (module.is_none()) continue;

            py::dict members = module.attr("__dict__");
            for (auto& [key, value] : members) {
                py::object obj = py::reinterpret_borrow<py::object>(value);
                std::string func_name = py::str(key).cast<std::string>();
                auto [inner_fn, is_autotuned] = unwrap_jit_function(obj, func_name);
                if (!inner_fn) continue;

                Status s = register_kernel_impl(py_file, func_name, func_name,
                                                inner_fn, is_autotuned);
                if (s.ok()) registered++;
                else errors += s.error_message() + "; ";
            }
        } catch (...) {
            continue;
        }
    }

    if (registered == 0 && !errors.empty()) {
        return Status::Error("No kernels registered. Errors: " + errors);
    }
    return Status::OK();
}

void KernelRegistry::print_kernel_signature(const std::string& kernel_name) const {
    auto desc = lookup(kernel_name);
    if (!desc) {
        TRT_DEBUG("Kernel not found: %s", kernel_name.c_str());
        return;
    }
    std::cout << "[KernelRegistry] '" << kernel_name
              << "': " << desc->signature_str << std::endl;
}

Status KernelRegistry::unregister(const std::string& kernel_name) {
    std::unique_lock lock(mutex_);
    auto it = registry_.find(kernel_name);
    if (it == registry_.end()) {
        return Status::Error("Kernel not found: " + kernel_name);
    }
    // Decref Python object
    {
        py::gil_scoped_acquire gil;
        it->second->jit_function = py::object();
    }
    registry_.erase(it);
    return Status::OK();
}

void KernelRegistry::shutdown() {
    std::unique_lock lock(mutex_);
    py::gil_scoped_acquire gil;

    registry_.clear();
}

} // namespace triton_runtime
