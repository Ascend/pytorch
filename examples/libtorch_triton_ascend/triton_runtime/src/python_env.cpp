#include "python_env.h"
#include <pybind11/embed.h>
#include <iostream>
#include <cstdlib>

namespace triton_runtime {

PythonEnv& PythonEnv::instance() {
    static PythonEnv inst;
    return inst;
}

void PythonEnv::ensure_initialized() {
    if (initialized_) return;

    if (!Py_IsInitialized()) {
        // PYTHONHOME should be set by the caller (e.g. shell script) so that
        // the embedded interpreter can locate site-packages automatically.
        Py_Initialize();

        pybind11::gil_scoped_acquire gil;

        pybind11::module::import("__main__");
    }

    initialized_ = true;
}

bool PythonEnv::is_initialized() const {
    return initialized_;
}

pybind11::object PythonEnv::exec(const std::string& code) {
    ensure_initialized();
    pybind11::object main = pybind11::module::import("__main__");
    pybind11::object globals = main.attr("__dict__");
    return pybind11::eval(pybind11::str(code), globals, globals);
}

pybind11::module PythonEnv::import(const std::string& module_name) {
    ensure_initialized();
    return pybind11::module::import(module_name.c_str());
}

} // namespace triton_runtime
