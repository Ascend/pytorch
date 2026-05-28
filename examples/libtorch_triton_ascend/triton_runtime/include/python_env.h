#pragma once
#include <string>
#include <pybind11/pybind11.h>

namespace triton_runtime {

class PythonEnv {
public:
    static PythonEnv& instance();

    void ensure_initialized();
    bool is_initialized() const;

    pybind11::object exec(const std::string& code);
    pybind11::module import(const std::string& module_name);

private:
    PythonEnv() = default;
    bool initialized_ = false;
    void setup_triton_env();
};

} // namespace triton_runtime
