#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/logging/Init.h"

#include <pybind11/chrono.h>
#include <pybind11/operators.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

#include "torch_npu/csrc/logging/LogContext.h"

namespace torch_npu {
namespace logging {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* logging_init(PyObject* _unused, PyObject* noargs)
{
    auto torch_npu_C_module = THPObjectPtr(PyImport_ImportModule("torch_npu._C"));
    if (!torch_npu_C_module) {
        Py_RETURN_FALSE;
    }
    auto torch_npu_C_m = py::handle(torch_npu_C_module).cast<py::module>();
    auto m = torch_npu_C_m.def_submodule("_logging", " logging bindings");
    auto module = py::handle(m).cast<py::module>();

    shared_ptr_class_<npu_logging::LogContext>(module, "_LogContext")
        .def_static("GetInstance", &npu_logging::LogContext::GetInstance, py::return_value_policy::reference)
        .def("setLogs", &npu_logging::LogContext::setLogs);
    Py_RETURN_TRUE;
}

// autograd methods on torch._C
static PyMethodDef TorchLoggingMethods[] = { // NOLINT
    {"_logging_init", logging_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};


PyMethodDef* logging_functions()
{
    return TorchLoggingMethods;
}

}
}
#endif