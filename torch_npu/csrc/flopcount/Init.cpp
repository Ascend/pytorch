#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/flopcount/Init.h"

#include <pybind11/chrono.h>
#include <pybind11/operators.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

#include "torch_npu/csrc/flopcount/FlopCountContext.h"

namespace torch_npu {
namespace flopcount {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* flops_count_init(PyObject* _unused, PyObject* noargs)
{
    auto torch_npu_C_module = THPObjectPtr(PyImport_ImportModule("torch_npu._C"));
    if (!torch_npu_C_module) {
        Py_RETURN_FALSE;
    }
    auto torch_npu_C_m = py::handle(torch_npu_C_module).cast<py::module>();
    auto m = torch_npu_C_m.def_submodule("_flops_count", " flops count bindings");
    auto module = py::handle(m).cast<py::module>();

    shared_ptr_class_<FlopCountContext>(module, "_FlopCountContext")
        .def_static("GetInstance", &FlopCountContext::GetInstance, py::return_value_policy::reference)
        .def("enable", &FlopCountContext::enable)
        .def("disable", &FlopCountContext::disable)
        .def("pause", &FlopCountContext::pause)
        .def("resume", &FlopCountContext::resume)
        .def("reset", &FlopCountContext::reset)
        .def_readonly("recordedCount", &FlopCountContext::recordedCount)
        .def_readonly("traversedCount", &FlopCountContext::traversedCount);

    Py_RETURN_TRUE;
}

// autograd methods on torch._C
static PyMethodDef TorchFlopsMethods[] = { // NOLINT
    {"_flops_count_init", flops_count_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};


PyMethodDef* flops_count_functions()
{
    return TorchFlopsMethods;
}

}
}
#endif // BUILD_LIBTORCH
