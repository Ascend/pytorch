#include "torch_npu/csrc/afd/Init.h"
#include <torch/csrc/python_headers.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include "torch_npu/csrc/afd/ScheduleContext.h"

namespace torch_npu {
namespace afd {

PyObject *afd_init(PyObject * _unused, PyObject * noargs)
{
    auto torch_npu_C_module = THPObjectPtr(PyImport_ImportModule("torch_npu._C"));
    if (!torch_npu_C_module) {
        throw python_error();
    }
    auto torch_npu_C_m = py::handle(torch_npu_C_module).cast<py::module>();

    auto m = torch_npu_C_m.def_submodule("_afd", "Attention-FFN Disaggregation");
    auto module = py::handle(m).cast<py::module>();

    py::class_<ScheduleContextHolder>(module, "ScheduleContextHolder")
        .def(py::init<int32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t,
            uint64_t, uint64_t>())
        .def("init", &ScheduleContextHolder::Init)
        .def("get_context_tensor", &ScheduleContextHolder::GetContextTensor)
        .def("stop_schedule", &ScheduleContextHolder::StopSchedule)
        .def("get_schedule_context_info", &ScheduleContextHolder::GetScheduleContextInfo);
    Py_RETURN_TRUE;
}

// methods on torch._C
PyMethodDef methods[] = {
    {"_afd_init", afd_init, METH_NOARGS, nullptr},
    {nullptr,     nullptr, 0,            nullptr}
};

PyMethodDef *python_functions()
{
    return methods;
}

} // namespace afd
} // namespace torch_npu
