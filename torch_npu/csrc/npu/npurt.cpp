#include "torch_npu/csrc/npu/npurt.h"

#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"

namespace py = pybind11;

namespace torch_npu {
namespace npurt {

namespace {
int npuHostRegister(uintptr_t ptr, size_t size, uint32_t flags)
{
    if (ptr == 0) {
        return static_cast<int>(ACL_ERROR_INVALID_PARAM);
    }
    py::gil_scoped_release no_gil;
    return static_cast<int>(
        c10_npu::acl::AclrtHostRegisterV2(reinterpret_cast<void*>(ptr), size, flags));
}

int npuHostUnregister(uintptr_t ptr)
{
    if (ptr == 0) {
        return static_cast<int>(ACL_ERROR_INVALID_PARAM);
    }
    py::gil_scoped_release no_gil;
    return static_cast<int>(
        c10_npu::acl::AclrtHostUnregister(reinterpret_cast<void*>(ptr)));
}

int npuStreamCreate(uintptr_t ptr)
{
    if (ptr == 0) {
        return static_cast<int>(ACL_ERROR_INVALID_PARAM);
    }
    py::gil_scoped_release no_gil;
    return static_cast<int>(
        aclrtCreateStream(reinterpret_cast<aclrtStream*>(ptr)));
}

int npuStreamDestroy(uintptr_t ptr)
{
    if (ptr == 0) {
        return static_cast<int>(ACL_ERROR_INVALID_PARAM);
    }
    py::gil_scoped_release no_gil;
    return static_cast<int>(
        aclrtDestroyStream(reinterpret_cast<aclrtStream>(ptr)));
}

PyObject* npurt_init(PyObject* /* unused */, PyObject* /* noargs */)
{
    auto torch_npu_C_module = THPObjectPtr(PyImport_ImportModule("torch_npu._C"));
    if (!torch_npu_C_module) {
        return nullptr;
    }

    auto torch_npu_C_m = py::handle(torch_npu_C_module).cast<py::module>();
    auto npurt = torch_npu_C_m.def_submodule("_npurt", "NPU runtime API bindings");
    npurt.def("npuHostRegister", &npuHostRegister);
    npurt.def("npuHostUnregister", &npuHostUnregister);
    npurt.def("npuStreamCreate", &npuStreamCreate);
    npurt.def("npuStreamDestroy", &npuStreamDestroy);

    Py_RETURN_TRUE;
}

static PyMethodDef NPURTMethods[] = {
    {"_npurt_init", npurt_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

} // namespace

PyMethodDef* npurt_functions()
{
    return NPURTMethods;
}

} // namespace npurt
} // namespace torch_npu
