#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/Dtype.h>

#include "torch_npu/csrc/utils/AutocastMode.h"

namespace torch_npu {
namespace autocast {

static PyObject* set_autocast_enabled(PyObject* _unused, PyObject* arg)
{
    HANDLE_TH_ERRORS
    if (!PyBool_Check(arg)) {
        throw torch::TypeError(
            "enabled must be a bool (got %s). %s",
            Py_TYPE(arg)->tp_name,
            PTA_ERROR(ErrCode::TYPE).c_str());
    }
    at::autocast::set_privateuseone_enabled(arg == Py_True);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* is_autocast_enabled(PyObject* _unused, PyObject* arg)
{
    HANDLE_TH_ERRORS
    if (at::autocast::is_privateuseone_enabled()) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
    END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_dtype(PyObject* _unused, PyObject* arg)
{
    HANDLE_TH_ERRORS
    if (!THPDtype_Check(arg)) {
        throw torch::TypeError(
            "dtype must be a torch.dtype (got %s). %s",
            Py_TYPE(arg)->tp_name,
            PTA_ERROR(ErrCode::TYPE).c_str());
    }
    at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
    at::autocast::set_autocast_privateuseone_dtype(targetType);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_dtype(PyObject* _unused, PyObject* arg)
{
    HANDLE_TH_ERRORS
    at::ScalarType current_dtype = at::autocast::get_autocast_privateuseone_dtype();
    auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
    Py_INCREF(dtype);
    return dtype;
    END_HANDLE_TH_ERRORS
}

// autocast methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"set_autocast_enabled", set_autocast_enabled, METH_O, nullptr},
    {"is_autocast_enabled", is_autocast_enabled, METH_NOARGS, nullptr},
    {"set_autocast_dtype", set_autocast_dtype, METH_O, nullptr},
    {"get_autocast_dtype", get_autocast_dtype, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* autocast_mode_functions()
{
    return methods;
}

}
}
