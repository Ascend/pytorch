#include <torch/csrc/python_headers.h>
#include <pybind11/chrono.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"

struct PyFuncStruct {
    PyFuncStruct(PyObject *pyFunc, PyObject *pyFuncArgs)
        : pyFunc(pyFunc), pyFuncArgs(pyFuncArgs)
        {
            Py_XINCREF(pyFunc);
            Py_XINCREF(pyFuncArgs);
        }

    ~PyFuncStruct()
    {
        Py_XDECREF(pyFunc);
        Py_XDECREF(pyFuncArgs);
    }

    PyObject* pyFunc = nullptr;
    PyObject* pyFuncArgs = nullptr;
};

struct ThreadArgs {
    ThreadArgs(aclrtContext context, bool exitFlag)
        : context(context), exitFlag(exitFlag) {}

    aclrtContext context;
    bool exitFlag;
};