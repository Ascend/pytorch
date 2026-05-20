#include <torch/csrc/python_headers.h>
#include <cstdint>
#include <vector>
#include <pybind11/chrono.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "third_party/acl/inc/acl/acl_sk.h"

struct PendingTensorData {
    PendingTensorData(uintptr_t dataPtr, Py_ssize_t nbytes, PyObject* shape, PyObject* dtype)
        : dataPtr(dataPtr), nbytes(nbytes), shape(shape), dtype(dtype)
    {
        Py_XINCREF(shape);
        Py_XINCREF(dtype);
    }

    uintptr_t dataPtr = 0;
    Py_ssize_t nbytes = 0;
    PyObject* shape = nullptr;
    PyObject* dtype = nullptr;
};

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

struct PendingCallPayload {
    PendingCallPayload(PyObject* pyFunc, PyObject* pyFuncArgs)
        : pyFuncData(pyFunc, pyFuncArgs)
    {
    }

    ~PendingCallPayload()
    {
        Py_CLEAR(pyFuncData.pyFuncArgs);
        for (auto& tensorData : pendingTensorData) {
            Py_XDECREF(tensorData.shape);
            Py_XDECREF(tensorData.dtype);
        }
    }

    PyFuncStruct pyFuncData;
    std::vector<PendingTensorData> pendingTensorData;
};

struct ThreadArgs {
    ThreadArgs(aclrtContext context, bool exitFlag)
        : context(context), exitFlag(exitFlag) {}

    aclrtContext context;
    bool exitFlag;
};