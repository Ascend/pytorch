// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <structmember.h>
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

#include "torch_npu/csrc/npu/Stream.h"
#include "torch_npu/csrc/npu/Module.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"


PyObject *THNPStreamClass = nullptr;

static PyObject* THNPStream_pynew(
    PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    HANDLE_TH_ERRORS

    int current_device;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&current_device));

    int is_sync_launch = 0;
    int priority = 0;
    uint64_t cdata = 0;

    static char *kwlist[] = {"priority", "is_sync_launch", "_cdata", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "|iiK", kwlist, &priority, &is_sync_launch, &cdata)) {
        return nullptr;
    }

    THPObjectPtr ptr(type->tp_alloc(type, 0));
    if (!ptr) {
        return nullptr;
    }

    c10_npu::NPUStream stream =
        cdata ?
        c10_npu::NPUStream::unpack(cdata) :
        c10_npu::getNPUStreamFromPool();

    stream.setSyncLaunchStream(is_sync_launch);

    THNPStream* self = (THNPStream *)ptr.get();
    self->cdata = stream.pack();
    new (&self->npu_stream) c10_npu::NPUStream(stream);

    return (PyObject *)ptr.release();
    END_HANDLE_TH_ERRORS
}

static void THNPStream_dealloc(THNPStream *self) {
  self->npu_stream.~NPUStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THNPStream_get_device(THNPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->npu_stream.device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_get_npu_stream(THNPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->npu_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_get_priority(THNPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(false, "NPU dose not support Stream.get_priority() currently.", PTA_ERROR(ErrCode::NOT_SUPPORT));
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_priority_range() {
  HANDLE_TH_ERRORS
  TORCH_CHECK(false, "NPU does not support Stream.priority_range() currently.", PTA_ERROR(ErrCode::NOT_SUPPORT));
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_query(THNPStream *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->npu_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_synchronize(THNPStream *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  {
    pybind11::gil_scoped_release no_gil;
    self->npu_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_set_data_preprocess_stream(THNPStream *self, PyObject *arg) {
  HANDLE_TH_ERRORS
  {
    pybind11::gil_scoped_release no_gil;
    bool is_data_preprocess_stream = THPUtils_unpackBool(arg);
    self->npu_stream.setDataPreprocessStream(is_data_preprocess_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_eq(THNPStream *self, THNPStream *other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->npu_stream == other->npu_stream);
  END_HANDLE_TH_ERRORS
}

static struct PyMemberDef THNPStream_members[] = {
    {(char*)"_cdata", T_ULONGLONG, offsetof(THNPStream, cdata), READONLY, nullptr},
    {nullptr}
};

static struct PyGetSetDef THNPStream_properties[] = {
    {"device", (getter)THNPStream_get_device, nullptr, nullptr, nullptr},
    {"npu_stream", (getter)THNPStream_get_npu_stream, nullptr, nullptr, nullptr},
    {"priority", (getter)THNPStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}
};

static PyMethodDef THNPStream_methods[] = {
    {(char*)"query", (PyCFunction)THNPStream_query, METH_NOARGS, nullptr},
    {(char*)"synchronize", (PyCFunction)THNPStream_synchronize, METH_NOARGS, nullptr},
    {(char*)"priority_range", (PyCFunction)(void(*)(void))THNPStream_priority_range, METH_STATIC | METH_NOARGS, nullptr},
    {(char*)"__eq__", (PyCFunction)THNPStream_eq, METH_O, nullptr},
    {(char*)"set_data_preprocess_stream", (PyCFunction)THNPStream_set_data_preprocess_stream, METH_O, nullptr},
    {nullptr}
};

PyTypeObject THNPStreamType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch_npu._C._NPUStreamBase",            /* tp_name */
  sizeof(THNPStream),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THNPStream_dealloc,        /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THNPStream_methods,                    /* tp_methods */
  THNPStream_members,                    /* tp_members */
  THNPStream_properties,                /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THNPStream_pynew,                      /* tp_new */
};

void THNPStream_init(PyObject *module)
{
    THNPStreamClass = (PyObject *)&THNPStreamType;
    if (PyType_Ready(&THNPStreamType) < 0) {
        throw python_error();
    }
    Py_INCREF(&THNPStreamType);
    if (PyModule_AddObject(module, "_NPUStreamBase", (PyObject *)&THNPStreamType) < 0) {
        throw python_error();
    }
}
