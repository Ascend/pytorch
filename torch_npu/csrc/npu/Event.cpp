#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <structmember.h>

#include "torch_npu/csrc/npu/Event.h"
#include "torch_npu/csrc/npu/Stream.h"

PyObject *THNPEventClass = nullptr;

static PyObject* THNPEvent_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    HANDLE_TH_ERRORS
    unsigned char enable_timing = 0;
    unsigned char blocking = 0;
    unsigned char interprocess = 0;

    constexpr const char* kwlist[] = {"enable_timing", "blocking", "interprocess", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|bbb", const_cast<char**>(kwlist),
        &enable_timing, &blocking, &interprocess)) {
        return nullptr;
    }

    THPObjectPtr ptr(type->tp_alloc(type, 0));
    if (!ptr) {
        return nullptr;
    }

    THNPEvent* self = (THNPEvent *)ptr.get();

    unsigned int flags = 0;
    if (c10_npu::acl::IsExistCreateEventExWithFlag()) {
        flags = enable_timing ? (ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC) : ACL_EVENT_SYNC;
    } else {
        flags = enable_timing ? ACL_EVENT_TIME_LINE : ACL_EVENT_DEFAULT;
    }
    new (&self->npu_event) c10_npu::NPUEvent(flags);

    return (PyObject *)ptr.release();
    END_HANDLE_TH_ERRORS
}

static void THNPEvent_dealloc(THNPEvent *self)
{
    self->npu_event.~NPUEvent();
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THNPEvent_get_npu_event(THNPEvent *self, void *unused)
{
    HANDLE_TH_ERRORS
    return PyLong_FromVoidPtr(self->npu_event.event());
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_get_device(THNPEvent *self, void *unused)
{
    HANDLE_TH_ERRORS
    at::optional<at::Device> device = self->npu_event.device();
    if (!device) {
        Py_RETURN_NONE;
    }
    return THPDevice_New(device.value());
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_record(THNPEvent *self, THNPStream *stream)
{
    HANDLE_TH_ERRORS
    self->npu_event.record(stream->npu_stream);
    ASCEND_LOGI("Event: record api is successfully executed, event=%p", self->npu_event.event());
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_wait(THNPEvent *self, THNPStream *stream)
{
    HANDLE_TH_ERRORS
    {
        pybind11::gil_scoped_release no_gil;
        self->npu_event.block(stream->npu_stream);
        ASCEND_LOGI("Event: wait api is successfully executed, event=%p", self->npu_event.event());
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_query(THNPEvent *self, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    return PyBool_FromLong(self->npu_event.query());
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_elapsed_time(THNPEvent *self, THNPEvent *other)
{
    HANDLE_TH_ERRORS
    return PyFloat_FromDouble(self->npu_event.elapsed_time(other->npu_event));
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_recorded_time(THNPEvent *self, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    return PyLong_FromUnsignedLongLong(self->npu_event.recorded_time());
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_synchronize(THNPEvent *self, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    {
        pybind11::gil_scoped_release no_gil;
        self->npu_event.synchronize();
        ASCEND_LOGI("Event: synchronize api is successfully executed, event=%p", self->npu_event.event());
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THNPEvent_properties[] = {
    {"device", (getter)THNPEvent_get_device, nullptr, nullptr, nullptr},
    {"npu_event", (getter)THNPEvent_get_npu_event, nullptr, nullptr, nullptr},
    {nullptr}
};

static PyMethodDef THNPEvent_methods[] = {
    {(char*)"record", (PyCFunction)THNPEvent_record, METH_O, nullptr},
    {(char*)"wait", (PyCFunction)THNPEvent_wait, METH_O, nullptr},
    {(char*)"query", (PyCFunction)THNPEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", (PyCFunction)THNPEvent_elapsed_time, METH_O, nullptr},
    {(char*)"recorded_time", (PyCFunction)THNPEvent_recorded_time, METH_NOARGS, nullptr},
    {(char*)"synchronize", (PyCFunction)THNPEvent_synchronize, METH_NOARGS, nullptr},
    {nullptr}
};

PyTypeObject THNPEventType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch_npu._C._NPUEventBase",          /* tp_name */
    sizeof(THNPEvent),                     /* tp_basicsize */
    0,                                     /* tp_itemsize */
    (destructor)THNPEvent_dealloc,         /* tp_dealloc */
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
    THNPEvent_methods,                     /* tp_methods */
    0,                                     /* tp_members */
    THNPEvent_properties,                  /* tp_getset */
    0,                                     /* tp_base */
    0,                                     /* tp_dict */
    0,                                     /* tp_descr_get */
    0,                                     /* tp_descr_set */
    0,                                     /* tp_dictoffset */
    0,                                     /* tp_init */
    0,                                     /* tp_alloc */
    THNPEvent_pynew,                       /* tp_new */
};

void THNPEvent_init(PyObject *module)
{
    THNPEventClass = (PyObject*)&THNPEventType;
    if (PyType_Ready(&THNPEventType) < 0) {
        throw python_error();
    }
    Py_INCREF(&THNPEventType);
    if (PyModule_AddObject(module, "_NPUEventBase", (PyObject *)&THNPEventType) < 0) {
        throw python_error();
    }
}
