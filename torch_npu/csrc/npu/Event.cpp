#include "torch_npu/csrc/npu/Event.h"

#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <structmember.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"

#include "torch_npu/csrc/npu/Stream.h"

#define ACL_EVENT_DEFAULT 0x0000000Eu

PyObject *THNPEventClass = nullptr;

static PyObject* THNPEvent_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    HANDLE_TH_ERRORS
    unsigned char enable_timing = 0;
    unsigned char blocking = 0;
    unsigned char interprocess = 0;
    unsigned char external = 0;

    constexpr const char* kwlist[] = {"enable_timing", "blocking", "interprocess", "graph_external", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|bbbb", const_cast<char**>(kwlist),
        &enable_timing, &blocking, &interprocess, &external)) {
        return nullptr;
    }

    TORCH_CHECK(
        !interprocess || c10_npu::acl::IsSupportIpcEvent(),
        "Parameter interprocess is not supported, please upgrade the HDK(driver) or CANN package.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));

    // Runtime requires that the ACL_EVENT_IPC cannot be used together with other flags.
    // If this restriction is removed in the future, this check can be removed.
    TORCH_CHECK(
        !interprocess || (!enable_timing && !external),
        "Parameter interprocess cannot be specified together with other parameters.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));

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
    if (interprocess) {
        flags = ACL_EVENT_IPC;
    }
    if (external) {
        flags = ACL_EVENT_EXTERNAL;
    }
    new (&self->npu_event) c10_npu::NPUEvent(flags);

    return (PyObject *)ptr.release();
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_from_ipc_handle(
    PyObject* _type,
    PyObject* args,
    PyObject* kwargs)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(
        c10_npu::acl::IsSupportIpcEvent(),
        "The from_ipc_handle method is not supported, please upgrade the HDK(driver) or CANN package.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));

    auto type = (PyTypeObject*)_type;

    static torch::PythonArgParser parser({
        "from_ipc_handle(Device device, std::string ipc_handle)",
    });
    constexpr int kArgCount = 2;
    torch::ParsedArgs<kArgCount> parsed_args;
    auto r = parser.parse(args, kwargs, parsed_args);

    at::Device device = r.device(0);
    std::string handle_string = r.string(1);

    TORCH_CHECK(
        handle_string.size() == sizeof(aclrtIpcEventHandle),
        "aclrtIpcEventHandle expects byte-like object of size ",
        sizeof(aclrtIpcEventHandle),
        ", but got ",
        handle_string.size(),
        PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        device.type() == c10::DeviceType::PrivateUse1,
        "Event can only be created on "
        "PrivateUse1 devices, but got device type ",
        device.type(),
        PTA_ERROR(ErrCode::PARAM))

    THPObjectPtr ptr(type->tp_alloc(type, 0));
    if (!ptr) {
        return nullptr;
    }
    THNPEvent* self = (THNPEvent*)ptr.get();

    aclrtIpcEventHandle handle{};
    std::memcpy(&handle, handle_string.c_str(), handle_string.size());
    new (&self->npu_event) c10_npu::NPUEvent(device.index(), &handle);

    return (PyObject*)ptr.release();
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

static PyObject* THNPEvent_reset(THNPEvent *self, THNPStream *stream)
{
    HANDLE_TH_ERRORS
    {
        pybind11::gil_scoped_release no_gil;
        self->npu_event.reset(stream->npu_stream);
        ASCEND_LOGI("Event: reset api is successfully executed, event=%p", self->npu_event.event());
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPEvent_ipc_handle(PyObject* _self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    auto self = (THNPEvent*)_self;
    aclrtIpcEventHandle handle{};
    self->npu_event.ipc_handle(&handle);
    return PyBytes_FromStringAndSize((const char*)&handle, sizeof(handle));
    END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THNPEvent_properties[] = {
    {"device", (getter)THNPEvent_get_device, nullptr, nullptr, nullptr},
    {"npu_event", (getter)THNPEvent_get_npu_event, nullptr, nullptr, nullptr},
    {nullptr}
};

static PyMethodDef THNPEvent_methods[] = {
    {(char*)"from_ipc_handle",
     (PyCFunction)THNPEvent_from_ipc_handle,
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {(char*)"record", (PyCFunction)THNPEvent_record, METH_O, nullptr},
    {(char*)"wait", (PyCFunction)THNPEvent_wait, METH_O, nullptr},
    {(char*)"query", (PyCFunction)THNPEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", (PyCFunction)THNPEvent_elapsed_time, METH_O, nullptr},
    {(char*)"recorded_time", (PyCFunction)THNPEvent_recorded_time, METH_NOARGS, nullptr},
    {(char*)"synchronize", (PyCFunction)THNPEvent_synchronize, METH_NOARGS, nullptr},
    {(char*)"reset", (PyCFunction)THNPEvent_reset, METH_O, nullptr},
    {(char*)"ipc_handle", THNPEvent_ipc_handle, METH_NOARGS, nullptr},
    {nullptr}
};

PyTypeObject THNPEventType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch_npu._C._NPUEventBase",          /* tp_name */
    sizeof(THNPEvent),                     /* tp_basicsize */
    0,                                     /* tp_itemsize */
    (destructor)THNPEvent_dealloc,         /* tp_dealloc */
    0,                                     /* tp_vectorcall_offset */
    nullptr,                               /* tp_getattr */
    nullptr,                               /* tp_setattr */
    nullptr,                               /* tp_reserved */
    nullptr,                               /* tp_repr */
    nullptr,                               /* tp_as_number */
    nullptr,                               /* tp_as_sequence */
    nullptr,                               /* tp_as_mapping */
    nullptr,                               /* tp_hash  */
    nullptr,                               /* tp_call */
    nullptr,                               /* tp_str */
    nullptr,                               /* tp_getattro */
    nullptr,                               /* tp_setattro */
    nullptr,                               /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr,                                  /* tp_doc */
    nullptr,                               /* tp_traverse */
    nullptr,                               /* tp_clear */
    nullptr,                               /* tp_richcompare */
    0,                                     /* tp_weaklistoffset */
    nullptr,                               /* tp_iter */
    nullptr,                               /* tp_iternext */
    THNPEvent_methods,                     /* tp_methods */
    nullptr,                               /* tp_members */
    THNPEvent_properties,                  /* tp_getset */
    nullptr,                               /* tp_base */
    nullptr,                               /* tp_dict */
    nullptr,                               /* tp_descr_get */
    nullptr,                               /* tp_descr_set */
    0,                                     /* tp_dictoffset */
    nullptr,                               /* tp_init */
    nullptr,                               /* tp_alloc */
    THNPEvent_pynew,                       /* tp_new */
};

void THNPEvent_init(PyObject *module)
{
    TORCH_CHECK(THPEventClass, "THPEvent has not been initialized yet.");
    Py_INCREF(THPEventClass);
    THNPEventType.tp_base = THPEventClass;
    THNPEventClass = (PyObject*)&THNPEventType;
    if (PyType_Ready(&THNPEventType) < 0) {
        throw python_error();
    }
    Py_INCREF(&THNPEventType);
    if (PyModule_AddObject(module, "_NPUEventBase", (PyObject *)&THNPEventType) < 0) {
        throw python_error();
    }
}

c10_npu::NPUEvent* THNPUtils_PyObject_to_NPUEvent(PyObject* py_event)
{
    TORCH_CHECK(py_event != nullptr, "Expected py_event is a non-null PyObject pointer.", PTA_ERROR(ErrCode::PARAM));
    TORCH_CHECK(PyObject_IsInstance(py_event, THNPEventClass), "Need torch_npu.npu.Event argument type.", PTA_ERROR(ErrCode::PARAM));
    THNPEvent* th_event = reinterpret_cast<THNPEvent *>(py_event);
    return &th_event->npu_event;
}
