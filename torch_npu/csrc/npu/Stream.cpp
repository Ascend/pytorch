#include <pybind11/pybind11.h>
#include <structmember.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>

#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/npu/Module.h"
#include "torch_npu/csrc/npu/Stream.h"

PyObject *THNPStreamClass = nullptr;

static PyObject *THNPStream_pynew(
    PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    HANDLE_TH_ERRORS

    int current_device;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&current_device));

    int is_sync_launch = 0;
    int priority = 0;
    int64_t stream_id = 0;
    int64_t device_index = 0;
    int64_t device_type = 0;
    uint64_t stream_ptr = 0;

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    constexpr const char *kwlist[] = {
        "priority",
        "is_sync_launch",
        "stream_id",
        "device_index",
        "device_type",
        "stream_ptr",
        nullptr};
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "|iiLLLK",
        const_cast<char **>(kwlist),
        &priority,
        &is_sync_launch,
        &stream_id,
        &device_index,
        &device_type,
        &stream_ptr)) {
        return nullptr;
    }

    THPObjectPtr ptr(type->tp_alloc(type, 0));
    if (!ptr) {
        return nullptr;
    }

    c10_npu::NPUStream stream =
        (stream_id || device_index || device_type) ?
        c10_npu::NPUStream::unpack3(
            stream_id, device_index, static_cast<c10::DeviceType>(device_type)) :
        (is_sync_launch ? c10_npu::getNPUStreamFromSyncLaunchPool() :
        c10_npu::getNPUStreamFromPool());

    THNPStream *self = (THNPStream *)ptr.get();
    self->stream_id = static_cast<int64_t>(stream.id());
    self->device_index = static_cast<int64_t>(stream.device_index());
    self->device_type = static_cast<int64_t>(stream.device_type());
    new (&self->npu_stream) c10_npu::NPUStream(stream);

    return (PyObject *)ptr.release();
    END_HANDLE_TH_ERRORS
}

static void THNPStream_dealloc(THNPStream *self)
{
    self->npu_stream.~NPUStream();
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THNPStream_get_device(THNPStream *self, void *unused)
{
    HANDLE_TH_ERRORS
    return THPDevice_New(self->npu_stream.device());
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_get_npu_stream(THNPStream *self, void *unused)
{
    HANDLE_TH_ERRORS
    return PyLong_FromVoidPtr(self->npu_stream.stream());
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_get_priority(THNPStream *self, void *unused)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(false, "NPU dose not support Stream.get_priority() currently.", PTA_ERROR(ErrCode::NOT_SUPPORT));
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_priority_range()
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(false, "NPU does not support Stream.priority_range() currently.", PTA_ERROR(ErrCode::NOT_SUPPORT));
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_query(THNPStream *self, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    return PyBool_FromLong(self->npu_stream.query());
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_synchronize(THNPStream *self, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    {
        pybind11::gil_scoped_release no_gil;
        self->npu_stream.synchronize();
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_set_data_preprocess_stream(THNPStream *self, PyObject *arg)
{
    HANDLE_TH_ERRORS
    {
        pybind11::gil_scoped_release no_gil;
        bool is_data_preprocess_stream = THPUtils_unpackBool(arg);
        self->npu_stream.setDataPreprocessStream(is_data_preprocess_stream);
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPStream_eq(THNPStream *self, THNPStream *other)
{
    HANDLE_TH_ERRORS
    return PyBool_FromLong(self->npu_stream == other->npu_stream);
    END_HANDLE_TH_ERRORS
}

static struct PyMemberDef THNPStream_members[] = {
    {(char*)"stream_id", T_ULONGLONG, offsetof(THNPStream, stream_id), READONLY, nullptr},
    {(char*)"device_type", T_ULONGLONG, offsetof(THNPStream, device_type), READONLY, nullptr},
    {(char*)"device_index", T_ULONGLONG, offsetof(THNPStream, device_index), READONLY, nullptr},
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
  THNPStream_methods,                    /* tp_methods */
  THNPStream_members,                    /* tp_members */
  THNPStream_properties,                /* tp_getset */
  nullptr,                               /* tp_base */
  nullptr,                               /* tp_dict */
  nullptr,                               /* tp_descr_get */
  nullptr,                               /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  nullptr,                               /* tp_init */
  nullptr,                               /* tp_alloc */
  THNPStream_pynew,                      /* tp_new */
};

void THNPStream_init(PyObject *module)
{
    Py_INCREF(THPStreamClass);
    THNPStreamType.tp_base = THPStreamClass;
    THNPStreamClass = (PyObject *)&THNPStreamType;
    if (PyType_Ready(&THNPStreamType) < 0) {
        throw python_error();
    }
    Py_INCREF(&THNPStreamType);
    if (PyModule_AddObject(module, "_NPUStreamBase", (PyObject *)&THNPStreamType) < 0) {
        throw python_error();
    }
}

std::vector<c10::optional<c10_npu::NPUStream>> THNPUtils_PySequence_to_NPUStreamList(PyObject* obj)
{
    if (!PySequence_Check(obj)) {
        throw std::runtime_error("Expected a sequence in THNPUtils_PySequence_to_NPUStreamList" + PTA_ERROR(ErrCode::PARAM));
    }
    THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, nullptr));
    if (seq.get() == nullptr) {
        throw std::runtime_error("expected PySequence, but got " + std::string(THPUtils_typename(obj)) + PTA_ERROR(ErrCode::PARAM));
    }

    std::vector<c10::optional<c10_npu::NPUStream>> streams;
    Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject* stream = PySequence_Fast_GET_ITEM(seq.get(), i);

        if (PyObject_IsInstance(stream, THNPStreamClass)) {
            streams.emplace_back(c10_npu::NPUStream::unpack3(
                (reinterpret_cast<THNPStream*>(stream))->stream_id,
                (reinterpret_cast<THNPStream*>(stream))->device_index,
                static_cast<c10::DeviceType>((reinterpret_cast<THNPStream*>(stream))->device_type)));
        } else if (stream == Py_None) {
            streams.emplace_back();
        } else {
            std::runtime_error("Unknown data type found in stream list. Need torch_npu.npu.Stream or None" + PTA_ERROR(ErrCode::TYPE));
        }
    }
    return streams;
}

c10_npu::NPUStream THNPUtils_PyObject_to_NPUStream(PyObject* stream)
{
    TORCH_CHECK(PyObject_IsInstance(stream, THNPStreamClass), "Need torch_npu.npu.Stream argument type.");
    return c10_npu::NPUStream::unpack3(
        (reinterpret_cast<THNPStream *>(stream))->stream_id,
        (reinterpret_cast<THNPStream *>(stream))->device_index,
        static_cast<c10::DeviceType>((reinterpret_cast<THNPStream *>(stream))->device_type));
}
