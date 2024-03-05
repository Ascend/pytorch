// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

// required for old g++ to compile PRId64 macros
// for context

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/Device.h>
#include <c10/util/Exception.h>

#include <cstring>
#include <limits>
#include <structmember.h>
#include <sstream>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/utils/Device.h"
#include "torch_npu/csrc/utils/DeviceParser.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

PyObject *TNPDevice_New(const at::Device& device)
{
  auto type = (PyTypeObject*)&TNPDeviceType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) {
      throw python_error();
  }
  auto self_ = reinterpret_cast<TNPDevice*>(self.get());
  self_->device = device;
  return self.release();
}

PyObject *TNPDevice_repr(TNPDevice *self)
{
  std::ostringstream oss;
  if (self->device.type() == at_npu::key::NativeDeviceType) {
    oss << "device(type=\'" << at_npu::key::npu_device_str << "\'";
  } else {
    oss << "device(type=\'" << self->device.type() << "\'";
  }
  
  if (self->device.has_index()) {
    // `self->device.index()` returns uint8_t which is treated as ascii while printing,
    // hence casting it to uint16_t.
    oss << ", index=" << static_cast<uint16_t>(self->device.index());
  }
  oss << ")";
  return THPUtils_packString(oss.str().c_str());
}

PyObject *TNPDevice_str(TNPDevice *self)
{
  std::ostringstream oss;
    std::string str = c10::DeviceTypeName(self->device.type(), true);
    if (at_npu::key::default_device_str == str) {
        str = at_npu::key::npu_device_str;
    }
    if (self->device.has_index()) {
        str.push_back(':');
        str.append(std::to_string(self->device.index()));
    }
    oss << str;
  return THPUtils_packString(oss.str().c_str());
}

PyObject *TNPDevice_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "Device(PyObject* device)",
    "Device(std::string type, int64_t? index=-1)"
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto as_device = at_npu::key::parse_npu_device(r.pyobject(0));
    return TNPDevice_New(as_device);
  } else {
    auto as_device = at_npu::key::parse_npu_device(r.args[0]);  // this works, because device can take strings
    auto device_type = r.string(0);
    if (as_device.has_index() && !r.isNone(1)) {
      throw std::runtime_error("type (string) must not include an index because index "
                                "was passed explicitly: " + device_type);
    }
    int32_t device_index = as_device.has_index() ? as_device.index() : -1;
    if (!r.isNone(1)) {
      device_index = r.toInt64(1);
      // -1 is allowed in ATen/C++, to mean the default device, but not in
      // Python.
      TORCH_CHECK(device_index >= 0, "Device index must not be negative", PTA_ERROR(ErrCode::VALUE));
    }
    at::Device device(as_device.type(), device_index);
    return TNPDevice_New(device);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *TNPDevice_type(TNPDevice *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  std::ostringstream oss;
  if (self->device.type() == at_npu::key::NativeDeviceType) {
    oss << at_npu::key::npu_device_str;
  } else {
    oss << self->device.type();
  }
  return THPUtils_packString(oss.str().c_str());
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *TNPDevice_index(TNPDevice *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  if (self->device.has_index()) {
    return THPUtils_packInt64(self->device.index());
  } else {
    Py_RETURN_NONE;
  }
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t TNPDevice_hash(TNPDevice *self)
{
  HANDLE_TH_ERRORS
  return static_cast<Py_ssize_t>(static_cast<Py_ssize_t>(std::hash<at::Device>{}(self->device)) % std::numeric_limits<Py_ssize_t>::max());
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *TNPDevice_rc(PyObject *a, PyObject *b, int op) {
    HANDLE_TH_ERRORS
    if (!TNPDevice_Check(a) || !TNPDevice_Check(b)) {
        // Py_RETURN_NOTIMPLEMENTED not in python 2.
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    TNPDevice *da = reinterpret_cast<TNPDevice*>(a);
    TNPDevice *db = reinterpret_cast<TNPDevice*>(b);

    switch (op) {
        case Py_EQ:
            if (da->device == db->device) {
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
        case Py_NE:
            if (da->device == db->device) {
                Py_RETURN_FALSE;
            } else {
                Py_RETURN_TRUE;
            }
        case Py_LT:
        case Py_LE:
        case Py_GT:
        case Py_GE:
            throw torch::TypeError("comparison not implemented");
        default:
            throw torch::TypeError("unexpected comparison op");
    }
    END_HANDLE_TH_ERRORS
}

PyObject *TNPDevice_reduce(PyObject *_self, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    auto self = (TNPDevice*)_self;
    auto ret = THPObjectPtr{PyTuple_New(2)};
    if (!ret) {
        throw python_error();
    }

    py::object torch_module = py::module::import("torch_npu._C");
    py::object torch_device = torch_module.attr("device");
    PyTuple_SET_ITEM(ret.get(), 0, torch_device.release().ptr());

    THPObjectPtr args;
    std::ostringstream oss;
    if (self->device.type() == at_npu::key::NativeDeviceType) {
      oss << at_npu::key::npu_device_str;
    } else {
      oss << self->device.type();
    }
    if (self->device.has_index()) {
      args = THPObjectPtr{Py_BuildValue("(si)", oss.str().c_str(), self->device.index())};
    } else {
      args = THPObjectPtr{Py_BuildValue("(s)", oss.str().c_str())};
    }
    if (!args) {
        throw python_error();
    }
    PyTuple_SET_ITEM(ret.get(), 1, args.release());

    return ret.release();
    END_HANDLE_TH_ERRORS
}

using getter = PyObject* (*)(PyObject *, void *);

// NB: If you edit these properties/methods, update torch/_C/__init__.pyi.in

static struct PyGetSetDef TNPDevice_properties[] = {
    {"type",       (getter)TNPDevice_type, nullptr, nullptr, nullptr},
    {"index",      (getter)TNPDevice_index, nullptr, nullptr, nullptr},
    {nullptr}
};

static PyMethodDef TNPDevice_methods[] = {
    {"__reduce__", TNPDevice_reduce, METH_NOARGS, nullptr},
    {nullptr}  /* Sentinel */
};

PyTypeObject TNPDeviceType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch_npu._C.device",                        /* tp_name */
  sizeof(TNPDevice),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  nullptr,                               /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  nullptr,                               /* tp_getattr */
  nullptr,                               /* tp_setattr */
  nullptr,                               /* tp_reserved */
  (reprfunc)TNPDevice_repr,              /* tp_repr */
  nullptr,                               /* tp_as_number */
  nullptr,                               /* tp_as_sequence */
  nullptr,                               /* tp_as_mapping */
  (hashfunc)TNPDevice_hash,              /* tp_hash  */
  nullptr,                               /* tp_call */
  (reprfunc)TNPDevice_str,               /* tp_str */
  nullptr,                               /* tp_getattro */
  nullptr,                               /* tp_setattro */
  nullptr,                               /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  nullptr,                               /* tp_doc */
  nullptr,                               /* tp_traverse */
  nullptr,                               /* tp_clear */
  (richcmpfunc)TNPDevice_rc,             /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  nullptr,                               /* tp_iter */
  nullptr,                               /* tp_iternext */
  TNPDevice_methods,                     /* tp_methods */
  nullptr,                               /* tp_members */
  TNPDevice_properties,                  /* tp_getset */
  nullptr,                               /* tp_base */
  nullptr,                               /* tp_dict */
  nullptr,                               /* tp_descr_get */
  nullptr,                               /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  nullptr,                               /* tp_init */
  nullptr,                               /* tp_alloc */
  TNPDevice_pynew,                       /* tp_new */
};

void TNPDevice_init(PyObject *module)
{
  if (PyType_Ready(&TNPDeviceType) < 0) {
    throw python_error();
  }
  Py_INCREF(&TNPDeviceType);
  if (PyModule_AddObject(module, "device", (PyObject *)&TNPDeviceType) != 0) {
    throw python_error();
  }
}
