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

// ${generated_comment}

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch
// and also copied into 'torch' module.

#include <Python.h>

// Undefine the copysign macro so that at::copysign works as intended with MSVC
#ifdef _MSC_VER
#undef copysign
#endif // _MSC_VER

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/out_types.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/tensor_layouts.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/utils/structseq.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <utility>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/utils/LazyInit.h"
#include "torch_npu/csrc/utils/DeviceParser.h"
#include "torch_npu/csrc/aten/VariableType.h"
#include "torch_npu/csrc/framework/autograd/wrap_outputs.h"
#include "op_plugin/OpInterface.h"

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;
using at::ArrayRef;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

namespace torch_npu { namespace autograd {

static PyObject* THPVariableFunctionsModule = NULL;

// generated forward declarations start here

${py_forwards}

${py_device_forwards}

// Wrapper converts a raised TypeError into returning NotImplemented
// Used to implement binary arithmetic operators
template <PyObject* (*Func)(PyObject*, PyObject*, PyObject*)>
static PyObject * TypeError_to_NotImplemented_(PyObject* self, PyObject* args, PyObject* kwargs) {
  PyObject* ret = Func(self, args, kwargs);
  if (!ret && PyErr_ExceptionMatches(PyExc_TypeError)) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    ret = Py_NotImplemented;
  }
  return ret;
}


inline Tensor dispatch_arange(Scalar end, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::arange_out(result, end);
}

inline Tensor dispatch_arange(Scalar end, const TensorOptions& options) {
  torch_npu::utils::maybe_initialize_npu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::arange(end, options);
}

inline Tensor dispatch_arange(Scalar start, Scalar end, Scalar step, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::arange_out(result, start, end, step);
}

inline Tensor dispatch_arange(Scalar start, Scalar end, Scalar step, const TensorOptions& options) {
  torch_npu::utils::maybe_initialize_npu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::arange(start, end, step, options);
}


static PyObject * THPVariable_arange(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "arange(Scalar end, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "arange(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
      }, true);

  torch::ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  if (r.idx == 0) {
    auto device  = at_npu::key::parse_npu_device(r.args[4]);
    if (r.isNone(1)) {
      auto end = r.scalar(0);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      c10::optional<ScalarType> scalarType = r.scalartypeOptional(2);
      const auto options = TensorOptions()
          .dtype(scalarType)
          .device(device)
          .layout(r.layout(3))
          .requires_grad(r.toBool(6))
          .pinned_memory(r.toBool(5));
      return torch::autograd::utils::wrap(dispatch_arange(end, options));
    } else {
      TORCH_CHECK(!r.toBool(5), " `pin_memory` and `out` parameters are incompatible", OPS_ERROR(ErrCode::PARAM));
      check_out_type_matches(r.tensor(1), r.scalartype(2), r.isNone(2), r.layout(3),
                             device, r.isNone(4));
      return torch::autograd::utils::wrap(dispatch_arange(r.scalar(0), r.tensor(1)).set_requires_grad(r.toBool(6)));
    }
  } else if (r.idx == 1) {
    auto device = at_npu::key::parse_npu_device(r.args[6]);
    if (r.isNone(3)) {
      auto start = r.scalar(0);
      auto end = r.scalar(1);
      auto step = r.scalar(2);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      c10::optional<ScalarType> scalarType = r.scalartypeOptional(4);
      const auto options = TensorOptions()
          .dtype(scalarType)
          .device(device)
          .layout(r.layout(5))
          .requires_grad(r.toBool(8))
          .pinned_memory(r.toBool(7));
      return torch::autograd::utils::wrap(dispatch_arange(start, end, step, options));
    } else {
      TORCH_CHECK(!r.toBool(7), " `pin_memory` and `out` parameters are incompatible", OPS_ERROR(ErrCode::PARAM));
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4), r.layout(5), device, r.isNone(6));
      return torch::autograd::utils::wrap(dispatch_arange(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)).set_requires_grad(r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_range(Scalar start, Scalar end, Scalar step, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(at::device_of(result));
  return at::range_out(result, start, end, step);
}

inline Tensor dispatch_range(Scalar start, Scalar end, Scalar step, const TensorOptions& options) {
  torch_npu::utils::maybe_initialize_npu(options);
  pybind11::gil_scoped_release no_gil;
  DeviceGuard device_guard(options.device());
  return torch::range(start, end, step, options);
}

static PyObject * THPVariable_range(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "range(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  torch::ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = at_npu::key::parse_npu_device(r.args[6]);
  if (r.idx == 0) {
    auto ret = PyErr_WarnEx(
        PyExc_UserWarning,
        "torch.range is deprecated and will be removed in a future release "
        "because its behavior is inconsistent with Python's range builtin. "
        "Instead, use torch.arange, which produces values in [start, end).",
        1);
    if (ret != 0) throw python_error();
    if (r.isNone(3)) {
      const auto options = TensorOptions()
          .dtype(r.scalartype(4))
          .device(device)
          .layout(r.layout(5))
          .requires_grad(r.toBool(7));
      return torch::autograd::utils::wrap(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), device, r.isNone(6));
      return torch::autograd::utils::wrap(dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_full(
    IntArrayRef size,
    Scalar fill_val,
    const TensorOptions& options) {
  torch_npu::utils::maybe_initialize_npu(options);
  pybind11::gil_scoped_release no_gil;
  return at::full(size, fill_val, options);
}

inline Tensor dispatch_full(
    IntArrayRef size,
    Scalar fill_val,
    c10::optional<DimnameList> names,
    const TensorOptions& options) {
  torch_npu::utils::maybe_initialize_npu(options);
  pybind11::gil_scoped_release no_gil;
  return at::full(size, fill_val, names, options);
}

inline Tensor dispatch_full(
    IntArrayRef size,
    Scalar fill_val,
    Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::full_out(result, size, fill_val);
}

static PyObject * THPVariable_full(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({
    "full(IntArrayRef size, Scalar fill_value, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
    "full(IntArrayRef size, Scalar fill_value, *, DimnameList names=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
      }, true);

  // Acquires (common) arguments
  torch::ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  auto size = r.intlist(0);
  auto fill_val = r.scalar(1);
  auto device = at_npu::key::parse_npu_device(r.args[5]);
  const auto options = TensorOptions{}
      .dtype(r.scalartypeOptional(3))
      .layout(r.layout(4))
      .device(device)
      .pinned_memory(r.toBool(6));

  if (r.idx == 0) {
    // full
    if (r.isNone(2)) {
      return torch::autograd::utils::wrap(dispatch_full(size, fill_val, options).set_requires_grad(r.toBool(7)));
    }

    // full.out
    // Validates out tensor and other kwargs
    auto result = r.tensor(2);
    TORCH_CHECK(!r.toBool(6), " `pin_memory` and `out` parameters are incompatible", OPS_ERROR(ErrCode::PARAM));
    check_out_type_matches(result, r.scalartype(3), r.isNone(3), r.layout(4), device, r.isNone(5));

    return torch::autograd::utils::wrap(dispatch_full(size, fill_val, result).set_requires_grad(r.toBool(7)));
  } else if (r.idx == 1) {
    // full.names
    if (r.isNone(2)) {
      return torch::autograd::utils::wrap(dispatch_full(size, fill_val, c10::nullopt, options).set_requires_grad(r.toBool(7)));
    }

    // Converts from c10::optional<std:vector...> to c10::optional<ArrayRef...>
    auto raw_names = r.toDimnameListOptional(2);
    c10::optional<DimnameList> names(*raw_names);
    return torch::autograd::utils::wrap(dispatch_full(size, fill_val, names, options).set_requires_grad(r.toBool(7)));
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

inline Tensor dispatch_randint(int64_t high, IntArrayRef size, c10::optional<Generator> generator, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, high, size, generator);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  torch_npu::utils::maybe_initialize_npu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, high, size);
}
inline Tensor dispatch_randint(int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch_npu::utils::maybe_initialize_npu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(high, size, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, low, high, size, generator);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  torch_npu::utils::maybe_initialize_npu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(low, high, size, generator, options);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, Tensor result) {
  pybind11::gil_scoped_release no_gil;
  return at::randint_out(result, low, high, size);
}
inline Tensor dispatch_randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options) {
  torch_npu::utils::maybe_initialize_npu(options);
  pybind11::gil_scoped_release no_gil;
  return torch::randint(low, high, size, options);
}

static PyObject * THPVariable_randint(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "randint(int64_t high, IntArrayRef size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
    "randint(int64_t low, int64_t high, IntArrayRef size, *, Generator generator=None, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
      }, false);

  torch::ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if(r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  if (r.idx == 0) {
    auto device = at_npu::key::parse_npu_device(r.args[6]);
    if (r.isNone(3)) {
      auto high = r.toInt64(0);
      auto size = r.intlist(1);
      auto generator = r.generator(2);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(4, at::ScalarType::Long);
      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(5))
          .requires_grad(r.toBool(7));
      return torch::autograd::utils::wrap(dispatch_randint(high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(3), r.scalartype(4), r.isNone(4),
                             r.layout(5), device, r.isNone(6));
      return torch::autograd::utils::wrap(dispatch_randint(r.toInt64(0), r.intlist(1), r.generator(2), r.tensor(3)).set_requires_grad(r.toBool(7)));
    }
  } else if (r.idx == 1) {
    auto device = at_npu::key::parse_npu_device(r.args[7]);
    if (r.isNone(4)) {
      auto low = r.toInt64(0);
      auto high = r.toInt64(1);
      auto size = r.intlist(2);
      auto generator = r.generator(3);
      // NOTE: r.scalartype(X) gives the default dtype if r.isNone(X)
      auto dtype = r.scalartypeWithDefault(5, at::ScalarType::Long);

      const auto options = TensorOptions()
          .dtype(dtype)
          .device(device)
          .layout(r.layout(6))
          .requires_grad(r.toBool(8));
      return torch::autograd::utils::wrap(dispatch_randint(low, high, size, generator, options));
    } else {
      check_out_type_matches(r.tensor(4), r.scalartype(5), r.isNone(5), r.layout(6), device, r.isNone(7));
      return torch::autograd::utils::wrap(dispatch_randint(r.toInt64(0), r.toInt64(1), r.intlist(2), r.generator(3), r.tensor(4)).set_requires_grad(r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  if (kwargs && PyDict_Check(kwargs) && PyDict_Contains(kwargs, THPUtils_internString("device"))) {
    PyObject* obj = PyDict_GetItem(kwargs, THPUtils_internString("device"));
    auto device = at_npu::key::parse_npu_device(obj);
    torch_npu::utils::maybe_initialize_npu(device);
    PyDict_SetItem(kwargs, THPUtils_internString("device"), THPDevice_New(device));
  }
  return THPVariable_Wrap(torch::utils::tensor_ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject *THPVariable_new_device(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "Device(std::string type, int64_t? index=-1)",
    "Device(Device device)"
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 1) {
    auto device = at_npu::key::parse_npu_device(r.args[0]);
    return THPDevice_New(device);
  } else if (r.idx == 0) {
    auto as_device = at_npu::key::parse_npu_device(r.args[0]);  // this works, because device can take strings
    auto device_type = r.string(0);
    int32_t device_index = -1;
    if (!r.isNone(1)) {
      device_index = r.toInt64(1);
      // -1 is allowed in ATen/C++, to mean the default device, but not in
      // Python.
      TORCH_CHECK(device_index >= 0, "Device index must not be negative", OPS_ERROR(ErrCode::VALUE));
    }
    if (as_device.has_index()) {
      if (device_index != -1) {
        throw std::runtime_error("type (string) including an index but not equal to index: " 
                                  + device_type + ", argument index = " + std::to_string(device_index));
      } else {
        device_index = as_device.index();
      }

    }
    at::Device device(as_device.type(), device_index);
    return THPDevice_New(device);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
static PyMethodDef torch_functions[] = {
  {"tensor", castPyCFunctionWithKeywords(THPVariable_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"new_device", castPyCFunctionWithKeywords(THPVariable_new_device), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"arange", castPyCFunctionWithKeywords(THPVariable_arange), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"full", castPyCFunctionWithKeywords(THPVariable_full), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"randint", castPyCFunctionWithKeywords(THPVariable_randint), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"range", castPyCFunctionWithKeywords(THPVariable_range), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  ${py_method_defs}
  ${py_device_method_defs}
  {NULL}
};

static PyTypeObject THPVariableFunctions = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch_npu._C._VariableFunctionsClass",    /* tp_name */
  0,                                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
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
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  torch_functions,                       /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  0                                      /* tp_new */
};

TORCH_NPU_API void initTorchFunctions(PyObject* module) {
  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPVariableFunctions);

  // Steals
  Py_INCREF(&THPVariableFunctions);
  if (PyModule_AddObject(module, "_VariableFunctionsClass", reinterpret_cast<PyObject*>(&THPVariableFunctions)) < 0) {
    throw python_error();
  }
  // PyType_GenericNew returns a new reference
  THPVariableFunctionsModule = PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(module, "_VariableFunctions", THPVariableFunctionsModule) < 0) {
    throw python_error();
  }
}

// generated methods start here

${py_methods}

${py_device_methods}

}} // namespace torch_npu::autograd
