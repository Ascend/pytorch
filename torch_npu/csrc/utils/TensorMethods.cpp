// Copyright (c) 2022 Huawei Technologies Co., Ltd
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


#include "torch_npu/csrc/utils/TensorMethods.h"
#include "torch_npu/csrc/utils/DeviceParser.h"

namespace torch_npu {
namespace utils {

const char* _backend_to_string_npu(const at::Backend& backend) {
    switch (backend) {
        case at::Backend::CPU: return "torch";
        case at_npu::key::NativeBackend: return "torch.npu";
        default: AT_ERROR("Unimplemented backend ", backend);
    }
}

std::string _options_to_string_npu(const at::TensorOptions options) {
  std::ostringstream ss;
  ss << _backend_to_string_npu(options.backend()) << "." << toString(at::typeMetaToScalarType(options.dtype())) << "Tensor";
  return ss.str();
}

std::string _type_to_string_npu(const at::DeprecatedTypeProperties& type) {
  std::ostringstream ss;
  ss << _backend_to_string_npu(type.backend()) << "." << toString(type.scalarType()) << "Tensor";
  return ss.str();
}

std::vector<at::DeprecatedTypeProperties*> allTypesForBackends(at::ArrayRef<at::Backend> backends) {
  std::vector<at::DeprecatedTypeProperties*> res;
  res.reserve(backends.size());
  for (auto p : backends) {
    for (int64_t s = 0; s < static_cast<int64_t>(at::ScalarType::NumOptions); s++) {
      auto& type = at::getDeprecatedTypeProperties(static_cast<at::Backend>(p), static_cast<at::ScalarType>(s));
      res.emplace_back(&type);
    }
  }
  return res;
}

std::vector<at::DeprecatedTypeProperties*> allNPUTypes() {
  return allTypesForBackends({ at_npu::key::NativeBackend });
}

at::TensorOptions _options_from_string(const std::string& str) {
  static std::string cuda_prefix("torch.cuda.");
  static std::string npu_prefix("torch.npu.");
  static std::once_flag cpu_once;
  static std::once_flag cuda_once;
  static std::once_flag npu_once;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cpu_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cuda_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> npu_map;

  const std::unordered_map<std::string, at::DeprecatedTypeProperties*>* map = nullptr;

  if (str == "torch.Tensor") {
    auto backend = dispatchKeyToBackend(torch::tensors::get_default_dispatch_key());
    auto scalar_type = torch::tensors::get_default_scalar_type();
    return at::getDeprecatedTypeProperties(backend, scalar_type).options();
  }

  if (std::mismatch(cuda_prefix.begin(), cuda_prefix.end(), str.begin()).first == cuda_prefix.end()) {
    // torch.cuda. is prefix of str
    std::call_once(cuda_once, []() {
      for (auto type : torch::autograd::VariableType::allCUDATypes()) {
        cuda_map.emplace(torch::utils::type_to_string(*type), type);
      }
    });
    map = &cuda_map;
  } else if (std::mismatch(npu_prefix.begin(), npu_prefix.end(), str.begin())
          .first == npu_prefix.end()) {
    // torch.npu. is prefix of str
    std::call_once(npu_once, []() {
      for (auto type : allNPUTypes()) {
        npu_map.emplace(_type_to_string_npu(*type), type);
      }
    });
    map = &npu_map;
  } else {
    std::call_once(cpu_once, []() {
      for (auto type : torch::autograd::VariableType::allCPUTypes()) {
        cpu_map.emplace(torch::utils::type_to_string(*type), type);
      }
    });
    map = &cpu_map;
  }

  auto it = map->find(str);
  if (it == map->end()) {
    throw torch::ValueError("invalid type: '%s'", str.c_str());
  }
  return it->second->options();
}

std::tuple<at::Tensor, c10::optional<at::Device>, c10::optional<at::ScalarType>, bool, bool, c10::optional<at::MemoryFormat>> parse_to_conversion(torch::PythonArgs& r, bool allow_copy);

void InitNPUWithIndex(c10::DeviceIndex index = -1);

static at::Tensor dispatch_to(const at::Tensor & self, c10::Device device, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing. However, the behavior of aten::to
  // is different with respect to TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to(self.options().device(device).memory_format(optional_memory_format), non_blocking, copy);
}

static at::Tensor dispatch_to(const at::Tensor & self, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AutoNoGIL no_gil;
  return self.to(self.options().memory_format(optional_memory_format), non_blocking, copy);
}

static at::Tensor dispatch_to(const at::Tensor & self, c10::ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(dtype, non_blocking, copy, optional_memory_format);
}

static at::Tensor dispatch_to(const at::Tensor & self, c10::Device device, c10::ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(device, dtype, non_blocking, copy, optional_memory_format);
}

static PyObject* THPVariable_npu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "npu(Tensor temp, Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "npu(Tensor temp, Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto local_device = r.isNone(1) ? c10::Device(at_npu::key::NativeDeviceType) : r.device(1);
  auto device = c10::Device(at_npu::key::NativeDeviceType, local_device.index());
  auto opt_memory_format = r.memoryformatOptional(3);
  TORCH_CHECK((device.type() == at_npu::key::NativeDeviceType), "Invalid device, must be npu device", PTA_ERROR(ErrCode::PARAM));
  maybe_initialize_npu(device);
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_to(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "to(Tensor temp, Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor temp, ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor temp, Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
  });
  torch::ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  auto parsed = torch_npu::utils::parse_to_conversion(r, true);
  auto self_ = std::get<0>(parsed);
  auto& device = std::get<1>(parsed);
  auto& scalarType = std::get<2>(parsed);
  auto non_blocking = std::get<3>(parsed);
  auto copy = std::get<4>(parsed);
  auto opt_memory_format = std::get<5>(parsed);

  maybe_initialize_npu(device);
  if (!device && !scalarType && !copy && !opt_memory_format.has_value()) {
    Py_INCREF(self);
    return THPVariable_Wrap(self_);
  } else if (!device && !scalarType) {
    return THPVariable_Wrap(
        dispatch_to(self_, non_blocking, copy, opt_memory_format));
  } else if (!device) {
    return THPVariable_Wrap(dispatch_to(self_, *scalarType, non_blocking, copy, opt_memory_format));
  } else if (!scalarType) {
    return THPVariable_Wrap(dispatch_to(self_, *device, non_blocking, copy, opt_memory_format));
  } else {
    return THPVariable_Wrap(dispatch_to(self_, *device, *scalarType, non_blocking, copy, opt_memory_format));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "type(Tensor temp, PyObject* dtype=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "type(Tensor temp, PyObject* dtype=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.isNone(1)) {
    return THPUtils_packString(_options_to_string_npu(self_.options()));
  }
  auto obj = r.pyobject(1);
  auto opt_memory_format = r.memoryformatOptional(3);
  std::string type_name;
  bool is_dtype = false;
  if (PyType_Check(obj)) {
    if (obj == THPVariableClass) {
      type_name = "torch.Tensor";
    } else {
      type_name = ((PyTypeObject*)obj)->tp_name;
    }
  } else if (THPUtils_checkString(obj)) {
    type_name = THPUtils_unpackString(obj);
  } else if (THPDtype_Check(obj)) {
    is_dtype = true;
  } else {
    throw torch::TypeError("dtype must be a type, str, or dtype object");
  }
  c10::ScalarType scalar_type;
  c10::Device device = self_.device();
  if (is_dtype) {
    scalar_type = r.scalartype(1);
  } else {
    at::TensorOptions options = _options_from_string(type_name);
    scalar_type = at::typeMetaToScalarType(options.dtype());
    auto device_type = options.device().type();
    if (device_type != device.type()) {
      device = at::Device(device_type);
    }
  }
  maybe_initialize_npu(device);
  return THPVariable_Wrap(dispatch_to(self_, device, scalar_type, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_npu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "type(Tensor temp)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  return torch::autograd::utils::wrap(at_npu::key::isDeviceTensor(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_new_empty(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {
          "new_empty(Tensor self, IntArrayRef size, *, ScalarType dtype=None, "
          "Layout layout=torch.strided, Device device=None, bool "
          "pin_memory=False, bool requires_grad=False)",
      },
      true);
  torch::ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  auto device = at_npu::key::parse_npu_device_with_default(r.args[4], self_.device());
  maybe_initialize_npu(device);
  const auto options = at::TensorOptions()
      .dtype(r.scalartypeWithDefault(2, self_.scalar_type()))
      .device(device)
      .layout(r.layoutWithDefault(3, c10::layout_from_backend(self_.options().backend())))
      .requires_grad(r.toBool(6))
      .pinned_memory(r.toBool(5));
  auto dispatch_new_empty = [](at::Tensor & self, c10::IntArrayRef size, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_empty(size, options);
  };
  return torch::autograd::utils::wrap(dispatch_new_empty(self_, r.intlist(1), options).set_requires_grad(r.toBool(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_new_empty_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {
          "new_empty_strided(Tensor self, SymIntArrayRef size, SymIntArrayRef "
          "stride, *, ScalarType dtype=None, Layout layout=torch.strided, "
          "Device device=None, bool pin_memory=False, bool "
          "requires_grad=False)",
      },
      true);
  torch::ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  auto device = at_npu::key::parse_npu_device_with_default(r.args[5], self_.device());
  maybe_initialize_npu(device);
  const auto options = at::TensorOptions()
      .dtype(r.scalartypeWithDefault(3, self_.scalar_type()))
      .device(device)
      .layout(r.layoutWithDefault(4, c10::layout_from_backend(self_.options().backend())))
      .requires_grad(r.toBool(7))
      .pinned_memory(r.toBool(6));
  auto dispatch_new_empty_strided = [](at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_empty_strided_symint(size, stride, options);
  };
  return torch::autograd::utils::wrap(dispatch_new_empty_strided(self_, r.symintlist(1), r.symintlist(2), options).set_requires_grad(r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_record_stream(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {
          "record_stream(Tensor self, Stream s)",
      },
      false);
  torch::ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  const at::Tensor& self = _r.tensor(0);
  if (_r.has_torch_function()) {
    return torch::handle_torch_function(_r, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  struct c10::StreamData3 data = _r.stream(1).pack3();
  c10_npu::NPUCachingAllocator::recordStream(self.storage().data_ptr(),
      c10_npu::NPUStream::unpack3(data.stream_id, data.device_index, data.device_type));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_new_full(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {
          "new_full(Tensor self, IntArrayRef size, Scalar fill_value, *, ScalarType dtype=None, "
          "Layout layout=torch.strided, Device device=None, bool "
          "pin_memory=False, bool requires_grad=False) ",
      },
      true);
  torch::ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  auto device = at_npu::key::parse_npu_device_with_default(r.args[5], self_.device());
  maybe_initialize_npu(device);
  const auto options = at::TensorOptions()
      .dtype(r.scalartypeWithDefault(3, self_.scalar_type()))
      .device(device)
      .layout(r.layoutWithDefault(4, c10::layout_from_backend(self_.options().backend())))
      .requires_grad(r.toBool(7))
      .pinned_memory(r.toBool(6));
  auto dispatch_new_full = [](at::Tensor & self, c10::IntArrayRef size, c10::Scalar fill_val,
                               at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_full(size, fill_val, options);
  };
  return torch::autograd::utils::wrap(dispatch_new_full(self_, r.intlist(1), r.scalar(2), options).set_requires_grad(r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_new_zeros(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {
          "new_zeros(Tensor self, IntArrayRef size, *, ScalarType dtype=None, "
          "Layout layout=torch.strided, Device device=None, bool "
          "pin_memory=False, bool requires_grad=False)",
      },
      true);
  torch::ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  auto device = at_npu::key::parse_npu_device_with_default(r.args[4], self_.device());
  maybe_initialize_npu(device);
  const auto options = at::TensorOptions()
      .dtype(r.scalartypeWithDefault(2, self_.scalar_type()))
      .device(device)
      .layout(r.layoutWithDefault(3, c10::layout_from_backend(self_.options().backend())))
      .requires_grad(r.toBool(6))
      .pinned_memory(r.toBool(5));
  auto dispatch_new_zeros = [](at::Tensor & self, c10::IntArrayRef size, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_zeros(size, options);
  };
  return torch::autograd::utils::wrap(dispatch_new_zeros(self_, r.intlist(1), options).set_requires_grad(r.toBool(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_new_ones(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {
          "new_ones(IntArrayRef size, *, ScalarType dtype=None, "
          "Layout layout=torch.strided, Device device=None, bool "
          "pin_memory=False, bool requires_grad=False)",
      },
      true);
  PyObject* self_obj = PyTuple_GetItem(args, 0);
  PyObject* new_args = PyTuple_GetSlice(args, 1, PyTuple_GET_SIZE(args));
  torch::ParsedArgs<6> parsed_args;
  auto r = parser.parse(new_args, kwargs, parsed_args);
  auto self_ = THPVariable_Unpack(self_obj);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, new_args, kwargs, THPVariableClass, "torch.Tensor");
  }
  auto device = at_npu::key::parse_npu_device_with_default(r.args[3], self_.device());
  maybe_initialize_npu(device);
  const auto options = at::TensorOptions()
      .dtype(r.scalartypeWithDefault(1, self_.scalar_type()))
      .device(device)
      .layout(r.layoutWithDefault(2, c10::layout_from_backend(self_.options().backend())))
      .requires_grad(r.toBool(5))
      .pinned_memory(r.toBool(4));
  auto dispatch_new_ones = [](at::Tensor & self, c10::IntArrayRef size, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_ones(size, options);
  };
  return torch::autograd::utils::wrap(dispatch_new_ones(self_, r.intlist(0), options).set_requires_grad(r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_new_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  if (kwargs && PyDict_Check(kwargs) && PyDict_Contains(kwargs, THPUtils_internString("device"))) {
    PyObject* obj = PyDict_GetItem(kwargs, THPUtils_internString("device"));
    auto device = at_npu::key::parse_npu_device(obj);
    torch_npu::utils::maybe_initialize_npu(device);
    PyDict_SetItem(kwargs, THPUtils_internString("device"), THPDevice_New(device));
  }
  auto& self_ = THPVariable_Unpack(PyDict_GetItem(kwargs, THPUtils_internString("tensor")));
  TORCH_CHECK(PyDict_DelItem(kwargs, THPUtils_internString("tensor")) == 0, "Fail to Del 'tensor' in kwargs", PTA_ERROR(ErrCode::ACL));
  c10::OptionalDeviceGuard device_guard(at::device_of(self_));
  return THPVariable_Wrap(torch::utils::new_tensor(at::legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

// autograd methods on torch._C
static PyMethodDef TorchTensorMethods[] = { // NOLINT
    {"npu", castPyCFunctionWithKeywords(THPVariable_npu), METH_VARARGS | METH_KEYWORDS, NULL},
    {"to", castPyCFunctionWithKeywords(THPVariable_to), METH_VARARGS | METH_KEYWORDS, NULL},
    {"type", castPyCFunctionWithKeywords(THPVariable_type), METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_npu", castPyCFunctionWithKeywords(THPVariable_is_npu), METH_VARARGS | METH_KEYWORDS, NULL},
    {"record_stream", castPyCFunctionWithKeywords(THPVariable_record_stream), METH_VARARGS | METH_KEYWORDS, NULL},
    {"new_empty", castPyCFunctionWithKeywords(THPVariable_new_empty), METH_VARARGS | METH_KEYWORDS, NULL},
    {"new_empty_strided", castPyCFunctionWithKeywords(THPVariable_new_empty_strided), METH_VARARGS | METH_KEYWORDS, NULL},
    {"new_full", castPyCFunctionWithKeywords(THPVariable_new_full), METH_VARARGS | METH_KEYWORDS, NULL},
    {"new_zeros", castPyCFunctionWithKeywords(THPVariable_new_zeros), METH_VARARGS | METH_KEYWORDS, NULL},
    {"new_ones", castPyCFunctionWithKeywords(THPVariable_new_ones), METH_VARARGS | METH_KEYWORDS, NULL},
    {"new_tensor", castPyCFunctionWithKeywords(THPVariable_new_tensor), METH_VARARGS | METH_KEYWORDS, NULL},
    {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* tensor_functions() {
  return TorchTensorMethods;
}

std::tuple<at::Tensor, c10::optional<at::Device>, c10::optional<at::ScalarType>, bool, bool, c10::optional<at::MemoryFormat>> parse_to_conversion(torch::PythonArgs& r, bool allow_copy) {
  if (r.idx == 0) {
    if (!allow_copy && !r.isNone(4))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(r.tensor(0), at_npu::key::parse_npu_device_optional(r.args[1]), r.scalartypeOptional(2), r.toBool(3), r.toBool(4), r.memoryformatOptional(5));
  } else if (r.idx == 1) {
    if (!allow_copy && !r.isNone(4))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(r.tensor(0), c10::nullopt, r.scalartype(1), r.toBool(2), r.toBool(3), r.memoryformatOptional(4));
  } else {
    auto tensor = r.tensor(1);
    if (!allow_copy && !r.isNone(5))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(
        std::move(r.tensor(0)),
        tensor.device(),
        tensor.scalar_type(),
        r.toBool(2),
        r.toBool(3),
        r.memoryformatOptional(4)
    );
  }
}

void InitNPUWithIndex(c10::DeviceIndex index) {
  {
    pybind11::gil_scoped_release no_gil;
    auto status = c10_npu::NpuSysCtrl::GetInstance().Initialize((int)index);
    if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
      throw python_error();
    }
  }
  torch_npu::utils::npu_lazy_init();
}

}
}