#include "torch_npu/csrc/utils/TensorMethods.h"

namespace torch_npu {
namespace utils {

static const char* _backend_to_string_npu(const at::Backend& backend) {
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

static PyObject * THPVariable_npu(PyObject* self, PyObject* args, PyObject* kwargs)
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
  TORCH_CHECK((device.type() == at_npu::key::NativeDeviceType), "Invalid device, must be npu device");
  maybe_initialize_npu(device);
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to(PyObject* self, PyObject* args, PyObject* kwargs)
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

static PyObject * THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "type(Tensor temp, PyObject* dtype=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "type(Tensor temp, PyObject* dtype=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if(r.has_torch_function()){
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
    at::TensorOptions options = torch::utils::options_from_string(type_name);
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

static PyObject * THPVariable_is_npu(PyObject* self, PyObject* args, PyObject* kwargs)
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

static PyObject * THPVariable_new_empty(PyObject* self, PyObject* args, PyObject* kwargs)
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

static PyObject * THPVariable_new_empty_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {
          "new_empty_strided(Tensor self, IntArrayRef size, IntArrayRef "
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
  auto dispatch_new_empty_strided = [](at::Tensor & self, c10::IntArrayRef size, c10::IntArrayRef stride, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_empty_strided(size, stride, options);
  };
  return torch::autograd::utils::wrap(dispatch_new_empty_strided(self_, r.intlist(1), r.intlist(2), options).set_requires_grad(r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_record_stream(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  PyObject *_tensor, *_stream;
  if (!PyArg_ParseTuple(args, "OO", &_tensor, &_stream)) {
    throw torch::TypeError("record_stream useage: tensor.record_stream(stream)");
  }
  auto& self_ = THPVariable_Unpack(_tensor);
  c10_npu::NPUCachingAllocator::recordStream(self_.storage().data_ptr(), c10_npu::NPUStream::unpack(((THNPStream*)_stream)->cdata));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_full(PyObject* self, PyObject* args, PyObject* kwargs)
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

static PyObject * THPVariable_new_zeros(PyObject* self, PyObject* args, PyObject* kwargs)
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

// autograd methods on torch._C
static PyMethodDef TorchTensorMethods[] = { // NOLINT
  {"npu", castPyCFunctionWithKeywords(THPVariable_npu), METH_VARARGS | METH_KEYWORDS, NULL},
  {"to", castPyCFunctionWithKeywords(THPVariable_to), METH_VARARGS | METH_KEYWORDS, NULL},
  {"type", castPyCFunctionWithKeywords(THPVariable_type), METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_npu", castPyCFunctionWithKeywords(THPVariable_is_npu), METH_VARARGS | METH_KEYWORDS, NULL},
  {"record_stream", (PyCFunction)(void(*)(void))THPVariable_record_stream, METH_VARARGS, NULL},
  {"new_empty", castPyCFunctionWithKeywords(THPVariable_new_empty), METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_empty_strided", castPyCFunctionWithKeywords(THPVariable_new_empty_strided), METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_full", castPyCFunctionWithKeywords(THPVariable_new_full), METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_zeros", castPyCFunctionWithKeywords(THPVariable_new_zeros), METH_VARARGS | METH_KEYWORDS, NULL},
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