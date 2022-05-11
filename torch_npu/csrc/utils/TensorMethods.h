#include <mutex>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/python_arg_parsing.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_types.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/python_variable.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/utils/LazyInit.h"

namespace torch_npu {
namespace utils {


std::tuple<at::Tensor, c10::optional<at::Device>, c10::optional<at::ScalarType>, bool, bool, c10::optional<at::MemoryFormat>> parse_to_conversion(torch::PythonArgs& r, bool allow_copy);

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
  auto local_device = r.isNone(1) ? c10::Device(at::DeviceType::NPU) : r.device(1);
  auto device = c10::Device(at::DeviceType::NPU, local_device.index()); 
  auto opt_memory_format = r.memoryformatOptional(3);
  TORCH_CHECK(device.is_npu(), "Invalid device, must be npu device");
  torch_npu::utils::npu_lazy_init();
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
  if (device && device->is_cuda()) {
    device = c10::Device(at::DeviceType::NPU, device->index()); 
  }
  auto& scalarType = std::get<2>(parsed);
  auto non_blocking = std::get<3>(parsed);
  auto copy = std::get<4>(parsed);
  auto opt_memory_format = std::get<5>(parsed);
  
  if (device && device->is_npu()) {
    torch_npu::utils::npu_lazy_init();
  }
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
    return THPUtils_packString(torch::utils::options_to_string(self_.options()));
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
  if (device.is_npu()) {
    torch_npu::utils::npu_lazy_init();
  }
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
  return torch::autograd::utils::wrap(self_.is_npu());
  END_HANDLE_TH_ERRORS
}

// autograd methods on torch._C
static PyMethodDef TorchTensorMethods[] = { // NOLINT
  {"npu", castPyCFunctionWithKeywords(THPVariable_npu), METH_VARARGS | METH_KEYWORDS, NULL},
  {"to", castPyCFunctionWithKeywords(THPVariable_to), METH_VARARGS | METH_KEYWORDS, NULL},
  {"type", castPyCFunctionWithKeywords(THPVariable_type), METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_npu", castPyCFunctionWithKeywords(THPVariable_is_npu), METH_VARARGS | METH_KEYWORDS, NULL},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* tensor_functions();

}
}