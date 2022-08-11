#include <torch/csrc/Generator.h>

#include <structmember.h>
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>

// #include <TH/TH.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/tensor_types.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/npu/Generator.h"
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

using namespace at;
using namespace torch;

PyObject *THPGeneratorClass = nullptr;

PyObject * initDefaultGenerator(at::Generator cdata)
{
  auto type = (PyTypeObject*)THPGeneratorClass;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPGenerator*>(self.get());
  self_->cdata = cdata;
  return self.release();
}


static void THPGenerator_dealloc(PyObject* _self)
{
  auto self = reinterpret_cast<THPGenerator*>(_self);
  if (self->cdata.defined()) {
    self->cdata.set_pyobj(nullptr);
    self->cdata.~Generator();
  }
  Py_TYPE(_self)->tp_free(_self);
}

static PyObject * THPGenerator_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "Generator(Device device=None)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.deviceWithDefault(0, at::Device(at::kCPU));

  THPGeneratorPtr self((THPGenerator *)type->tp_alloc(type, 0));

  if (device.type() == at::kCPU) {
    self->cdata = at::make_generator<CPUGeneratorImpl>();
  } else if (device.type() == at_npu::key::NativeDeviceType){
    self->cdata = at::make_generator<at_npu::NPUGeneratorImpl>(device.index());
  } else {
    AT_ERROR("Device type ", c10::DeviceTypeName(device.type()),
             " is not supported for torch.Generator() api.");
  }
  return (PyObject*)self.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_getState(PyObject *_self, PyObject *noargs)
{
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  auto& gen = ((THPGenerator*)_self)->cdata;

  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen.mutex());
  auto state_tensor = gen.get_state();

  return THPVariable_Wrap(std::move(state_tensor));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_setState(PyObject *_self, PyObject *_new_state)
{
  using namespace torch::autograd;
  
  HANDLE_TH_ERRORS
  if (!THPVariable_Check(_new_state)) {
    throw torch::TypeError("expected a torch.ByteTensor, but got %s", Py_TYPE(_new_state)->tp_name);
  }
  auto self = (THPGenerator*)_self;
  auto& gen = self->cdata;
  auto& new_state_tensor = THPVariable_Unpack(_new_state);
  
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen.mutex());
  gen.set_state(new_state_tensor);
  
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_manualSeed(PyObject *_self, PyObject *seed)
{
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;
  auto generator = self->cdata;
  THPUtils_assert(THPUtils_checkLong(seed), "manual_seed expected a long, "
          "but got %s", THPUtils_typename(seed));
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(generator.mutex());
  uint64_t seed_unpacked;
  try {
    // First try to interpret as unsigned long
    seed_unpacked = THPUtils_unpackUInt64(seed);
  } catch(...) {
    if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
      // If an overflow happened, then the seed could be negative,
      // so try to interpret it as signed long
      PyErr_Clear();
      int64_t seed_unpacked_signed = THPUtils_unpackLong(seed);
      seed_unpacked = *(reinterpret_cast<uint64_t*>(&seed_unpacked_signed));
    } else {
      // If any other type of exception happened, rethrow it
      throw;
    }
  }
  generator.set_current_seed(seed_unpacked);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_seed(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  // See Note [Acquire lock when using random generators]
  auto self = (THPGenerator*)_self;
  std::lock_guard<std::mutex> lock(self->cdata.mutex());
  uint64_t seed_val = self->cdata.seed();
  return THPUtils_packUInt64(seed_val);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_initialSeed(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;
  return THPUtils_packUInt64(self->cdata.current_seed());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_get_device(THPGenerator *self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->cdata.device());
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THPGenerator_properties[] = {
  {"device", (getter)THPGenerator_get_device, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THPGenerator_methods[] = {
  {"get_state",       THPGenerator_getState,       METH_NOARGS,  nullptr},
  {"set_state",       THPGenerator_setState,       METH_O,       nullptr},
  {"manual_seed",     THPGenerator_manualSeed,     METH_O,       nullptr},
  {"seed",            THPGenerator_seed,           METH_NOARGS,  nullptr},
  {"initial_seed",    THPGenerator_initialSeed,    METH_NOARGS,  nullptr},
  {nullptr}
};

static struct PyMemberDef THPGenerator_members[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPGenerator, cdata), READONLY, nullptr},
  {nullptr}
};

PyTypeObject THPGeneratorType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch_npu._C.Generator",                   /* tp_name */
  sizeof(THPGenerator),                        /* tp_basicsize */
  0,                                           /* tp_itemsize */
  THPGenerator_dealloc,                        /* tp_dealloc */
  0,                                           /* tp_vectorcall_offset */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  nullptr,                                     /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  nullptr,                                     /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    /* tp_flags */
  nullptr,                                     /* tp_doc */
  nullptr,                                     /* tp_traverse */
  nullptr,                                     /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                           /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  THPGenerator_methods,                        /* tp_methods */
  THPGenerator_members,                        /* tp_members */
  THPGenerator_properties,                     /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  THPGenerator_pynew,                          /* tp_new */
};

bool THPGenerator_init(PyObject *module)
{
  THPGeneratorClass = (PyObject*)&THPGeneratorType;
  if (PyType_Ready(&THPGeneratorType) < 0)
    return false;
  Py_INCREF(&THPGeneratorType);
  PyModule_AddObject(module, "Generator", (PyObject *)&THPGeneratorType);
  return true;
}
