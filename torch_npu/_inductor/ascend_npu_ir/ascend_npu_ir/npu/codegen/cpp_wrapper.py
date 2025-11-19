def cpp_launcher(signature, kernel_name, ranks, dynamic=False) -> str:
    def _ty_to_cpp(ty):
        if ty[0] == '*':
            return "void*"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "f16": "float",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    def _extracted_ty(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'f16': 'float',
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    def _format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]
    if dynamic:
        arg_decls = ', '.join(
            f"{_ty_to_cpp(ty)} arg{i}"
            + ("" if "torch." in ty else f", {_ty_to_cpp(ty)} arg_allocate{i}, {_ty_to_cpp(ty)} offset{i}" + (', ' if ranks[i] > 0 else '')
            + ', '.join(f"{_ty_to_cpp(ty)} sizes{i}_{rank}" for rank in range(ranks[i])) + (', ' if ranks[i] > 0 else '')
            + ', '.join(f"{_ty_to_cpp(ty)} strides{i}_{rank}" for rank in range(ranks[i])))
            for i, ty in signature.items()
        )
        py_args_format = "iKkkLOOOOO" + ''.join([_format_of(_extracted_ty(ty)) + ('' if "torch." in ty else _format_of(_extracted_ty(ty)) + 'L' + 'L' * ranks[i] * 2) for i, ty in signature.items()])
        return f"""
#include <cpp_common.h>
#include <stdbool.h>
#include <string>
#include <dlfcn.h>
#include <iostream>

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either int64 or have data_ptr method");
  return ptr_info;
}}

static void _launch(void* func, void* tiling_func, int64_t tiling_size, void* arg_tiling_host, void* arg_tiling_device, rtStream_t stream, int gridX, {arg_decls}) {{
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  
  if (tiling_size == 0) {{
    auto launch_call = [func, tiling_func, tiling_size, arg_tiling_host, arg_tiling_device, gridX, stream, {', '.join(f"arg{i}" + ("" if "torch." in ty else f", arg_allocate{i}, offset{i}" +(', ' if ranks[i] > 0 else '') + ', '.join(f"sizes{i}_{rank}" for rank in range(ranks[i])) + (', ' if ranks[i] > 0 else '') + ', '.join(f"strides{i}_{rank}" for rank in range(ranks[i]))) for i, ty in signature.items())}]() {{
      struct __attribute__((packed)) {{
      
      {' '.join(f'{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8}))); ' + ('' if "torch." in ty else f'{_ty_to_cpp(ty)} arg_allocate{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8}))); {_ty_to_cpp(ty)} offset{i} __attribute__((aligned(8))); ' + ' '.join(f'{_ty_to_cpp(ty)} sizes{i}_{rank} __attribute__((aligned(8)));' for rank in range(ranks[i])) + ' ' + ' '.join(f'{_ty_to_cpp(ty)} strides{i}_{rank} __attribute__((aligned(8)));' for rank in range(ranks[i]))) for i, ty in signature.items())}

      }} args = {{
      {', '.join(f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" + ("" if "torch." in ty else f", static_cast<{_ty_to_cpp(ty)}>(arg_allocate{i}), static_cast<{_ty_to_cpp(ty)}>(offset{i})"+ (', ' if ranks[i] > 0 else '') + ', '.join(f"static_cast<{_ty_to_cpp(ty)}>(sizes{i}_{rank})" for rank in range(ranks[i])) + (', ' if ranks[i] > 0 else '') + ', '.join(f"static_cast<{_ty_to_cpp(ty)}>(strides{i}_{rank})" for rank in range(ranks[i]))) for i, ty in signature.items())}

      }};
      
      rtError_t ret = common_launch_dyn(const_cast<char*>("{kernel_name}"), func, tiling_func, tiling_size, arg_tiling_host, arg_tiling_device, gridX, static_cast<void *>(&args), sizeof(args), stream);
      return ret;
    }};
    opcommand_call("{kernel_name}", launch_call);
  }} else {{
    int64_t __attribute__((aligned(8))) key_tiling;
    // void* arg_tiling_host = nullptr;
    void* offset_tiling = 0;
    void* sizes_tiling = (void*)(tiling_size / sizeof(int64_t));
    void* strides_tiling = (void*)1;
    auto launch_call = [func, tiling_func, tiling_size, arg_tiling_host, arg_tiling_device, gridX, stream, {', '.join(f"arg{i}" + ("" if "torch." in ty else f", arg_allocate{i}, offset{i}" + (', ' if ranks[i] > 0 else '') + ', '.join(f"sizes{i}_{rank}" for rank in range(ranks[i])) + (', ' if ranks[i] > 0 else '') + ', '.join(f"strides{i}_{rank}" for rank in range(ranks[i]))) for i, ty in signature.items())}, key_tiling, offset_tiling, sizes_tiling, strides_tiling]() {{
      struct __attribute__((packed)) {{
      
      {' '.join(f'{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8}))); ' + ('' if "torch." in ty else f'{_ty_to_cpp(ty)} arg_allocate{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8}))); {_ty_to_cpp(ty)} offset{i} __attribute__((aligned(8))); ' + ' '.join(f'{_ty_to_cpp(ty)} sizes{i}_{rank} __attribute__((aligned(8)));' for rank in range(ranks[i])) + ' ' + ' '.join(f'{_ty_to_cpp(ty)} strides{i}_{rank} __attribute__((aligned(8)));' for rank in range(ranks[i]))) for i, ty in signature.items())}

      void* key_tiling __attribute__((aligned(8)));
      void* arg_tiling_host __attribute__((aligned(8)));
      void* arg_tiling_device __attribute__((aligned(8)));
      void* offset_tiling __attribute__((aligned(8)));
      void* sizes_tiling __attribute__((aligned(8)));
      void* strides_tiling __attribute__((aligned(8)));

      }} args = {{
      {', '.join(f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" + ("" if "torch." in ty else f", static_cast<{_ty_to_cpp(ty)}>(arg_allocate{i}), static_cast<{_ty_to_cpp(ty)}>(offset{i})" + (', ' if ranks[i] > 0 else '') + ', '.join(f"static_cast<{_ty_to_cpp(ty)}>(sizes{i}_{rank})" for rank in range(ranks[i])) + (', ' if ranks[i] > 0 else '') + ', '.join(f"static_cast<{_ty_to_cpp(ty)}>(strides{i}_{rank})" for rank in range(ranks[i]))) for i, ty in signature.items())+ ', '}

      (void*)(&key_tiling), arg_tiling_host, arg_tiling_device, static_cast<void*>(offset_tiling), static_cast<void*>(sizes_tiling), static_cast<void*>(strides_tiling)
      }};
      
      rtError_t ret = common_launch_dyn(const_cast<char*>("{kernel_name}"), func, tiling_func, tiling_size, arg_tiling_host, arg_tiling_device, gridX, static_cast<void *>(&args), sizeof(args), stream);
      return ret;
    }};
    opcommand_call("{kernel_name}", launch_call);
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX;
  rtStream_t stream;
  PyObject *func;
  PyObject *tiling_func;
  int64_t tiling_size;
  PyObject *arg_tiling_host;
  PyObject *arg_tiling_device;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *metadata = NULL;
  {'; '.join(f"{_extracted_ty(ty)} _arg{i}" + ("" if "torch." in ty else f"; {_extracted_ty(ty)} _arg_allocate{i}; int64_t offset{i}" + ('; ' if ranks[i] > 0 else '') + ''.join(f"int64_t sizes{i}_{rank}" + ('; ' if ranks[i] > 0 else '') for rank in range(ranks[i])) + '; '.join(f"int64_t strides{i}_{rank}" for rank in range(ranks[i]))) for i, ty in signature.items()) + '; '}
  if(!PyArg_ParseTuple(
      args, \"{py_args_format}\",
      &gridX, &stream, &func, &tiling_func, &tiling_size, &arg_tiling_host, &arg_tiling_device,
      &launch_enter_hook, &launch_exit_hook, &metadata
      {', ' + ', '.join((f"&_arg{i}" + ("" if "torch." in ty else f", &_arg_allocate{i}, &offset{i}" + (', ' if ranks[i] > 0 else '') + ', '.join(f"&sizes{i}_{rank}" for rank in range(ranks[i])) + (', ' if ranks[i] > 0 else '') + ', '.join(f"&strides{i}_{rank}" for rank in range(ranks[i])))) for i, ty in signature.items()) if len(signature) > 0 else ''}
      )
    ) {{
    return NULL;
  }}


  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL" if ty[0] == "*" else "" for i, ty in signature.items()]) + "; "}
  {"; ".join([f"DevicePtrInfo ptr_allocate_info{i} = getPointer(_arg_allocate{i}, {i}); if (!ptr_allocate_info{i}.valid) return NULL" if (ty[0] == "*" and "torch." not in ty) else "" for i, ty in signature.items()]) + "; "}

  DevicePtrInfo arg_tiling_host_ptr = getPointer(arg_tiling_host, 0);
  DevicePtrInfo arg_tiling_device_ptr = getPointer(arg_tiling_device, 0);

  _launch(reinterpret_cast<void *>(func), reinterpret_cast<void *>(tiling_func), tiling_size, arg_tiling_host_ptr.dev_ptr, arg_tiling_device_ptr.dev_ptr, stream, gridX, {', '.join([f"ptr_info{i}.dev_ptr" + ('' if "torch." in ty else ', ' + f"ptr_allocate_info{i}.dev_ptr, reinterpret_cast<void *>(offset{i})" + (', ' if ranks[i] > 0 else '') + ', '.join(f"reinterpret_cast<void *>(sizes{i}_{rank})" for rank in range(ranks[i])) + (', ' if ranks[i] > 0 else '') + ', '.join(f"reinterpret_cast<void *>(strides{i}_{rank})" for rank in range(ranks[i]))) for i, ty in signature.items()])});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyObject* get_host_func_and_tiling_size(PyObject* self, PyObject* args) {{
  const char *func_name;
  const char *tiling_func_name;
  const char *get_tiling_struct_size_func_name;
  const char *so_file;
  if(!PyArg_ParseTuple(
    args, "ssss", &func_name, &tiling_func_name, &get_tiling_struct_size_func_name, &so_file
    )
  ) {{
    return NULL;
  }}
  void *handle = dlopen(so_file, RTLD_LAZY);
  if (handle == NULL) {{
      std::cout<<"handle == NULL!"<<std::endl;
      return Py_None;
  }}

  typedef void (*mlir_func)(uint32_t, void*, void*, void*);
  mlir_func func = (mlir_func)dlsym(handle, func_name);
  if (func == NULL) {{
      std::cout<<"Failed to load symbol for func: "<<dlerror()<<std::endl;
      dlclose(handle);
      return Py_None;
  }}

  typedef int (*mlir_get_size_func)();
  mlir_get_size_func get_size_func = (mlir_get_size_func)dlsym(handle, get_tiling_struct_size_func_name);
  if (get_size_func == NULL) {{
      std::cout<<"Failed to load symbol for get_size_func: "<<dlerror()<<std::endl;
      dlclose(handle);
      return Py_None;
  }}

  int64_t tilingSize = get_size_func();
  tilingSize *= sizeof(int64_t);

  typedef int64_t (*mlir_tiling_func)(void*);
  mlir_tiling_func tiling_func = NULL;
  if (tilingSize != 0) {{
    tiling_func = (mlir_tiling_func)dlsym(handle, tiling_func_name);
    if (tiling_func == NULL) {{
      std::cout<<"Failed to load symbol for tiling_func: "<<dlerror()<<std::endl;
      dlclose(handle);
      return Py_None;
    }}
  }}

  return PyTuple_Pack(3, PyLong_FromUnsignedLong(reinterpret_cast<uintptr_t>(func)), PyLong_FromUnsignedLong(reinterpret_cast<uintptr_t>(tiling_func)), PyLong_FromLongLong(tilingSize));
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{"get_host_func_and_tiling_size", get_host_func_and_tiling_size, METH_VARARGS, "Get host func from kernel.so"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  "__launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    py_args_format = "iKKOOO" + ''.join([_format_of(_extracted_ty(ty)) for ty in signature.values()])
    return f"""
#include <cpp_common.h>
#include <stdbool.h>
#include <string>

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static void _launch(const void* func, rtStream_t stream, int gridX, {arg_decls}) {{
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  auto launch_call = [func, gridX, stream, {', '.join(f" arg{i}" for i, ty in signature.items())}]() {{
    struct __attribute__((packed)) {{
      {' '.join(f'{_ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items())}
    }} args = {{
      {', '.join(f'static_cast<{_ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items())}
    }};

    rtError_t ret = common_launch(const_cast<char*>("{kernel_name}"), func, gridX, static_cast<void *>(&args), sizeof(args), stream);
    return ret;
  }};
  opcommand_call("{kernel_name}", launch_call);
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX;
  rtStream_t stream;
  const void *function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *metadata = NULL;
  {' '.join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(
      args, \"{py_args_format}\",
      &gridX, &stream, &function,
      &launch_enter_hook, &launch_exit_hook, &metadata
      {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''}
      )
    ) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};

  _launch(function, stream, gridX, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  "__launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
