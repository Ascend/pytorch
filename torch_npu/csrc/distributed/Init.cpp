// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include <deque>

#include <torch/custom_class.h>
#include <torch/csrc/python_headers.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10d/ProcessGroup.hpp>
#include <c10d/comm.hpp>
#include <c10d/Work.hpp>
#include <pybind11/chrono.h>

#include <torch/csrc/Exceptions.h>
#include <ATen/core/functional.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/tensor_flatten.h>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>

#include "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp"
#include "torch_npu/csrc/distributed/Init.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"


namespace {

// Wrapper to ensure GIL is released before destructing ProcessGroupGloo
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_;

public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) =
      default;
  IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  C10_NODISCARD T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

} // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true);


namespace torch_npu {
namespace distributed {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>>;

PyObject* c10d_npu_init(PyObject* _unused, PyObject* noargs) {

  auto torch_npu_C_module = THPObjectPtr(PyImport_ImportModule("torch_npu._C"));
  if (!torch_npu_C_module) {
    throw python_error();
  }
  auto torch_npu_C_m = py::handle(torch_npu_C_module).cast<py::module>();
  
  auto m =
      torch_npu_C_m.def_submodule("_distributed_c10d", "distributed c10d bindings");
  auto module = py::handle(m).cast<py::module>();

  py::module_ dist = py::module_::import("torch._C._distributed_c10d");
  auto processGroupHCCL = intrusive_ptr_no_gil_destructor_class_<::c10d_npu::ProcessGroupHCCL>(
      module, "ProcessGroupHCCL", dist.attr("Backend"))
      .def(py::init<const c10::intrusive_ptr<::c10d::Store>&,
               int,
               int,
               c10::intrusive_ptr<::c10d_npu::ProcessGroupHCCL::Options>>(),
           py::call_guard<py::gil_scoped_release>())
      .def(py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
               int rank,
               int size,
               const std::chrono::milliseconds& timeout) {
               auto options = ::c10d_npu::ProcessGroupHCCL::Options::create();
                   options->is_high_priority_stream = false;
                   options->timeout = timeout;
                   return c10::make_intrusive<::c10d_npu::ProcessGroupHCCL>(
                       store, rank, size, options);
              }),
           py::arg("store"),
           py::arg("rank"),
           py::arg("size"),
           py::arg("timeout") = kProcessGroupDefaultTimeout,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("options", &::c10d_npu::ProcessGroupHCCL::getOptions);

  intrusive_ptr_class_<::c10d_npu::ProcessGroupHCCL::Options>(
      processGroupHCCL,
      "Options",
      dist.attr("ProcessGroup").attr("Options"))
      .def(py::init<>())
      .def_readwrite("op_timeout", &::c10d_npu::ProcessGroupHCCL::Options::opTimeout);

  Py_RETURN_TRUE;
}

// c10d methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"_c10d_npu_init", c10d_npu_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace distributed
} // namespace torch_npu
