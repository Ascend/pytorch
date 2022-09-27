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

#include <torch/csrc/python_headers.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10d/ProcessGroup.hpp>
#include <c10d/PyProcessGroup.hpp>
#include <pybind11/chrono.h>

#include <torch/csrc/Exceptions.h>
#include <ATen/core/functional.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/tensor_flatten.h>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>

#include <torch/custom_class.h>

#include "torch_npu/csrc/distributed/ProcessGroupHCCL.hpp"
#include "torch_npu/csrc/distributed/Init.h"
#include "torch_npu/csrc/distributed/reducer.hpp"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "patch/include/c10d/comm.hpp"



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

class BroadcastWork {
public:
  inline std::vector<at::Tensor> cast_tensors(at::TensorList tensors) {
    static auto cast_back_to_ori_format = [](const at::Tensor &t) {
      return at_npu::native::NPUNativeFunctions::npu_format_cast(t, torch_npu::NPUBridge::GetNpuStorageImpl(t)->npu_desc_.origin_format_);
    };
    return c10::fmap(tensors, cast_back_to_ori_format);
  }

  BroadcastWork(
      const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
      std::vector<at::Tensor> bucket_tensors,
      int root_rank = 0)
      : bucket_tensors_(std::move(bucket_tensors)),
        cast_tensors_(cast_tensors(bucket_tensors_)),
        flat_tensor_({torch::utils::flatten_dense_tensors(cast_tensors_)}) {
    c10d::BroadcastOptions broadcastOptions;
    broadcastOptions.rootRank = root_rank;
    work_ = process_group->broadcast(flat_tensor_, broadcastOptions);
  }

  void finish() {
    work_->wait();
    auto output_tensors = torch::utils::unflatten_dense_tensors(
        flat_tensor_.front(), cast_tensors_);
    TORCH_INTERNAL_ASSERT(output_tensors.size() == bucket_tensors_.size());
    for(const auto i : c10::irange(output_tensors.size())) {
      bucket_tensors_[i].copy_(output_tensors[i], true);
    }
  }

protected:
  // The list of tensors to broadcast. They are guaranteed to be
  // placed on the same device and have the same dtype.
  std::vector<at::Tensor> bucket_tensors_;
  // Some tensors with format, such as FRACTAL_Z, 5HD, may be padded to
  // keep alignment with 16*16 cube kernel which will modify storage as
  // input tensor for cat operation during flatten to a buffer tensor.
  // So, it needs to cast all bucket tensors to tensors with format HCHW
  std::vector<at::Tensor> cast_tensors_;
  // The vector with a single flattened tensor containing the contents
  // of the tensors in bucket_tensors_. It must be stored in a vector
  // because c10d::ProcessGroup::broadcast takes a vector argument.
  std::vector<at::Tensor> flat_tensor_;

private:
  // The broadcast work that is kicked off upon construction.
  c10::intrusive_ptr<c10d::ProcessGroup::Work> work_;
};

// Broadcast many tensors to all processes in the process group.
void broadcast_coalesced(
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank) {
  // Coalesce tensors into buckets taking into account the maximum buffer size.
  // This routine is multi-device aware, so the tensors can be split across
  // multiple devices and can contain a mix of CPU and CUDA tensors.
  std::vector<std::vector<size_t>> buckets;
  std::tie(buckets, std::ignore) =
      c10d_npu::compute_bucket_assignment_by_size(tensors.vec(), {buffer_size});

  // Returns tensor at specified index in input tensor list.
  const auto lookup = [&tensors](size_t index) { return tensors[index]; };

  // We maintain a maximum of 2 in flight broadcast operations to avoid
  // allocating too much memory (in case the specified tensors are very large).
  std::deque<BroadcastWork> in_flight;
  constexpr auto max_in_flight = 2;
  for (const auto& bucket : buckets) {
    if (in_flight.size() >= max_in_flight) {
      in_flight.front().finish();
      in_flight.pop_front();
    }

    in_flight.emplace_back(process_group, c10::fmap(bucket, lookup), rank);
  }

  while (!in_flight.empty()) {
    in_flight.front().finish();
    in_flight.pop_front();
  }
}

// Called from DDP's Python API to create a c10d Python comm hook object.
// The input state and callable comm_hook are Python objects. It later calls
// register_comm_hook function of the reducer input to register the hook.
void _register_comm_hook(
    c10d_npu::Reducer& reducer,
    py::object state,
    py::object comm_hook) {
  reducer.register_comm_hook(std::make_unique<::c10d::PythonCommHook>(
      std::move(state), std::move(comm_hook)));
}

// Called from DDP's Python API to create a c10d C++ comm hook.
// The input is an enum hook type. It later calls register_builtin_comm_hook
// function of the reducer input to set the hook type.
void _register_builtin_comm_hook(
    c10d_npu::Reducer& reducer,
    ::c10d::BuiltinCommHookType comm_hook_type) {
  reducer.register_builtin_comm_hook(comm_hook_type);
}

PyObject* c10d_init(PyObject* _unused, PyObject* noargs) {
  C10_LOG_API_USAGE_ONCE("npu_c10d.python.import");

  auto c10d_module = THPObjectPtr(PyImport_ImportModule("torch_npu.distributed"));
  if (!c10d_module) {
    throw python_error();
  }

  auto torchnpu_C_module = THPObjectPtr(PyImport_ImportModule("torch_npu._C"));
  if (!torchnpu_C_module) {
    throw python_error();
  }

  auto torchnpu_C_m = py::handle(torchnpu_C_module).cast<py::module>();
  auto m =
      torchnpu_C_m.def_submodule("_distributed_c10d", "distributed c10d bindings");

  auto module = py::handle(m).cast<py::module>();

  module.def(
      "_compute_bucket_assignment_by_size",
      [](const std::vector<at::Tensor>& tensors,
            const std::vector<size_t>& bucket_size_limits,
            const std::vector<bool>& expect_sparse_gradient,
            const std::vector<int64_t>& tensor_indices,
            const c10::optional<std::shared_ptr<::c10d::Logger>>& logger) {
             if (logger.has_value()) {
                std::weak_ptr<::c10d::Logger> logger_weakref = logger.value();
                return ::c10d_npu::compute_bucket_assignment_by_size(tensors, bucket_size_limits, expect_sparse_gradient, tensor_indices, {logger_weakref});
             } else {
                return ::c10d_npu::compute_bucket_assignment_by_size(tensors, bucket_size_limits, expect_sparse_gradient, tensor_indices, {});
             }
      },
      py::arg("tensors"),
      py::arg("bucket_size"),
      py::arg("expect_sparse_gradient") = std::vector<bool>(),
      py::arg("tensor_indices") = std::vector<int64_t>(),
      py::arg("logger") = c10::optional<std::shared_ptr<::c10d::Logger>>{},
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_verify_params_across_processes",
      [](const c10::intrusive_ptr<::c10d::ProcessGroup>& process_group,
         const std::vector<at::Tensor>& params,
         const c10::optional<std::shared_ptr<::c10d::Logger>>& logger) {
             if (logger.has_value()) {
                std::weak_ptr<::c10d::Logger> logger_weakref = logger.value();
                c10d_npu::verify_params_across_processes(process_group, params, {logger_weakref});
             } else {
                c10d_npu::verify_params_across_processes(process_group, params, {});
             }
      },
      py::arg("process_group"),
      py::arg("params"),
      py::arg("logger") = c10::optional<std::shared_ptr<::c10d::Logger>>{},
      py::call_guard<py::gil_scoped_release>());

  module.def("_broadcast_coalesced",
             // Define a lambda such that the pybind11 prototype can take a std::vector
             // for the tensor list argument, but still pass it to the underlying
             // function as a c10::ArrayRef.
             [](c10::intrusive_ptr<::c10d::ProcessGroup> process_group,
                 std::vector<at::Tensor> tensors, // NOLINT
                 size_t buffer_size,
                 int rank) {
                 torch_npu::distributed::broadcast_coalesced(
                     std::move(process_group), tensors, buffer_size, rank);
             },
             py::arg("process_group"),
             py::arg("tensors"),
             py::arg("buffer_size"),
             // The source of truth rank to broadcast the tensors from.
             py::arg("src") = 0,
             py::call_guard<py::gil_scoped_release>());

  module
      .def("_register_comm_hook",
           &_register_comm_hook,
           py::arg("reducer"),
           py::arg("state"),
           py::arg("comm_hook"),
           py::call_guard<py::gil_scoped_release>())
      .def("_register_builtin_comm_hook",
           &_register_builtin_comm_hook,
           py::arg("reducer"),
           py::arg("comm_hook_type"));


  shared_ptr_class_<c10d_npu::Reducer>(module, "Reducer")
      .def(py::init<
               std::vector<at::Tensor>,
               std::vector<std::vector<size_t>>,
               std::vector<size_t>,
               c10::intrusive_ptr<::c10d::ProcessGroup>,
               std::vector<bool>,
               int64_t,
               bool,
               bool,
               std::unordered_map<size_t, std::string>,
               int64_t>(),
           py::arg("params"),
           py::arg("bucket_indices"),
           py::arg("per_bucket_size_limits"),
           py::arg("process_group"),
           py::arg("expect_sparse_gradients") = std::vector<bool>(),
           py::arg("bucket_bytes_cap") = ::c10d::kDefaultBucketBytesCap,
           py::arg("find_unused_parameters") = false,
           py::arg("gradient_as_bucket_view") = false,
           py::arg("param_to_name_mapping") =
              std::unordered_map<size_t, std::string>(),
           py::arg("first_bucket_bytes_cap") = ::c10d::kDefaultFirstBucketBytes,
           py::call_guard<py::gil_scoped_release>())
      .def("prepare_for_forward",
           &c10d_npu::Reducer::prepare_for_forward,
           py::call_guard<py::gil_scoped_release>())
      .def("prepare_for_backward",
           &c10d_npu::Reducer::prepare_for_backward,
           py::call_guard<py::gil_scoped_release>())
      .def("prepare_for_backward",
           [](c10d_npu::Reducer& reducer, const at::Tensor& output)
               -> void { reducer.prepare_for_backward({output}); },
           py::call_guard<py::gil_scoped_release>())
      .def("get_backward_stats", &c10d_npu::Reducer::get_backward_stats)
      .def("_install_post_backward_futures", [](::c10d_npu::Reducer& reducer, const std::vector<std::shared_ptr<torch::jit::PythonFutureWrapper>>& futs) {
              c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures(c10::FutureType::create(c10::TensorType::get()));
              for (const auto & fut : futs) {
              futures.push_back(fut->fut);
              }
              reducer.install_futures(std::move(futures));
          },
           py::call_guard<py::gil_scoped_release>())
      .def("_rebuild_buckets",
           &::c10d_npu::Reducer::rebuild_buckets,
           py::call_guard<py::gil_scoped_release>())
      .def("_get_zeros_like_grad_buckets",
           [](::c10d_npu::Reducer& reducer) {
             return reducer.get_grad_buckets(true);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("_push_all_rebuilt_params",
           &::c10d_npu::Reducer::push_rebuilt_params_for_all_indices,
           py::call_guard<py::gil_scoped_release>())
      .def("_set_forward_pass_work_handle",
           &::c10d_npu::Reducer::set_forward_pass_work_handle,
           py::call_guard<py::gil_scoped_release>())
      .def("_get_local_used_map",
           &::c10d_npu::Reducer::get_local_used_map_on_device)
      .def("_set_ddp_runtime_logging_sample_rate",
           &::c10d_npu::Reducer::set_ddp_runtime_logging_sample_rate,
           py::arg("sample_rate"),
           py::call_guard<py::gil_scoped_release>())
      .def("_set_static_graph",
           &::c10d_npu::Reducer::set_static_graph,
           py::call_guard<py::gil_scoped_release>())
      .def("_ddp_graph_static",
           &::c10d_npu::Reducer::ddp_graph_static,
           py::call_guard<py::gil_scoped_release>())
      .def("_delay_all_reduce",
           &::c10d_npu::Reducer::delay_all_reduce,
           py::call_guard<py::gil_scoped_release>())
      .def("_run_comm_hook",
           [](::c10d_npu::Reducer& reducer, ::c10d::GradBucket& bucket)
               -> std::shared_ptr<torch::jit::PythonFutureWrapper> {
             c10::intrusive_ptr<c10::ivalue::Future> fut =
                 reducer.run_comm_hook(bucket);
             return std::make_shared<torch::jit::PythonFutureWrapper>(fut);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("set_logger",
           [](::c10d_npu::Reducer& reducer,
              const std::shared_ptr<::c10d::Logger> logger) {
             std::weak_ptr<::c10d::Logger> logger_weakref = logger;
             reducer.set_logger(logger_weakref);
           });


  py::module_ dist = py::module_::import("torch.distributed");
  auto processGroupHCCL = intrusive_ptr_no_gil_destructor_class_<::c10d_npu::ProcessGroupHCCL>(
      module, "ProcessGroupHCCL", dist.attr("ProcessGroup"))
      .def(py::init<c10::intrusive_ptr<::c10d::Store>&,
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
                           options->opTimeout = timeout;
                           return c10::make_intrusive<::c10d_npu::ProcessGroupHCCL>(
                               store, rank, size, options);
                        }
                   ),
           py::arg("store"),
           py::arg("rank"),
           py::arg("size"),
           py::arg("timeout") = kProcessGroupDefaultTimeout,
           py::call_guard<py::gil_scoped_release>())
          .def_property_readonly("options", &::c10d_npu::ProcessGroupHCCL::getOptions);

  intrusive_ptr_class_<::c10d_npu::ProcessGroupHCCL::Options>(
      processGroupHCCL, "Options")
      .def(py::init<>())
      .def_readwrite("op_timeout", &::c10d_npu::ProcessGroupHCCL::Options::opTimeout);

  module.attr("_DEFAULT_FIRST_BUCKET_BYTES") = ::c10d_npu::kDefaultFirstBucketBytes;
  Py_RETURN_TRUE;
}

// c10d methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"_c10d_init", c10d_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace distributed
} // namespace torch_npu
