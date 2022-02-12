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


namespace torch_npu {
namespace distributed {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

class BroadcastWork {
public:
  inline std::vector<at::Tensor> cast_tensors(at::TensorList tensors) {
    static auto cast_back_to_ori_format = [](const at::Tensor &t) { 
      return NPUNativeFunctions::npu_format_cast(t, t.storage().unsafeGetStorageImpl()->npu_desc_.origin_format_); 
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
    for (size_t i = 0; i < output_tensors.size(); i++) {
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
  const auto buckets =
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
    int split_num = (bucket.size() / 30) + 1;
    int split_size = bucket.size() / split_num;

    for (int i=0; i<split_num; i++) {
      std::vector<at::Tensor> split_bucket;
      split_bucket.reserve((i+1)*split_size > bucket.size() ? bucket.size() - i * split_size : split_size);
      auto start_itr = std::next(bucket.begin(), i*split_size);
      auto end_itr = (i+1)*split_size > bucket.size() ? bucket.end() : std::next(bucket.begin(), (i+1)*split_size);
      std::for_each(start_itr, end_itr, [&tensors, &split_bucket](size_t index) { split_bucket.push_back(tensors[index]); });
      in_flight.emplace_back(process_group, split_bucket, rank);
    }
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

  module.def("_compute_bucket_assignment_by_size",
             &::c10d_npu::compute_bucket_assignment_by_size,
             py::arg("tensors"),
             py::arg("bucket_size"),
             py::arg("expect_sparse_gradient") = std::vector<bool>(),
             py::arg("tensor_indices") = std::vector<int64_t>(),
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
           py::arg("comm_hook_type"))
      .def("_set_construction_logging_data",
           [](
               c10d_npu::Reducer& reducer,
               const std::string& module_name,
               const std::vector<int>& device_ids,
               int output_device,
               bool broadcast_buffers) -> void {
             reducer.set_construction_logging_data(
                 module_name, device_ids, output_device, broadcast_buffers);
           },
           py::arg("reducer"),
           py::arg("module_name"),
           py::arg("device_ids"),
           py::arg("output_device"),
           py::arg("broadcast_buffers"))
      .def("_get_ddp_logging_data",
           [](c10d_npu::Reducer& reducer) -> c10::DDPLoggingData {
             return reducer.get_ddp_logging_data();
           },
           py::arg("reducer"));

  shared_ptr_class_<c10d_npu::Reducer>(module, "Reducer")
      .def(py::init<std::vector<std::vector<torch::autograd::Variable>>,
                    std::vector<std::vector<size_t>>,
                    c10::intrusive_ptr<::c10d::ProcessGroup>,
                    std::vector<std::vector<bool>>,
                    int64_t,
                    bool,
                    bool>(),
           py::arg("replicas"),
           py::arg("bucket_indices"),
           py::arg("process_group"),
           py::arg("expect_sparse_gradients") = std::vector<std::vector<bool>>(),
           py::arg("bucket_bytes_cap") = ::c10d_npu::kDefaultBucketBytesCap,
           py::arg("find_unused_parameters") = false,
           py::arg("gradient_as_bucket_view") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("initialize_buckets",
           &c10d_npu::Reducer::initialize_buckets,
           py::call_guard<py::gil_scoped_release>())
      .def("prepare_for_backward",
           &c10d_npu::Reducer::prepare_for_backward,
           py::call_guard<py::gil_scoped_release>())
      .def("prepare_for_backward",
           [](c10d_npu::Reducer& reducer, const torch::autograd::Variable& output)
               -> void { reducer.prepare_for_backward({output}); },
           py::call_guard<py::gil_scoped_release>())
      .def("get_backward_stats", &c10d_npu::Reducer::get_backward_stats)
      .def("_rebuild_buckets",
           &c10d_npu::Reducer::rebuild_buckets,
           py::call_guard<py::gil_scoped_release>())
      .def("get_bucket_tensors",
           &c10d_npu::Reducer::get_bucket_tensors,
           py::call_guard<py::gil_scoped_release>())
      .def("_push_all_rebuilt_params",
           &c10d_npu::Reducer::push_rebuilt_params_for_all_indices,
           py::call_guard<py::gil_scoped_release>())
      .def("_set_forward_pass_work_handle",
           &c10d_npu::Reducer::set_forward_pass_work_handle,
           py::call_guard<py::gil_scoped_release>())
      .def("_get_local_used_maps",
           &c10d_npu::Reducer::get_local_used_maps_on_device);

  py::module_ dist = py::module_::import("torch.distributed");
  auto processGroupHCCL = intrusive_ptr_class_<::c10d_npu::ProcessGroupHCCL>(
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
                           options->opTimeout = timeout;
                           return c10::make_intrusive<::c10d_npu::ProcessGroupHCCL>(
                               store, rank, size, options);
                        }
                   ),
           py::arg("store"),
           py::arg("rank"),
           py::arg("size"),
           py::arg("timeout") = std::chrono::milliseconds(
               ::c10d_npu::ProcessGroupHCCL::kProcessGroupHCCLOpTimeoutMillis));

  intrusive_ptr_class_<::c10d_npu::ProcessGroupHCCL::Options>(
      processGroupHCCL, "Options")
      .def(py::init<>())
      .def_readwrite("op_timeout", &::c10d_npu::ProcessGroupHCCL::Options::opTimeout);

  module.attr("_DEFAULT_FIRST_BUCKET_BYTES") = ::c10d_npu::kDefaultFirstBucketBytes;
  Py_RETURN_TRUE;
}


// XXX: Ideally the Options of ProcessGroupNCCL should be
// bound using `def_readwrite` like in pybind11, but we
// didn't do that because: 1. no milisecond support yet
// 2. no def_readwrite or property support yet.
static const auto ProcessGroupHCCLOptionsTorchBind =
    torch::class_<::c10d_npu::ProcessGroupHCCL::Options>(
        "dist_c10d",
        "ProcessGroupHCCLOptions")
        .def(torch::init([](int64_t timeout) {
          auto opTimeout = std::chrono::milliseconds(timeout);
          return ::c10d_npu::ProcessGroupHCCL::Options::create(
              opTimeout);
        }));

static const auto ProcessGroupHCCLTorchBind =
    torch::class_<::c10d_npu::ProcessGroupHCCL>("dist_c10d", "ProcessGroupHCCL")
        .def_pickle([](const c10::intrusive_ptr<::c10d_npu::ProcessGroupHCCL>& self) {
                        auto base_process_group = c10::static_intrusive_pointer_cast<::c10d::ProcessGroup>(self);
                        auto name = ::c10d::DistributedC10d::get()->getNameOfProcessGroup(self);
                        return std::vector<std::string>{name};
                    },
                    [](std::vector<std::string> state) {
                        TORCH_CHECK(
                            state.size() == 1,
                            "Expecting exactly 1 state when restoring ProcessGroupHCCL, got: ",
                            state.size());
                        const auto& process_group_name = state.front();
                        auto base_process_group =
                            ::c10d::DistributedC10d::get()->getProcessGroupByName(
                                process_group_name);
                        TORCH_CHECK(
                            base_process_group.defined(),
                            "Needed process group not found, ",
                            "please create a process group with name: ",
                            process_group_name);
                        c10::intrusive_ptr<::c10d_npu::ProcessGroupHCCL> process_group_hccl =
                            c10::dynamic_intrusive_pointer_cast<::c10d_npu::ProcessGroupHCCL>(
                                base_process_group);
                        TORCH_CHECK(
                            process_group_hccl.defined(),
                            "Process group ",
                            process_group_name,
                            " isn't configured for HCCL backend");
                        return process_group_hccl;
                    })
        .def(torch::init(
            [](const c10::intrusive_ptr<::c10d::Store>& store,
                int64_t rank,
                int64_t size,
                c10::intrusive_ptr<::c10d_npu::ProcessGroupHCCL::Options> options,
                const std::string& name) {
                    auto pg = c10::make_intrusive<::c10d_npu::ProcessGroupHCCL>(store, rank, size, options);
                    ::c10d::DistributedC10d::get()->registerProcessGroupName(pg, name);
                    return pg;
                }
            ))
        .def("alltoall_base",
             [](const c10::intrusive_ptr<::c10d_npu::ProcessGroupHCCL>& self,
                at::Tensor output,
                at::Tensor input,
                std::vector<int64_t> outputSplitSizes,
                std::vector<int64_t> inputSplitSizes) {
               return self->alltoall_base(
                   output,
                   input,
                   outputSplitSizes,
                   inputSplitSizes,
                   ::c10d::AllToAllOptions());
            })
        .def("size", [](const c10::intrusive_ptr<::c10d_npu::ProcessGroupHCCL>& self) {
            return (int64_t) self->getSize();
        })
        .def("rank", [](const c10::intrusive_ptr<::c10d_npu::ProcessGroupHCCL>& self) {
            return (int64_t) self->getRank();
        });

// c10d methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"_c10d_init", c10d_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace distributed
} // namespace torch_npu
