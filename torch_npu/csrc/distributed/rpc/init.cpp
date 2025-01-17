#ifdef USE_RPC_FRAMEWORK

#include "torch_npu/csrc/distributed/rpc/init.h"

#include <pybind11/chrono.h>
#include <pybind11/operators.h>
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

#include "torch_npu/csrc/distributed/rpc/tensorpipe_agent.h"

namespace torch_npu {
namespace distributed {
namespace rpc {

constexpr std::chrono::milliseconds kDeleteAllUsersTimeout(100000);

using torch::distributed::rpc::DeviceMap;
using torch::distributed::rpc::kDefaultInitMethod;
using torch::distributed::rpc::kDefaultRpcTimeoutSeconds;
using torch::distributed::rpc::RequestCallbackImpl;
using torch::distributed::rpc::RpcAgent;
using torch::distributed::rpc::RpcBackendOptions;
using torch::distributed::rpc::TensorPipeRpcBackendOptions;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject *rpc_npu_init(PyObject *_unused, PyObject *noargs)
{
    auto torch_npu_C_module = THPObjectPtr(PyImport_ImportModule("torch_npu._C"));
    if (!torch_npu_C_module) {
        throw python_error();
    }
    auto torch_npu_C_m = py::handle(torch_npu_C_module).cast<py::module>();
    auto m = torch_npu_C_m.def_submodule("_distributed_rpc", "distributed rpc bindings");
    auto module = py::handle(m).cast<py::module>();
    // Import the rpc_module so we can subclass TensorPipeAgent
    py::module rpc_module = py::module::import("torch.distributed.rpc");

    shared_ptr_class_<TensorPipeAgent>(module, "TensorPipeAgent", rpc_module.attr("TensorPipeAgent"))
        .def(py::init([](const c10::intrusive_ptr<::c10d::Store> &store, std::string selfName, worker_id_t selfId,
                         c10::optional<int> worldSize, TensorPipeRpcBackendOptions opts,
                         std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,
                         std::vector<c10::Device> devices) {
                 return std::shared_ptr<TensorPipeAgent>(
                     new TensorPipeAgent(store, std::move(selfName), selfId, worldSize, std::move(opts),
                                         std::move(reverseDeviceMaps), std::move(devices),
                                         std::make_unique<RequestCallbackImpl>()),
                     torch::impl::destroy_without_gil<TensorPipeAgent>);
             }),
             py::arg("store"), py::arg("name"), py::arg("rank"), py::arg("world_size"), py::arg("rpc_backend_options"),
             py::arg("reverse_device_maps"), py::arg("devices"))
        .def("join", &TensorPipeAgent::join, py::call_guard<py::gil_scoped_release>(), py::arg("shutdown") = false,
             py::arg("timeout") = 0)
        .def("shutdown", &TensorPipeAgent::shutdown, py::call_guard<py::gil_scoped_release>())
        .def("get_worker_info", (const WorkerInfo &(TensorPipeAgent::*)(void) const) &RpcAgent::getWorkerInfo,
             py::call_guard<py::gil_scoped_release>())
        .def("get_worker_info",
             (const WorkerInfo &(TensorPipeAgent::*)(const std::string &) const) &TensorPipeAgent::getWorkerInfo,
             py::call_guard<py::gil_scoped_release>())
        .def("get_worker_info",
             (const WorkerInfo &(TensorPipeAgent::*)(worker_id_t id) const) &TensorPipeAgent::getWorkerInfo,
             py::call_guard<py::gil_scoped_release>())
        .def("get_worker_infos",
             (std::vector<WorkerInfo>(TensorPipeAgent::*)() const) &TensorPipeAgent::getWorkerInfos,
             py::call_guard<py::gil_scoped_release>())
        .def("_get_device_map",
             (DeviceMap(TensorPipeAgent::*)(const WorkerInfo &dst) const) &TensorPipeAgent::getDeviceMap,
             py::call_guard<py::gil_scoped_release>())
        .def("_get_backend_options", &TensorPipeAgent::getBackendOptions, py::call_guard<py::gil_scoped_release>())
        .def("_update_group_membership", &TensorPipeAgent::updateGroupMembership,
             py::call_guard<py::gil_scoped_release>())
        .def_readonly("is_static_group", &TensorPipeAgent::isStaticGroup_)
        .def_property_readonly("store", &TensorPipeAgent::getStore);

    Py_RETURN_TRUE;
}

} // namespace rpc
} // namespace distributed
} // namespace torch_npu

#endif
