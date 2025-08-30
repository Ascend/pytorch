#include <thread>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/npu/Stream.h"
#include "torch_npu/csrc/npu/Graph.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;
static std::map<c10_npu::NPUStream, std::vector<PyFuncStruct *>> callbacks = {};
constexpr int processReportTimeout = 100;
static ThreadArgs* threadArgs = nullptr;
static uint64_t threadId = -1;

void *process_callback(void *arg)
{
    ThreadArgs* args = static_cast<ThreadArgs *>(arg);
    auto ret = aclrtSetCurrentContext(args->context);
    while (!args->exitFlag) {
        (void)aclrtProcessReport(processReportTimeout);
    }
    delete args;
    args = nullptr;
    return nullptr;
}

void LaunchCallFunc(void *userData)
{
    PyGILState_STATE state = PyGILState_Ensure();
    if (userData == nullptr) {
        return;
    }
    auto data = (PyFuncStruct *)(userData);
    PyObject *argslist = Py_BuildValue("(O)", data->pyFuncArgs);
    if (argslist == nullptr) {
        return;
    }
    PyObject *result = PyObject_CallObject(data->pyFunc, argslist);
    if (result == nullptr) {
        return;
    }
    if (argslist != nullptr) {
        Py_XDECREF(argslist);
    }
    if (result != nullptr) {
        Py_XDECREF(result);
    }
    PyGILState_Release(state);
}

void TORCH_NPU_API THNPGraph_init(PyObject* module) {
    // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
    // but CI linter and some builds prefer "module".
    auto torch_N_m = py::handle(module).cast<py::module>();

    py::class_<c10_npu::NPUTaskGroupHandle>(torch_N_m, "_NPUTaskGroupHandle")
            .def_readonly("task_group", &c10_npu::NPUTaskGroupHandle::task_group);

    torch_N_m.def("_graph_pool_handle", &c10_npu::graph_pool_handle)
        .def("_graph_task_group_begin", [](py::object py_stream) {
            auto stream = (*py_stream).ptr();
            c10_npu::graph_task_group_begin(THNPUtils_PyObject_to_NPUStream(stream));
        })
        .def("_graph_task_group_end", [](py::object py_stream) {
            auto stream = (*py_stream).ptr();
            return c10_npu::graph_task_group_end(THNPUtils_PyObject_to_NPUStream(stream));
        })
        .def("_graph_task_update_begin", [](py::object py_stream, c10_npu::NPUTaskGroupHandle handle) {
            auto stream = (*py_stream).ptr();
            c10_npu::graph_task_update_begin(THNPUtils_PyObject_to_NPUStream(stream), handle);
        })
        .def("_graph_task_update_end", [](py::object py_stream) {
            auto stream = (*py_stream).ptr();
            c10_npu::graph_task_update_end(THNPUtils_PyObject_to_NPUStream(stream));
        })
        .def("_launch_host_func", [](py::object py_stream, py::object py_func, py::object py_data) {
            auto func = (*py_func).ptr();
            auto userDataList = (*py_data).ptr();
            auto stream = THNPUtils_PyObject_to_NPUStream((*py_stream).ptr());
            PyFuncStruct *data = new(std::nothrow) PyFuncStruct(func, userDataList);
            c10_npu::launch_callback(stream, LaunchCallFunc, data);
            callbacks[stream].emplace_back(data);
        })
        .def("_subscribe_report", [](py::object py_stream) {
            auto stream = (*py_stream).ptr();
            aclrtContext context = aclrtContext();
            NPU_CHECK_ERROR(aclrtGetCurrentContext(&context));
            if ((threadArgs == nullptr) || (threadId == -1)) {
                threadArgs = new ThreadArgs(context, false);
                pthread_create(&threadId, nullptr, process_callback, threadArgs);
            }
            c10_npu::subscribe_report(threadId, THNPUtils_PyObject_to_NPUStream(stream));
        })
        .def("_unsubscribe_report", [](py::object py_stream) {
            auto stream = THNPUtils_PyObject_to_NPUStream((*py_stream).ptr());
            c10_npu::unsubscribe_report(threadId, stream);
            auto it = callbacks.find(stream);
            if (it != callbacks.end()) {
                std::vector<PyFuncStruct *>& funcs = it->second;
                for (PyFuncStruct* func : funcs) {
                    delete func;
                    func = nullptr;
                }
                funcs.clear();
                callbacks.erase(it);
            }
            if (callbacks.empty()) {
                threadArgs->exitFlag = true;
                threadId = -1;
            }
        });

    shared_ptr_class_<c10_npu::NPUGraph>(torch_N_m, "_NPUGraph")
        .def(py::init<>())
        .def(
            "capture_begin",
            [](c10_npu::NPUGraph& self,
               std::optional<c10_npu::MempoolId_t> pool_opt,
               std::string capture_error_mode) {
                aclmdlRICaptureMode capture_mode;
                c10_npu::MempoolId_t pool = pool_opt.has_value()
                    ? pool_opt.value() : c10_npu::MempoolId_t{0, 0};
                if (capture_error_mode == "global") {
                    capture_mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
                } else if (capture_error_mode == "thread_local") {
                    capture_mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL;
                } else if (capture_error_mode == "relaxed") {
                    capture_mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
                } else {
                    TORCH_CHECK(
                        false,
                        "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                        capture_error_mode);
                }
                return self.capture_begin(pool, capture_mode);
            },
            py::arg("pool"),
            py::arg("capture_error_mode"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "capture_end",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::capture_end))
        .def(
            "register_generator_state",
            [](c10_npu::NPUGraph& self, py::handle raw_generator) {
                auto generator = THPGenerator_Unwrap(raw_generator.ptr());
                // We've unwrapped Python object to C++ object,
                // so we could release GIL before calling into C++
                py::gil_scoped_release release;
                return self.register_generator_state(generator);
            },
            py::arg("generator"))
        .def(
            "replay",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::replay))
        .def(
            "reset",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::reset))
        .def(
            "pool",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::pool))
        .def(
            "debug_dump",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::debug_dump))
        .def(
            "enable_debug_mode",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::enable_debug_mode));
}
