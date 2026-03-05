#include <thread>
#include <vector>

#include "torch_npu/csrc/npu/Graph.h"

#include "op_plugin/OpApiInterface.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/npu/Event.h"
#include "torch_npu/csrc/npu/Stream.h"

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
        })
        .def("_npu_fused_infer_attention_score_out_graph", [](
            py::object py_stream,
            c10_npu::NPUTaskGroupHandle handle,
            py::object py_event,
            py::args args,
            py::kwargs kwargs
            ) -> std::tuple<at::Tensor &, at::Tensor &> {
                // 1. 定义parser
                static torch::PythonArgParser parser({
                    "npu_fused_infer_attention_score("
                    "Tensor query, Tensor key, Tensor value, *, "
                    "Tensor? pse_shift=None, "
                    "Tensor? atten_mask=None, "
                    "SymIntArrayRef actual_seq_lengths=None, "
                    "SymIntArrayRef actual_seq_lengths_kv=None, "
                    "Tensor? dequant_scale1=None, "
                    "Tensor? quant_scale1=None, "
                    "Tensor? dequant_scale2=None, "
                    "Tensor? quant_scale2=None, " // 10
                    "Tensor? quant_offset2=None, "
                    "Tensor? antiquant_scale=None, "
                    "Tensor? antiquant_offset=None, "
                    "Tensor? key_antiquant_scale=None, "
                    "Tensor? key_antiquant_offset=None, "
                    "Tensor? value_antiquant_scale=None, "
                    "Tensor? value_antiquant_offset=None, "
                    "Tensor? block_table=None, "
                    "Tensor? query_padding_size=None, "
                    "Tensor? kv_padding_size=None, " // 20
                    "Tensor? key_shared_prefix=None, "
                    "Tensor? value_shared_prefix=None, "
                    "SymIntArrayRef actual_shared_prefix_len=None, "
                    "Tensor? query_rope=None, "
                    "Tensor? key_rope=None, "
                    "Tensor? key_rope_antiquant_scale=None, "
                    "int64_t num_heads=1, "
                    "double scale=1.0, "
                    "int64_t pre_tokens=2147483647, "
                    "int64_t next_tokens=2147483647, " // 30
                    "std::string input_layout=\"BSH\", "
                    "int64_t num_key_value_heads=0, "
                    "int64_t sparse_mode=0, "
                    "int64_t inner_precise=0, "
                    "int64_t block_size=0, "
                    "int64_t antiquant_mode=0, "
                    "int64_t key_antiquant_mode=0, "
                    "int64_t value_antiquant_mode=0, "
                    "bool softmax_lse_flag=False, "
                    "Tensor? workspace=None, " // 40
                    "TensorList out)"
                });

                // 2. 转换为原生 PyObject*
                PyObject* args_ptr = args.ptr();
                PyObject* kwargs_ptr = kwargs.ptr();

                // 3. 解析参数
                torch::ParsedArgs<42> parsed;
                torch::PythonArgs py_args = parser.parse(args_ptr, kwargs_ptr, parsed);

                // 4. 必选参数
                at::Tensor query = py_args.tensor(0);
                at::Tensor key = py_args.tensor(1);
                at::Tensor value = py_args.tensor(2);

                // 5. 可选参数
                c10::optional<at::Tensor> pse_shift = py_args.optionalTensor(3);
                c10::optional<at::Tensor> atten_mask = py_args.optionalTensor(4);
                c10::OptionalArray<c10::SymInt> actual_seq_lengths = py_args.symintlistOptional(5);
                c10::OptionalArray<c10::SymInt> actual_seq_lengths_kv = py_args.symintlistOptional(6);
                c10::optional<at::Tensor> dequant_scale1 = py_args.optionalTensor(7);
                c10::optional<at::Tensor> quant_scale1 = py_args.optionalTensor(8);
                c10::optional<at::Tensor> dequant_scale2 = py_args.optionalTensor(9);
                c10::optional<at::Tensor> quant_scale2 = py_args.optionalTensor(10);
                c10::optional<at::Tensor> quant_offset2 = py_args.optionalTensor(11);
                c10::optional<at::Tensor> antiquant_scale = py_args.optionalTensor(12);
                c10::optional<at::Tensor> antiquant_offset = py_args.optionalTensor(13);
                c10::optional<at::Tensor> key_antiquant_scale = py_args.optionalTensor(14);
                c10::optional<at::Tensor> key_antiquant_offset = py_args.optionalTensor(15);
                c10::optional<at::Tensor> value_antiquant_scale = py_args.optionalTensor(16);
                c10::optional<at::Tensor> value_antiquant_offset = py_args.optionalTensor(17);
                c10::optional<at::Tensor> block_table = py_args.optionalTensor(18);
                c10::optional<at::Tensor> query_padding_size = py_args.optionalTensor(19);
                c10::optional<at::Tensor> kv_padding_size = py_args.optionalTensor(20);
                c10::optional<at::Tensor> key_shared_prefix = py_args.optionalTensor(21);
                c10::optional<at::Tensor> value_shared_prefix = py_args.optionalTensor(22);
                c10::OptionalArray<c10::SymInt> actual_shared_prefix_len = py_args.symintlistOptional(23);
                c10::optional<at::Tensor> query_rope = py_args.optionalTensor(24);
                c10::optional<at::Tensor> key_rope = py_args.optionalTensor(25);
                c10::optional<at::Tensor> key_rope_antiquant_scale = py_args.optionalTensor(26);
                int64_t num_heads = py_args.toInt64(27);
                double scale = py_args.toDouble(28);
                int64_t pre_tokens = py_args.toInt64(29);
                int64_t next_tokens = py_args.toInt64(30);
                std::string input_layout = py_args.string(31);
                int64_t num_key_value_heads = py_args.toInt64(32);
                int64_t sparse_mode = py_args.toInt64(33);
                int64_t inner_precise = py_args.toInt64(34);
                int64_t block_size = py_args.toInt64(35);
                int64_t antiquant_mode = py_args.toInt64(36);
                int64_t key_antiquant_mode = py_args.toInt64(37);
                int64_t value_antiquant_mode = py_args.toInt64(38);
                bool softmax_lse_flag = py_args.toBool(39);
                c10::optional<at::Tensor> workspace = py_args.optionalTensor(40);
                std::vector<at::Tensor> out = py_args.tensorlist(41);
                TORCH_CHECK(out.size() == 2,
                    "out must have 2 tensors (attention_out, softmax_lse), but got ",
                    out.size(), PTA_ERROR(ErrCode::PARAM));
                at::Tensor attention_out = out[0];
                at::Tensor softmax_lse = out[1];

                auto stream = THNPUtils_PyObject_to_NPUStream((*py_stream).ptr());
                auto event_ptr = THNPUtils_PyObject_to_NPUEvent((*py_event).ptr());
                pybind11::gil_scoped_release no_gil;

                c10_npu::graph_task_update_begin(stream, handle);

                auto fia_result = op_api::npu_fused_infer_attention_score_out_symint(
                    query, key, value,
                    pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
                    dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
                    antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset,
                    value_antiquant_scale, value_antiquant_offset, block_table,
                    query_padding_size, kv_padding_size, key_shared_prefix, value_shared_prefix,
                    actual_shared_prefix_len, query_rope, key_rope, key_rope_antiquant_scale,
                    num_heads, scale, pre_tokens, next_tokens, input_layout,
                    num_key_value_heads, sparse_mode, inner_precise, block_size,
                    antiquant_mode, key_antiquant_mode, value_antiquant_mode,
                    softmax_lse_flag, workspace, attention_out, softmax_lse
                );

                c10_npu::graph_task_update_end(stream);
                event_ptr->record(stream);
                return fia_result;
        });

    shared_ptr_class_<c10_npu::NPUGraph>(torch_N_m, "_NPUGraph")
        .def(py::init<>())
        .def(
            "capture_begin",
            [](c10_npu::NPUGraph& self,
               std::optional<c10_npu::MempoolId_t> pool_opt,
               std::string capture_error_mode,
			   bool report_shape) {
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
                return self.capture_begin(pool, capture_mode, report_shape);
            },
            py::arg("pool"),
            py::arg("capture_error_mode"),
            py::arg("report_shape"),
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
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::debug_dump),
            py::arg("debug_path"))
        .def(
            "enable_debug_mode",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::enable_debug_mode));
}
