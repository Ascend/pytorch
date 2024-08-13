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

#include <torch/csrc/python_headers.h>

#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/utils/python_arg_parsing.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include "torch_npu/csrc/profiler/init.h"
#include "torch_npu/csrc/profiler/profiler.h"
#include "torch_npu/csrc/profiler/profiler_python.h"
#include "torch_npu/csrc/profiler/npu_profiler.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/framework/interface/LibAscendHal.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace torch_npu {
namespace profiler {

PyObject* profiler_initExtension(PyObject* _unused, PyObject *unused) {
    auto torch_npu_C_module = THPObjectPtr(PyImport_ImportModule("torch_npu._C"));
    if (!torch_npu_C_module) {
        return nullptr;
    }
    auto torch_npu_C_m = py::handle(torch_npu_C_module).cast<py::module>();
    auto m = torch_npu_C_m.def_submodule("_profiler", "_profiler bindings");

    py::enum_<ProfilerState>(m, "ProfilerState")
        .value("Disabled", ProfilerState::Disabled)
        .value("CPU", ProfilerState::CPU)
        .value("CUDA", ProfilerState::CUDA)
        .value("NPU", ProfilerState::NPU)
        .value("NVTX", ProfilerState::NVTX)
        .value("KINETO", ProfilerState::KINETO);

    py::class_<ProfilerConfig>(m, "ProfilerConfig")
        .def(py::init<ProfilerState, bool, bool, bool, bool, bool>());

    py::enum_<NpuActivityType>(m, "ProfilerActivity")
        .value("CPU", NpuActivityType::CPU)
        .value("NPU", NpuActivityType::NPU);

    py::class_<ExperimentalConfig>(m, "_ExperimentalConfig")
        .def(py::init<std::string, std::string, bool, bool, bool, bool>(),
             py::arg("trace_level") = "Level0",
             py::arg("metrics") = "ACL_AICORE_NONE",
             py::arg("l2_cache") = false,
             py::arg("record_op_args") = false,
             py::arg("msprof_tx") = false,
             py::arg("op_attr") = false
        )
        .def(py::pickle(
            [](const ExperimentalConfig& p) {
                return py::make_tuple(p.trace_level, p.metrics, p.l2_cache, p.record_op_args, p.msprof_tx, p.op_attr);
            },
            [](py::tuple t) {
                if (t.size() < 6) {  // 6表示ExperimentalConfig的配置有六项
                    throw std::runtime_error("Expected atleast 5 values in state" + PROF_ERROR(ErrCode::PARAM));
                }
                return ExperimentalConfig(
                    t[0].cast<std::string>(),
                    t[1].cast<std::string>(),
                    t[2].cast<bool>(),
                    t[3].cast<bool>(),
                    t[4].cast<bool>(),
                    t[5].cast<bool>()
                );
            }
        ));

    py::class_<NpuProfilerConfig>(m, "NpuProfilerConfig")
        .def(py::init<std::string, bool, bool, bool, bool, bool, ExperimentalConfig>());

    py::class_<LegacyEvent>(m, "ProfilerEvent")
        .def("kind", &LegacyEvent::kindStr)
        .def("name", [](const LegacyEvent& e) { return e.name(); })
        .def("thread_id", &LegacyEvent::threadId)
        .def("fwd_thread_id", &LegacyEvent::fwdThreadId)
        .def("device", &LegacyEvent::device)
        .def("cpu_elapsed_us", &LegacyEvent::cpuElapsedUs)
        .def("cuda_elapsed_us", &LegacyEvent::cudaElapsedUs)
        .def("npu_elapsed_us", &LegacyEvent::npuElapsedUs)
        .def("npu_destropy_event", &LegacyEvent::npu_destropy_event)
        .def("has_cuda", &LegacyEvent::hasCuda)
        .def("has_npu", &LegacyEvent::hasNpu)
        .def("shapes", &LegacyEvent::shapes)
        .def("cpu_memory_usage", &LegacyEvent::cpuMemoryUsage)
        .def("cuda_memory_usage", &LegacyEvent::cudaMemoryUsage)
        .def("npu_memory_usage", &LegacyEvent::npuMemoryUsage)
        .def("handle", &LegacyEvent::handle)
        .def("node_id", &LegacyEvent::nodeId)
        .def("is_remote", &LegacyEvent::isRemote)
        .def("sequence_nr", &LegacyEvent::sequenceNr)
        .def("stack", &LegacyEvent::stack)
        .def("scope", &LegacyEvent::scope)
        .def("correlation_id", &LegacyEvent::correlationId)
        .def("start_us", &LegacyEvent::cpuUs)
        .def("flops", &LegacyEvent::flops);

    m.def("_record_function_enter", record_function_enter);
    m.def("_record_function_exit", record_function_exit);

    m.def("_enable_profiler_legacy", enableProfilerLegacy);
    py::class_<ProfilerDisableOptions>(m, "_ProfilerDisableOptions")
        .def(py::init<bool, bool>());
    m.def(
        "_disable_profiler_legacy",
        disableProfilerLegacy,
        py::arg("profiler_disable_options") = ProfilerDisableOptions());
    m.def("_profiler_enabled", profilerEnabled);
    m.def("_enable_record_function", [](bool enable) {
        at::enableRecordFunction(enable);
    });
    m.def("_set_empty_test_observer", [](bool is_global, double sampling_prob) {
        auto cb = at::RecordFunctionCallback(nullptr)
                  .needsInputs(true)
                  .samplingProb(sampling_prob);
        if (is_global) {
            at::addGlobalCallback(cb);
        } else {
            at::addThreadLocalCallback(cb);
        }
    });
    m.def("_clear_callbacks", []() {
        at::clearCallbacks();
    });

    m.def("_supported_npu_activities", []() {
        std::set<NpuActivityType> activities {
            NpuActivityType::CPU,
            NpuActivityType::NPU
        };
        return activities;
    });
    m.def("_init_profiler", initNpuProfiler);
    m.def("_start_profiler",
          &startNpuProfiler,
          py::arg("config"),
          py::arg("activities"),
          py::arg("scopes") = std::unordered_set<at::RecordScope>());
    m.def("_stop_profiler", stopNpuProfiler);
    m.def("_finalize_profiler", finalizeNpuProfiler);
    m.def("_get_freq", at_npu::native::getFreq);
    m.def("_get_syscnt_enable", at_npu::native::isSyscntEnable);
    m.def("_get_syscnt", torch_npu::toolkit::profiler::Utils::getClockSyscnt);
    m.def("_get_monotonic", torch_npu::toolkit::profiler::Utils::GetClockMonotonicRawNs);
    m.def("_get_host_uid", torch_npu::toolkit::profiler::Utils::GetHostUid);

    torch_npu::profiler::python_tracer::init();
    Py_RETURN_TRUE;
}

// autograd methods on torch._C
static PyMethodDef TorchProfilerMethods[] = { // NOLINT
    {"_profiler_init", profiler_initExtension, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};


PyMethodDef* profiler_functions() {
    return TorchProfilerMethods;
}

PyObject* THNPModule_rangeStart(PyObject* _unused, PyObject* args)
{
    HANDLE_TH_ERRORS
    char *message;
    PyObject* stream_o = nullptr;
    if (!PyArg_ParseTuple(args, "sO", &message, &stream_o)) {
        return nullptr;
    }
    aclrtStream stream = static_cast<aclrtStream>(PyLong_AsVoidPtr(stream_o));
    int id = mstxRangeStart(message, stream);
    return PyLong_FromLong(id);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_rangeStartOnHost(PyObject* _unused, PyObject* args)
{
    HANDLE_TH_ERRORS
    char *message;
    if (!PyArg_ParseTuple(args, "s", &message)) {
        return nullptr;
    }
    int id = mstxRangeStart(message, nullptr);
    return PyLong_FromLong(id);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_rangeEnd(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    mstxRangeId rangeId;
    if (!PyArg_ParseTuple(args, "k", &rangeId)) {
        return nullptr;
    }
    mstxRangeEnd(rangeId);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static std::vector<PyMethodDef> mstxMethods = {
    {"_range_start_on_host", (PyCFunction)THNPModule_rangeStartOnHost, METH_VARARGS, nullptr},
    {"_range_start", (PyCFunction)THNPModule_rangeStart, METH_VARARGS, nullptr},
    {"_range_end", (PyCFunction)THNPModule_rangeEnd, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

void initMstx(PyObject *module)
{
    static struct PyModuleDef mstx_module = {
        PyModuleDef_HEAD_INIT,
        "_mstx",
        nullptr,
        -1,
        mstxMethods.data()
    };
    PyObject* mstxModule = PyModule_Create(&mstx_module);
    if (mstxModule == nullptr) {
        return;
    }
    PyModule_AddObject(module, "_mstx", mstxModule);
}
}
}
