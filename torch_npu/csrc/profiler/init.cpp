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
