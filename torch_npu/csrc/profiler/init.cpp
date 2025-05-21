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

PyObject* profiler_initExtension(PyObject* _unused, PyObject *unused)
{
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
        .def(py::init<std::string, std::string, bool, bool, bool, bool, std::vector<std::string>,
             std::vector<std::string>, std::vector<std::string>, bool, bool>(),
             py::arg("trace_level") = "Level0",
             py::arg("metrics") = "ACL_AICORE_NONE",
             py::arg("l2_cache") = false,
             py::arg("record_op_args") = false,
             py::arg("msprof_tx") = false,
             py::arg("op_attr") = false,
             py::arg("host_sys") = std::vector<std::string>{},
             py::arg("mstx_domain_include") = std::vector<std::string>{},
             py::arg("mstx_domain_exclude") = std::vector<std::string>{},
             py::arg("sys_io") = false,
             py::arg("sys_interconnection") = false
        )
        .def(py::pickle(
            [](const ExperimentalConfig& p) {
                return py::make_tuple(p.trace_level, p.metrics, p.l2_cache, p.record_op_args, p.msprof_tx, p.op_attr,
                                      p.host_sys, p.mstx_domain_include, p.mstx_domain_exclude, p.sys_io,
                                      p.sys_interconnection);
            },
            [](py::tuple t) {
                if (t.size() < static_cast<size_t>(ExperConfigType::CONFIG_TYPE_MAX_COUNT)) {
                    throw std::runtime_error(
                        "Expected at least " + std::to_string(static_cast<size_t>(ExperConfigType::CONFIG_TYPE_MAX_COUNT)) +
                        " values in state" + PROF_ERROR(ErrCode::PARAM)
                    );
                }
                return ExperimentalConfig(
                    t[static_cast<size_t>(ExperConfigType::TRACE_LEVEL)].cast<std::string>(),
                    t[static_cast<size_t>(ExperConfigType::METRICS)].cast<std::string>(),
                    t[static_cast<size_t>(ExperConfigType::L2_CACHE)].cast<bool>(),
                    t[static_cast<size_t>(ExperConfigType::RECORD_OP_ARGS)].cast<bool>(),
                    t[static_cast<size_t>(ExperConfigType::MSPROF_TX)].cast<bool>(),
                    t[static_cast<size_t>(ExperConfigType::OP_ATTR)].cast<bool>(),
                    t[static_cast<size_t>(ExperConfigType::HOST_SYS)].cast<std::vector<std::string>>(),
                    t[static_cast<size_t>(ExperConfigType::MSTX_DOMAIN_INCLUDE)].cast<std::vector<std::string>>(),
                    t[static_cast<size_t>(ExperConfigType::MSTX_DOMAIN_EXCLUDE)].cast<std::vector<std::string>>(),
                    t[static_cast<size_t>(ExperConfigType::SYS_IO)].cast<bool>(),
                    t[static_cast<size_t>(ExperConfigType::SYS_INTERCONNECTION)].cast<bool>()
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
    m.def("_warmup_profiler", &warmupNpuProfiler, py::arg("config"), py::arg("activities"));
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


PyMethodDef* profiler_functions()
{
    return TorchProfilerMethods;
}

PyObject* THNPModule_markOnHost(PyObject* _unused, PyObject* args)
{
    HANDLE_TH_ERRORS
    const char* message;
    const char* domain;
    if (!PyArg_ParseTuple(args, "ss", &message, &domain)) {
        return nullptr;
    }
    mstxMark(message, nullptr, domain);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_mark(PyObject* _unused, PyObject* args)
{
    HANDLE_TH_ERRORS
    const char* message;
    const char* domain;
    PyObject* stream_o = nullptr;
    if (!PyArg_ParseTuple(args, "sOs", &message, &stream_o, &domain)) {
        return nullptr;
    }
    aclrtStream stream = static_cast<aclrtStream>(PyLong_AsVoidPtr(stream_o));
    mstxMark(message, stream, domain);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_rangeStart(PyObject* _unused, PyObject* args)
{
    HANDLE_TH_ERRORS
    const char* message;
    const char* domain;
    PyObject* stream_o = nullptr;
    if (!PyArg_ParseTuple(args, "sOs", &message, &stream_o, &domain)) {
        return nullptr;
    }
    aclrtStream stream = static_cast<aclrtStream>(PyLong_AsVoidPtr(stream_o));
    int id = mstxRangeStart(message, stream, domain);
    return PyLong_FromLong(id);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_rangeStartOnHost(PyObject* _unused, PyObject* args)
{
    HANDLE_TH_ERRORS
    const char* message;
    const char* domain;
    if (!PyArg_ParseTuple(args, "ss", &message, &domain)) {
        return nullptr;
    }
    int id = mstxRangeStart(message, nullptr, domain);
    return PyLong_FromLong(id);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_rangeEnd(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    int rangeId;
    const char* domain;
    if (!PyArg_ParseTuple(args, "is", &rangeId, &domain)) {
        return nullptr;
    }
    mstxRangeEnd(rangeId, domain);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static std::vector<PyMethodDef> mstxMethods = {
    {"_mark_on_host", (PyCFunction)THNPModule_markOnHost, METH_VARARGS, nullptr},
    {"_mark", (PyCFunction)THNPModule_mark, METH_VARARGS, nullptr},
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
