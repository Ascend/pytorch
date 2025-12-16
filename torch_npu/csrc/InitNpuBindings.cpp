#include <Python.h>
#include <ATen/Parallel.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/profiler/python/combined_traceback.h>

#include "torch_npu/csrc/npu/Event.h"
#include "torch_npu/csrc/npu/DataParallelComm.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUSwappedMemoryAllocator.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/inductor/aoti_package/shape_handling.h"
#include "torch_npu/csrc/distributed/Init.h"
#include "torch_npu/csrc/afd/Init.h"
#include "torch_npu/csrc/profiler/init.h"
#include "torch_npu/csrc/flopcount/Init.h"
#include "torch_npu/csrc/logging/Init.h"
#include "torch_npu/csrc/ipc/StorageSharing.h"
#include "torch_npu/csrc/npu/Module.h"
#include "torch_npu/csrc/custom_dtype/Init.h"
#include "torch_npu/csrc/npu/Stress_detect.h"
#include "torch_npu/csrc/utils/TensorType.h"
#include "torch_npu/csrc/utils/AutocastMode.h"
#include "torch_npu/csrc/core/npu/NPURecovery.h"
#include "torch_npu/csrc/profiler/python/combined_traceback.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/sanitizer/NPUTrace.h"
#endif

PyObject* module;


void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods)
{
    if (!vector.empty()) {
        // remove nullptr terminator
        vector.pop_back();
    }
    while (true) {
        vector.push_back(*methods);
        if (!methods->ml_name) {
            break;
        }
        methods++;
    }
}

PyObject* THPModule_npu_shutdown(PyObject* self, PyObject* arg)
{
    int check_error;
    if (!PyBool_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Expected a boolean value");
        return NULL;
    }
    check_error = PyObject_IsTrue(arg);

    // cudaFree is blocking and will synchronize across all kernels executing
    // on the current device, while aclrtFree Free device memory immediately.
    // aclrtSynchronizeDevice should be called before aclrtFree to ensure that
    // all of op tasks completed before device memory free.
    ASCEND_LOGI("NPU shutdown begin.");
    if (!c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        Py_RETURN_NONE;
    }

    ASCEND_LOGI("NPU shutdown ReleaseHcclCommList.");
    torch_npu::data_parallel::ReleaseHcclCommList();
    ASCEND_LOGI("NPU shutdown ReleaseHcclCommList success.");

    c10_npu::NpuSysCtrl::GetInstance().HostFinalize();
    at_npu::native::CachingHostAllocator_emptyCache();
    try {
        ASCEND_LOGI("NPU shutdown NPUCachingAllocator emptyCache.");
        c10_npu::NPUCachingAllocator::emptyCache(false);
    } catch (...) {
        ASCEND_LOGE("NPUCachingAllocator::emptyCache failed");
    }
    try {
        ASCEND_LOGI("NPU shutdown NPUSwappedMemoryAllocator emptyCache.");
        c10_npu::NPUSwappedMemoryAllocator::emptyCache();
    } catch (...) {
        ASCEND_LOGE("NPUSwappedMemoryAllocator::emptyCache failed");
    }

    ASCEND_LOGI("NPU shutdown NpuSysCtrl Finalize.");
    c10_npu::NpuSysCtrl::SysStatus status = c10_npu::NpuSysCtrl::GetInstance().Finalize();
    if (status != c10_npu::NpuSysCtrl::SysStatus::FINALIZE_SUCC) {
        ASCEND_LOGE("NPU shutdown failed.");
    } else {
        ASCEND_LOGI("NPU shutdown success.");
    }

    Py_RETURN_NONE;
}

PyObject* THPModule_npu_shutdown_synchronize(PyObject* /* unused */)
{
    ASCEND_LOGI("NPU shutdown synchronize begin.");
    if (!c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        Py_RETURN_FALSE;
    }

    StressDetector::stop_worker_thread();

    // Return aclrtSynchronizeDevice result. If sync device fails, release host
    // resources forcibly, only record WARN logs when acl interface of stream
    // or event fails.
    bool success = true;
    try {
        ASCEND_LOGI("NPU shutdown synchronize device.");
        success = c10_npu::npuSynchronizeUsedDevices(false);
    } catch (std::exception& e) {
        ASCEND_LOGE("npuSynchronizeDevice failed err=:%s", e.what());
        success = false;
    }

    if (success) {
        Py_RETURN_TRUE;
    } else {
        ASCEND_LOGE("NPU shutdown synchronize device failed.");
        Py_RETURN_FALSE;
    }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
static PyMethodDef TorchNpuMethods[] = {
    {"_npu_shutdown", (PyCFunction)THPModule_npu_shutdown, METH_O, nullptr},
    {"_npu_shutdown_synchronize", (PyCFunction)THPModule_npu_shutdown_synchronize, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

#ifndef BUILD_LIBTORCH
PyObject* THPModule_sanitizer_enable(PyObject* /* unused */, PyObject* args)
{
    int mode;
    if (!PyArg_ParseTuple(args, "i", &mode)) {
        return NULL;
    }
    c10_npu::impl::activateNPUTrace(mode);
    Py_RETURN_NONE;
}

static PyMethodDef TorchSanitizerMethods[] = {
    {"_activate_npu_trace", (PyCFunction)THPModule_sanitizer_enable, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};
#endif

void THNPStream_init(PyObject *module);
void THNPEvent_init(PyObject *module);
void THNPGraph_init(PyObject *module);
void THNPMemPool_init(PyObject* module);
void THNPShapeHandling_init(PyObject* module);
PyMethodDef* THNPModule_get_methods();

static std::vector<PyMethodDef> methods;

extern "C"

PyObject* initModule()
{
    at::internal::lazy_init_num_threads();

    AddPyMethodDefs(methods, TorchNpuMethods);
#ifndef BUILD_LIBTORCH
    AddPyMethodDefs(methods, TorchSanitizerMethods);
#endif
    AddPyMethodDefs(methods, THNPModule_get_methods());
    AddPyMethodDefs(methods, torch_npu::profiler::profiler_functions());
    AddPyMethodDefs(methods, torch_npu::distributed::python_functions());
    AddPyMethodDefs(methods, torch_npu::utils::npu_extension_functions());
    AddPyMethodDefs(methods, torch_npu::autocast::autocast_mode_functions());
    AddPyMethodDefs(methods, torch_npu::flopcount::flops_count_functions());
    AddPyMethodDefs(methods, torch_npu::logging::logging_functions());
    AddPyMethodDefs(methods, torch_npu::reductions::reductions_functions());
    AddPyMethodDefs(methods, c10_npu::custom_dtype_functions());
    AddPyMethodDefs(methods, torch_npu::afd::python_functions());
    static struct PyModuleDef torchnpu_module = {
        PyModuleDef_HEAD_INIT,
        "torch_npu._C",
        nullptr,
        -1,
        methods.data()
    };
    module = PyModule_Create(&torchnpu_module);

    // This will only initialize base classes and attach them to library namespace
    // They won't be ready for real usage until importing npu module, that will
    // complete the process (but it defines Python classes before calling back into
    // C, so these lines have to execute first)..
    THNPStream_init(module);
    THNPEvent_init(module);
    THNPGraph_init(module);
    THNPMemPool_init(module);
    THNPShapeHandling_init(module);

    RegisterNPUDeviceProperties(module);
    BindGetDeviceProperties(module);
    RegisterNPUDeviceMemories(module);
    BindGetDeviceMemories(module);
    RegisterNpuPluggableAllocator(module);
#ifndef BUILD_LIBTORCH
    c10_npu::bind_npu_recovery_functions(module);
#endif
    initCommMethods();
    torch::installCapturedTracebackPython();
    torch_npu::installCapturedTracebackPython();
    torch_npu::profiler::initMstx(module);
    return module;
}

PyMODINIT_FUNC PyInit__C(void)
{
    return initModule();
}
