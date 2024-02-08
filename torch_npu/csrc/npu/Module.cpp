#include <chrono>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <ATen/ATen.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_numbers.h>

#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/aten/common/SetNpu.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/OverflowUtils.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/profiler/cann_profiling.h"
#include "torch_npu/csrc/profiler/e2e_profiler.h"
#include "torch_npu/csrc/npu/Module.h"
#include "torch_npu/csrc/npu/NPUPluggableAllocator.h"
#include "torch_npu/csrc/utils/LazyInit.h"
#include "third_party/acl/inc/acl/acl.h"

struct NPUDeviceProp {
    std::string name;
    size_t totalGlobalMem = 0;
};

struct NPUDeviceMem {
    size_t totalGlobalMem = 0;
    size_t freeMem = 0;
};
NPUDeviceProp prop;
void RegisterNPUDeviceProperties(PyObject* module)
{
    auto m = py::handle(module).cast<py::module>();
    py::class_<NPUDeviceProp>(m, "_NPUDeviceProperties")
              .def_readonly("name", &NPUDeviceProp::name)
              .def_readonly("total_memory", &NPUDeviceProp::totalGlobalMem)
              .def("__repr__", [](const NPUDeviceProp &prop) {
                std::ostringstream stream;
                stream << "_NPUDeviceProperties(name='" << prop.name << "', total_memory="
                  << prop.totalGlobalMem / (CHANGE_UNIT_SIZE * CHANGE_UNIT_SIZE) << "MB)";
                return stream.str();
              });
}

NPUDeviceProp* GetDeviceProperties(int64_t deviceid)
{
    const char* device_name;
    size_t device_free;
    size_t device_total;
    device_name = c10_npu::acl::AclrtGetSocName();
    if (device_name == nullptr) {
      prop.name = " ";
      ASCEND_LOGE("NPU get device name fail.");
    } else {
      prop.name = std::string(device_name);
    }
    NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
    prop.totalGlobalMem = device_total;
    return &prop;
}

void BindGetDeviceProperties(PyObject* module)
{
    auto m = py::handle(module).cast<py::module>();
    m.def("_npu_getDeviceProperties", [](int deviceid) -> NPUDeviceProp* {
      return GetDeviceProperties(deviceid);
    }, py::return_value_policy::reference);
}

NPUDeviceMem memory;
void RegisterNPUDeviceMemories(PyObject* module)
{
    auto m = py::handle(module).cast<py::module>();
    py::class_<NPUDeviceMem>(m, "_NPUDeviceMemories")
              .def_readonly("total_memory", &NPUDeviceMem::totalGlobalMem)
              .def_readonly("free_memory", &NPUDeviceMem::freeMem);
}

NPUDeviceMem* GetDeviceMemories(int64_t deviceid)
{
    c10_npu::NPUGuard guard(deviceid);
    size_t device_free;
    size_t device_total;
    NPU_CHECK_ERROR(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
    memory.totalGlobalMem = device_total;
    memory.freeMem = device_free;
    return &memory;
}

void BindGetDeviceMemories(PyObject* module)
{
    auto m = py::handle(module).cast<py::module>();
    m.def("_npu_getDeviceMemories", [](int deviceid) -> NPUDeviceMem* {
      return GetDeviceMemories(deviceid);
    }, py::return_value_policy::reference);
}

void RegisterNpuPluggableAllocator(PyObject* module)
{
    auto m = py::handle(module).cast<py::module>();

    py::class_<
        c10_npu::NPUCachingAllocator::NPUAllocator,
        std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator>>(
        m, "_npu_NPUAllocator");
    m.def("_npu_getAllocator", []() {
      return py::cast(torch::npu::NPUPluggableAllocator::getCurrentAllocator());
    });

    m.def(
        "_npu_changeCurrentAllocator",
        [](std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator>
              allocator) {
            torch::npu::NPUPluggableAllocator::changeCurrentAllocator(allocator);
        });
    py::class_<
        torch::npu::NPUPluggableAllocator::NPUPluggableAllocator,
        c10_npu::NPUCachingAllocator::NPUAllocator,
        std::shared_ptr<
            torch::npu::NPUPluggableAllocator::NPUPluggableAllocator>>(
        m, "_NPUPluggableAllocator")
        .def(
        "set_init_fn",
        [](torch::npu::NPUPluggableAllocator::NPUPluggableAllocator& self,
            uint64_t func_ptr) {
            using FuncType = void(int);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_init_fn(func);
        })
        .def(
        "set_reset_fn",
        [](torch::npu::NPUPluggableAllocator::NPUPluggableAllocator& self,
            uint64_t func_ptr) {
            using FuncType = void(bool);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_reset_fn(func);
        })
        .def(
        "set_memory_fraction_fn",
        [](torch::npu::NPUPluggableAllocator::NPUPluggableAllocator& self,
            uint64_t func_ptr) {
            using FuncType = void(double, int);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_memory_fraction_fn(func);
        })
        .def(
        "set_base_alloc_fn",
        [](torch::npu::NPUPluggableAllocator::NPUPluggableAllocator& self,
            uint64_t func_ptr) {
            using FuncType = void*(void*, size_t*);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_base_alloc_fn(func);
        })
        .def(
        "set_record_stream_fn",
        [](torch::npu::NPUPluggableAllocator::NPUPluggableAllocator& self,
            uint64_t func_ptr) {
            using FuncType = void(void*, aclrtStream);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_record_stream_fn(func);
        })
        .def(
        "set_erase_stream_fn",
        [](torch::npu::NPUPluggableAllocator::NPUPluggableAllocator& self,
            uint64_t func_ptr) {
            using FuncType = void(void*, aclrtStream);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_erase_stream_fn(func);
        });
    m.def("_npu_customAllocator", [](uint64_t malloc_ptr, uint64_t free_ptr) {
        using MallocFuncType = void*(size_t, int, aclrtStream);
        using FreeFuncType = void(void*, size_t, int, aclrtStream);
        std::function<MallocFuncType> malloc_fn =
            reinterpret_cast<MallocFuncType*>(malloc_ptr);
        std::function<FreeFuncType> free_fn =
            reinterpret_cast<FreeFuncType*>(free_ptr);
        return torch::npu::NPUPluggableAllocator::createCustomAllocator(
            malloc_fn, free_fn);
    });
}

static PyObject* THNPModule_initExtension(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    {
      pybind11::gil_scoped_release no_gil;
      c10_npu::NpuSysCtrl::SysStatus status =
          c10_npu::NpuSysCtrl::GetInstance().Initialize();
      if (status !=
          c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
        throw python_error();
      }
    }
    auto m = THPObjectPtr(PyImport_ImportModule("torch.npu"));
    if (!m) {
        throw python_error();
    }

    auto set_module_attr = [&](const char* name, PyObject* v) {
      // PyObject_SetAttrString doesn't steal reference. So no need to incref.
      if (PyObject_SetAttrString(m, name, v) < 0) {
        throw python_error();
      }
    };
    auto num_npus = c10_npu::device_count();
    auto default_npu_generators = PyTuple_New(static_cast<Py_ssize_t>(num_npus));
    for (int i = 0; i < num_npus; i++) {
      auto gen = at_npu::detail::getDefaultNPUGenerator(i);
      auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
      // This reference is meant to be given away, so no need to incref here.
      PyTuple_SetItem(default_npu_generators, i, (PyObject*)cast_gen);
    }
    set_module_attr("default_generators", default_npu_generators);

    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuSynchronize(PyObject* _unused, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    pybind11::gil_scoped_release no_gil;
    c10_npu::npuSynchronizeDevice();
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

void THNPModule_setDevice(int device)
{
    NPU_CHECK_ERROR(c10_npu::SetDevice(device));
}

PyObject* THNPModule_setDevice_wrap(PyObject* self, PyObject* arg)
{
    HANDLE_TH_ERRORS
    int device = THPUtils_unpackLong(arg);
    {
      pybind11::gil_scoped_release no_gil;
      c10_npu::NpuSysCtrl::SysStatus status =
          c10_npu::NpuSysCtrl::GetInstance().Initialize(device);
      if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
        ASCEND_LOGE("Npu init fail.");
      }
    }

    int pre_device = 0;
    auto ret = c10_npu::GetDevice(&pre_device);
    if (ret != ACL_ERROR_NONE) {
        NPU_CHECK_ERROR(c10_npu::SetDevice(device));
    } else if (pre_device != device) {
        c10_npu::NpuSysCtrl::GetInstance().ExchangeDevice(pre_device, device);
    }

    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDevice_wrap(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    int device;
    torch_npu::utils::npu_lazy_init();
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    return PyLong_FromLong(device);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    return PyLong_FromLong(c10_npu::device_count());
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuCanDeviceAccessPeer_wrap(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    PyObject *value_1 = nullptr;
    PyObject *value_2 = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &value_1, &value_2)) {
      throw torch::TypeError("Pybind failed to parse parameters.");
    }
    int32_t device_id = THPUtils_unpackInt(value_1);
    int32_t peer_device_id = THPUtils_unpackInt(value_2);
    auto can_access_peer = c10_npu::acl::can_device_access_peer(device_id, peer_device_id);
    return PyBool_FromLong(can_access_peer);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDeviceUtilizationRate_wrap(PyObject* self, PyObject* device_index)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(device_index), "invalid argument to getDeviceUtilizationRate");
    int32_t device = static_cast<int32_t>(THPUtils_unpackUInt32(device_index));
    aclrtUtilizationInfo util_info;
    util_info.cubeUtilization = 0;
    util_info.vectorUtilization = 0;
    util_info.utilizationExtend = nullptr;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtGetDeviceUtilizationRate(device, &util_info));
    int32_t cube = util_info.cubeUtilization;
    int32_t vector = util_info.vectorUtilization;
    int32_t util_rate = 0;
    // 如果vector和cube谁支持,就返回谁的使用率，如果都支持计算(vector*1+cube*1)/2
    if (cube == DEVICE_UTILIZATION_NOT_SUPPORT && vector != DEVICE_UTILIZATION_NOT_SUPPORT) {
        util_rate = vector;
    } else if (cube != DEVICE_UTILIZATION_NOT_SUPPORT && vector == DEVICE_UTILIZATION_NOT_SUPPORT) {
        util_rate = cube;
    } else if (cube != DEVICE_UTILIZATION_NOT_SUPPORT && vector != DEVICE_UTILIZATION_NOT_SUPPORT) {
        util_rate = (cube + vector) / 2;
    }
    TORCH_CHECK(util_rate <=100 && util_rate >= 0, "invalid result to util_rate");
    return PyLong_FromLong(util_rate);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getCurrentStream_wrap(
    PyObject * /* unused */, PyObject *device_index)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
    int64_t device = THPUtils_unpackLong(device_index);
    auto stream = c10_npu::getCurrentNPUStream(device);
    PyObject* output_tuple = PyTuple_New(3);
    PyTuple_SetItem(
        output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
    PyTuple_SetItem(
        output_tuple,
        1,
        THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
    PyTuple_SetItem(
        output_tuple,
        2,
        THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
    return output_tuple;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDefaultStream_wrap(PyObject *self /* unused */, PyObject *device_index)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
    int64_t device = THPUtils_unpackLong(device_index);
    auto stream = c10_npu::getDefaultNPUStream(device);
    PyObject* output_tuple = PyTuple_New(3);
    PyTuple_SetItem(
        output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
    PyTuple_SetItem(
        output_tuple,
        1,
        THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
    PyTuple_SetItem(
        output_tuple,
        2,
        THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
    return output_tuple;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_setStream_wrap(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs)
{
    HANDLE_TH_ERRORS
    int64_t stream_id = 0;
    int64_t device_index = 0;
    int64_t device_type = 0;

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    constexpr char* kwlist[] = {
        "stream_id", "device_index", "device_type", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "|LLL",
        const_cast<char**>(kwlist),
        &stream_id,
        &device_index,
        &device_type)) {
    }

    auto stream = c10_npu::NPUStream::unpack3(
        stream_id, device_index, static_cast<c10::DeviceType>(device_type));

    int device;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&device));
    if (device != stream.device_index()) {
      THNPModule_setDevice(stream.device_index());
    }
    c10_npu::setCurrentNPUStream(stream);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject *THNPModule_is_jit_compile_false_wrap(PyObject *self, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    pybind11::gil_scoped_release no_gil;
    static const std::string jit_compile_option_name = "jitCompile";
    auto option_value = c10_npu::option::GetOption(jit_compile_option_name);
    if (option_value.has_value() && (option_value.value() == "disable")) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_setMemoryFraction(PyObject *_unused, PyObject *args)
{
    HANDLE_TH_ERRORS
    PyObject* fraction_o = nullptr;
    PyObject* device_o = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
      THPUtils_invalidArguments(
          args,
          nullptr,
          "set_memory_fraction",
          1,
          "(double fraction, int device);");
      return nullptr;
    }
    double fraction = PyFloat_AsDouble(fraction_o);
    int64_t device = PyLong_AsLongLong(device_o);

    c10_npu::NPUCachingAllocator::setMemoryFraction(fraction, device);
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
}

PyObject* THNPModule_emptyCache(PyObject *_unused, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    c10_npu::NPUCachingAllocator::emptyCache();
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
}

PyObject* THNPModule_memoryStats(PyObject *_unused, PyObject *arg)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
    const int device = (int) THPUtils_unpackLong(arg);

    using c10_npu::NPUCachingAllocator::StatType;
    using c10_npu::NPUCachingAllocator::Stat;
    using c10_npu::NPUCachingAllocator::StatArray;
    using c10_npu::NPUCachingAllocator::DeviceStats;

    const auto statToDict = [](const Stat& stat) {
      py::dict dict;

      dict["current"] = stat.current;
      dict["peak"] = stat.peak;
      dict["allocated"] = stat.allocated;
      dict["freed"] = stat.freed;
      return dict;
    };

    const auto statArrayToDict = [=](const StatArray& statArray) {
      const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)> statTypeNames = {
          "all", "small_pool", "large_pool"
      };
      py::dict dict;
      for (size_t i = 0; i < statTypeNames.size(); ++i) {
        dict[statTypeNames[i]] = statToDict(statArray[i]);
      }
      return dict;
    };

    const DeviceStats stats = c10_npu::NPUCachingAllocator::getDeviceStats(device);

    py::dict result;
    result["num_alloc_retries"] = stats.num_alloc_retries;
    result["num_ooms"] = stats.num_ooms;
    result["max_split_size"] = stats.max_split_size;
    result["allocation"] = statArrayToDict(stats.allocation);
    result["segment"] = statArrayToDict(stats.segment);
    result["active"] = statArrayToDict(stats.active);
    result["inactive_split"] = statArrayToDict(stats.inactive_split);
    result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
    result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
    result["active_bytes"] = statArrayToDict(stats.active_bytes);
    result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
    result["oversize_allocations"] = statToDict(stats.oversize_allocations);
    result["oversize_segments"] = statToDict(stats.oversize_segments);

    return result.release().ptr();
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_resetAccumulatedMemoryStats(PyObject *_unused, PyObject *arg)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to reset_accumulated_memory_stats");
    const int device = (int) THPUtils_unpackLong(arg);
    c10_npu::NPUCachingAllocator::resetAccumulatedStats(device);
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
}

PyObject* THNPModule_resetPeakMemoryStats(PyObject *_unused, PyObject *arg)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
    const int device = (int) THPUtils_unpackLong(arg);
    c10_npu::NPUCachingAllocator::resetPeakStats(device);
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
}

PyObject* THNPModule_memorySnapshot(PyObject *_unused, PyObject *noargs)
{
    HANDLE_TH_ERRORS

    using c10_npu::NPUCachingAllocator::SegmentInfo;
    using c10_npu::NPUCachingAllocator::BlockInfo;

    const auto segmentInfoToDict = [](const SegmentInfo& segmentInfo) {
      py::dict segmentDict;
      segmentDict["device"] = segmentInfo.device;
      segmentDict["address"] = segmentInfo.address;
      segmentDict["total_size"] = segmentInfo.total_size;
      segmentDict["allocated_size"] = segmentInfo.allocated_size;
      segmentDict["active_size"] = segmentInfo.active_size;
      segmentDict["segment_type"] = (segmentInfo.is_large ? "large" : "small");

      py::list blocks;
      for (const auto& blockInfo : segmentInfo.blocks) {
        py::dict blockDict;
        blockDict["size"] = blockInfo.size;
        blockDict["state"] = (blockInfo.allocated ? "active_allocated" : (blockInfo.active ? "active_pending_free" : "inactive"));
        blocks.append(blockDict);
      }
      segmentDict["blocks"] = blocks;

      return segmentDict;
    };

    const std::vector<SegmentInfo>& snapshot = c10_npu::NPUCachingAllocator::snapshot();
    py::list result;

    for (const auto& segmentInfo : snapshot) {
      result.append(segmentInfoToDict(segmentInfo));
    }

    return result.release().ptr();
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuCachingAllocator_raw_alloc(PyObject *_unused, PyObject *args) {
    HANDLE_TH_ERRORS
    PyObject* size_o = nullptr;
    PyObject* stream_o = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
      THPUtils_invalidArguments(
          args,
          nullptr,
          "caching_allocator_alloc",
          1,
          "(ssize_t size, intptr_t stream);");
      return nullptr;
    }
    ssize_t size = PyLong_AsSsize_t(size_o);
    aclrtStream stream = static_cast<aclrtStream>(PyLong_AsVoidPtr(stream_o));
    void* mem = c10_npu::NPUCachingAllocator::raw_alloc_with_stream(size, stream);
    return PyLong_FromVoidPtr(mem);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuCachingAllocator_raw_delete(PyObject *_unused, PyObject *obj) {
    HANDLE_TH_ERRORS
    void* mem_ptr = PyLong_AsVoidPtr(obj);
    c10_npu::NPUCachingAllocator::raw_delete(mem_ptr);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getAllocatorBackend(PyObject *_unused, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    return THPUtils_packString(c10_npu::NPUCachingAllocator::name());
    END_HANDLE_TH_ERRORS
}

// We need to ensure that as long as a thread will NEVER loose the GIL as long as
// it holds the NPU mutex. Otherwise another thread might be scheduled and try to
// e.g. allocate a new tensor which will cause a deadlock. It's enough to have a
// single global, because it can be only set once (npuMutex is not recursive)
// by the thread that owns the mutex (obviously there can be only one such thread).
static PyGILState_STATE npuMutexGILState;

PyObject* THNPModule_npuLockMutex(PyObject *module, PyObject *noargs)
{
    auto mutex = c10_npu::NPUCachingAllocator::getFreeMutex();
    // This has to be a busy loop because we **absolutely need to** hold the GIL
    // or it's a recipe for a deadlock otherwise (if we let other Python threads
    // run while we have the cudaMutex, but not the GIL, they might try to e.g.
    // free a CUDA tensor and acquire the cudaMutex without giving up the GIL,
    // because it happens deep within THC).
    while (true) {
      if (mutex->try_lock()) {
          break;
      }
      {
        pybind11::gil_scoped_release no_gil;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }

    npuMutexGILState = PyGILState_Ensure();
    Py_RETURN_NONE;
}

PyObject* THNPModule_npuUnlockMutex(PyObject *module, PyObject *noargs)
{
    auto mutex = c10_npu::NPUCachingAllocator::getFreeMutex();
    PyGILState_Release(npuMutexGILState);
    mutex->unlock();
    Py_RETURN_NONE;
}

PyObject* THNPModule_initDump(PyObject* _unused, PyObject* noargs) {
    HANDLE_TH_ERRORS
    pybind11::gil_scoped_release no_gil;
    NPU_CHECK_ERROR(aclmdlInitDump());
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_setDump(PyObject* _unused, PyObject* arg) {
    HANDLE_TH_ERRORS
    if (!THPUtils_checkString(arg)) {
      THPUtils_setError("npu set dump error, cfg_file must string");
    }
    std::string cfg_file = THPUtils_unpackString(arg);
    {
      pybind11::gil_scoped_release no_gil;
      NPU_CHECK_ERROR(aclmdlSetDump(cfg_file.c_str()));
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
  }

  PyObject* THNPModule_finalizeDump(PyObject* _unused, PyObject* noargs) {
    HANDLE_TH_ERRORS
    pybind11::gil_scoped_release no_gil;
    NPU_CHECK_ERROR(aclmdlFinalizeDump());
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_setOption_wrap(PyObject* self, PyObject* arg) {
    HANDLE_TH_ERRORS

    if (!PyDict_Check(arg)) {
      throw torch::TypeError("npu option must be a dict.");
    }

    PyObject *key = nullptr;
    PyObject *value = nullptr;
    Py_ssize_t pos = 0;
    std::map<std::string, std::string> option;

    while (PyDict_Next(arg, &pos, &key, &value)) {
      if (key == nullptr || !PyUnicode_Check(key)) {
        throw torch::TypeError("option name is nullptr or is not string.");
      }

      if (value == nullptr || !PyUnicode_Check(value)) {
        throw torch::TypeError("option value is nullptr or is not string.");
      }

      const char *pKey = PyUnicode_AsUTF8(key);
      const char *pValue = PyUnicode_AsUTF8(value);
      option[pKey] = pValue;
    }
    torch_npu::utils::npu_lazy_init();
    {
      pybind11::gil_scoped_release no_gil;
      c10_npu::option::SetOption(option);
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
  }

  PyObject* THNPModule_prof_start(PyObject* self, PyObject* args) {
    HANDLE_TH_ERRORS

    PyObject *value_1 = nullptr;
    PyObject *value_2 = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &value_1, &value_2)) {
      throw torch::TypeError("prof_start npu_event type or aicore_metrics set error.");
    }
    uint64_t npu_event = static_cast<uint64_t>(THPUtils_unpackLong(value_1));
    uint64_t aicore_metrics = static_cast<uint64_t>(THPUtils_unpackLong(value_2));
    pybind11::gil_scoped_release no_gil;
    torch_npu::profiler::NpuProfiling::Instance().Start(npu_event, aicore_metrics);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_enable_e2eProfiler(PyObject* self, PyObject* args) {
    HANDLE_TH_ERRORS

    PyObject *value_1 = nullptr;
    PyObject *value_2 = nullptr;
    PyObject *value_3 = nullptr;
    PyObject *value_4 = nullptr;
    if (!PyArg_ParseTuple(args, "OOOO", &value_1, &value_2, &value_3, &value_4)) {
      throw torch::TypeError("e2eProfiler set path or option error.");
    }
    const char *dump_path = PyUnicode_AsUTF8(value_1);
    if (dump_path == nullptr) {
      throw torch::TypeError("e2eProfiler path can not be nullptr.");
    }
    uint64_t npu_event = static_cast<uint64_t>(THPUtils_unpackLong(value_2));
    uint64_t aicore_metrics = static_cast<uint64_t>(THPUtils_unpackLong(value_3));
    pybind11::gil_scoped_release no_gil;
    bool call_stack = THPUtils_unpackBool(value_4);
    torch_npu::profiler::InitE2eProfiler(dump_path, npu_event, aicore_metrics, call_stack);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_disable_e2eProfiler(PyObject* _unused, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    pybind11::gil_scoped_release no_gil;
    torch_npu::profiler::FinalizeE2eProfiler();
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_set_run_yet_variable_to_false_wrap(
    PyObject* self,
    PyObject* noargs)
{
    HANDLE_TH_ERRORS
    torch_npu::utils::npu_set_run_yet_variable_to_false();
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_get_soc_version(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    return PyLong_FromLong(static_cast<long>(c10_npu::GetSocVersion()));
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_is_support_inf_nan(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    if (c10_npu::IsSupportInfNan()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_is_bf16_supported(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    if (c10_npu::IsBF16Supported()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_enable_overflow_npu(
    PyObject* self,
    PyObject* noargs)
{
    HANDLE_TH_ERRORS
    torch_npu::utils::OverflowUtil::GetInstance()->EnableOverflowNpu();
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_check_overflow_npu(
    PyObject* self,
    PyObject* noargs)
{
    HANDLE_TH_ERRORS
    auto has_overflow = torch_npu::utils::OverflowUtil::GetInstance() ->CheckOverflowNpu();
    if (has_overflow)
    {
      Py_RETURN_TRUE;
    }
    else
    {
      Py_RETURN_FALSE;
    }
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_clear_overflow_npu(
    PyObject* self,
    PyObject* noargs)
{
    HANDLE_TH_ERRORS
    torch_npu::utils::OverflowUtil::GetInstance()->ClearOverflowNpu();
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getOption_wrap(PyObject* self, PyObject* option_type)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkString(option_type), "invalid argument to option_type,option_type must string!");
    std::string option_type_str = THPUtils_unpackString(option_type);
    auto option_key = c10_npu::option::GetOption(option_type_str);
    if (option_key.has_value()) {
      return PyBytes_FromString(option_key.value().c_str());
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_set_sync_debug_mode(PyObject* _unused, PyObject* arg)
{
    HANDLE_TH_ERRORS
    TORCH_NPU_WARN_ONCE(
        "Synchronization debug mode is a prototype feature and does not yet detect all "
        "synchronizing operations");
    TORCH_CHECK(
        THPUtils_checkLong(arg), "invalid argument to set_sync_debug_mode, debug_mode type must long");
    int64_t debug_mode = THPUtils_unpackLong(arg);
    TORCH_CHECK(
        debug_mode >= 0 && debug_mode <= 2,
        "invalid value of debug_mode, expected one of 0,1,2");
    c10_npu::SyncDebugMode level;
    switch (debug_mode) {
        case 0:
            level = c10_npu::SyncDebugMode::L_DISABLED;
            break;
        case 1:
            level = c10_npu::SyncDebugMode::L_WARN;
            break;
        case 2:
            level = c10_npu::SyncDebugMode::L_ERROR;
            break;
        default:
            level = c10_npu::SyncDebugMode::L_DISABLED;
            break;
    }
    c10_npu::warning_state().set_sync_debug_mode(level);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_get_sync_debug_mode(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    auto debug_mode = c10_npu::warning_state().get_sync_debug_mode();
    switch (debug_mode) {
        case c10_npu::SyncDebugMode::L_DISABLED:
            return THPUtils_packInt32(0);
        case c10_npu::SyncDebugMode::L_WARN:
            return THPUtils_packInt32(1);
        case c10_npu::SyncDebugMode::L_ERROR:
            return THPUtils_packInt32(2);
        default:
            return THPUtils_packInt32(-1); // can't happen
    }
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_tensor_construct_from_storage(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    static torch::PythonArgParser parser(
        {"set_storage_with_format_(Storage source)", },
        /* traceable= */ false
        );

    torch::ParsedArgs<1> parsed_args;
    auto _r = parser.parse(args, nullptr, parsed_args);

    at::ScalarType storage_scalar_type;
    bool is_typed_storage = true;
    c10::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
    return THPVariable_Wrap(at_npu::native::set_tensor_with_storage_format(storage));

    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_tensor_storage_resize(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    static torch::PythonArgParser parser(
        {"resize_storage_nbytes_(Tensor self, int64_t size)", },
        false);

    torch::ParsedArgs<2> parsed_args;
    auto _r = parser.parse(args, nullptr, parsed_args);

    at::Tensor tensor = _r.tensor(0);
    int64_t new_size = _r.toInt64(1);
    return THPVariable_Wrap(at_npu::native::npu_storage_resize(tensor, new_size));

    END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THNPModule_methods[] = {
    {"_npu_init", (PyCFunction)THNPModule_initExtension, METH_NOARGS, nullptr},
    {"_npu_set_run_yet_variable_to_false", (PyCFunction)THNPModule_set_run_yet_variable_to_false_wrap, METH_NOARGS, nullptr},
    {"_npu_synchronize", (PyCFunction)THNPModule_npuSynchronize, METH_NOARGS, nullptr},
    {"_npu_setDevice", (PyCFunction)THNPModule_setDevice_wrap, METH_O, nullptr},
    {"_npu_getDevice", (PyCFunction)THNPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_npu_getDeviceCount", (PyCFunction)THNPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
    {"_npu_canDeviceAccessPeer", (PyCFunction)THNPModule_npuCanDeviceAccessPeer_wrap, METH_VARARGS, nullptr},
    {"_npu_getDeviceUtilizationRate", (PyCFunction)THNPModule_getDeviceUtilizationRate_wrap, METH_O, nullptr},
    {"_npu_getCurrentStream", (PyCFunction)THNPModule_getCurrentStream_wrap, METH_O, nullptr},
    {"_npu_getDefaultStream", (PyCFunction)THNPModule_getDefaultStream_wrap, METH_O, nullptr},
    {"_npu_setStream", (PyCFunction)THNPModule_setStream_wrap,  METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_npu_is_jit_compile_false", (PyCFunction)THNPModule_is_jit_compile_false_wrap, METH_NOARGS, nullptr},
    {"_npu_setMemoryFraction", (PyCFunction) THNPModule_setMemoryFraction, METH_VARARGS, nullptr},
    {"_npu_emptyCache", (PyCFunction) THNPModule_emptyCache, METH_NOARGS, nullptr},
    {"_npu_memoryStats", (PyCFunction) THNPModule_memoryStats, METH_O, nullptr},
    {"_npu_resetAccumulatedMemoryStats", (PyCFunction) THNPModule_resetAccumulatedMemoryStats, METH_O, nullptr},
    {"_npu_resetPeakMemoryStats", (PyCFunction) THNPModule_resetPeakMemoryStats, METH_O,  nullptr},
    {"_npu_memorySnapshot", (PyCFunction) THNPModule_memorySnapshot, METH_NOARGS, nullptr},
    {"_npu_npuCachingAllocator_raw_alloc", (PyCFunction)THNPModule_npuCachingAllocator_raw_alloc, METH_VARARGS, nullptr},
    {"_npu_npuCachingAllocator_raw_delete", (PyCFunction)THNPModule_npuCachingAllocator_raw_delete, METH_O, nullptr},
    {"_npu_getAllocatorBackend", (PyCFunction)THNPModule_getAllocatorBackend, METH_NOARGS, nullptr},
    {"_npu_lock_mutex",   (PyCFunction)THNPModule_npuLockMutex,   METH_NOARGS,  nullptr},
    {"_npu_unlock_mutex", (PyCFunction)THNPModule_npuUnlockMutex, METH_NOARGS,  nullptr},
    {"_npu_initDump", (PyCFunction)THNPModule_initDump, METH_NOARGS, nullptr},
    {"_npu_setDump", (PyCFunction)THNPModule_setDump, METH_O, nullptr},
    {"_npu_finalizeDump", (PyCFunction)THNPModule_finalizeDump, METH_NOARGS, nullptr},
    {"_npu_setOption", (PyCFunction)THNPModule_setOption_wrap, METH_O, nullptr},
    {"_prof_start", (PyCFunction)THNPModule_prof_start, METH_VARARGS, nullptr},
    {"_enable_e2e_profiler", (PyCFunction)THNPModule_enable_e2eProfiler, METH_VARARGS, nullptr},
    {"_disable_e2e_profiler", (PyCFunction)THNPModule_disable_e2eProfiler, METH_NOARGS, nullptr},
    {"_npu_get_soc_version", (PyCFunction)THNPModule_npu_get_soc_version, METH_NOARGS, nullptr},
    {"_enable_overflow_npu", (PyCFunction)THNPModule_enable_overflow_npu, METH_NOARGS, nullptr},
    {"_npu_is_support_inf_nan", (PyCFunction)THNPModule_npu_is_support_inf_nan, METH_NOARGS, nullptr},
    {"_npu_is_bf16_supported", (PyCFunction)THNPModule_npu_is_bf16_supported, METH_NOARGS, nullptr},
    {"_check_overflow_npu", (PyCFunction)THNPModule_check_overflow_npu, METH_NOARGS, nullptr},
    {"_clear_overflow_npu", (PyCFunction)THNPModule_clear_overflow_npu, METH_NOARGS, nullptr},
    {"_npu_getOption", (PyCFunction)THNPModule_getOption_wrap, METH_O, nullptr},
    {"_npu_set_sync_debug_mode", (PyCFunction)THNPModule_npu_set_sync_debug_mode, METH_O, nullptr},
    {"_npu_get_sync_debug_mode", (PyCFunction)THNPModule_npu_get_sync_debug_mode, METH_NOARGS, nullptr},
    {"_tensor_construct_from_storage", (PyCFunction)THNPModule_tensor_construct_from_storage, METH_VARARGS, nullptr},
    {"_tensor_storage_resize", (PyCFunction)THNPModule_tensor_storage_resize, METH_VARARGS, nullptr},
    {nullptr}};

TORCH_NPU_API PyMethodDef* THNPModule_get_methods()
{
    return THNPModule_methods;
}
