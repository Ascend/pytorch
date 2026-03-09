#include <chrono>
#include <sstream>
#include <thread>
#include <future>
#include <unordered_map>

#include <ATen/ATen.h>
#include <ATen/CachedTensorUtils.h>
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
#include <torch/csrc/profiler/python/combined_traceback.h>

#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/SetNpu.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUQueue.h"
#include "torch_npu/csrc/core/npu/NPUAffinityController.h"
#include "torch_npu/csrc/core/npu/NPUPeerToPeerAccess.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/OverflowUtils.h"
#include "torch_npu/csrc/npu/Module.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/npu/DataParallelComm.h"
#include "torch_npu/csrc/npu/NPUPluggableAllocator.h"
#include "torch_npu/csrc/npu/Stream.h"
#include "torch_npu/csrc/npu/Stress_detect.h"
#include "torch_npu/csrc/aten/python_functions.h"
#include "torch_npu/csrc/utils/LazyInit.h"
#include "third_party/acl/inc/acl/acl.h"
#include "torch_npu/csrc/npu/memory_snapshot.h"
#include "torch_npu/csrc/core/npu/interface/OpInterface.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/ipc/NPUIPCTypes.h"
#include "op_plugin/utils/custom_functions/opapi/FFTCommonOpApi.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/profiler/combined_traceback.h"
#include "torch_npu/csrc/profiler/python/combined_traceback.h"
#include "torch_npu/csrc/framework/interface/AclInterface.h"
#include "third_party/fmt/include/fmt/format.h"

std::shared_ptr<npu_logging::Logger> loggerRecovery = npu_logging::logging().getLogger("torch_npu.recovery");

struct NPUDeviceProp {
    std::string name;
    size_t totalGlobalMem = 0;
    int64_t cube_core_num = 0;
    int64_t vector_core_num = 0;
    int64_t L2_cache_size = 0;
    std::optional<int> major;
    std::optional<int> minor;
    std::optional<int> is_multi_gpu_board;
    std::optional<int> is_integrated;
    std::optional<int> multi_processor_count;
    std::optional<int> max_threads_per_multi_processor;
    std::optional<int> warp_size;
    std::optional<int> regs_per_multiprocessor;
    std::optional<std::string> gcnArchName;
    aclrtUuid uuid = {0}; // Initialize to 0
};

struct NPUDeviceMem {
    size_t totalGlobalMem = 0;
    size_t freeMem = 0;
};

namespace {
c10::DeviceIndex num_npus = -1;
std::deque<c10::once_flag> device_flags;
std::vector<NPUDeviceProp> device_properties;

void initNPUContextVectors()
{
    static bool init_flag [[maybe_unused]] = []() {
        num_npus = c10_npu::device_count();
        device_flags.resize(num_npus);
        device_properties.resize(num_npus);
        return true;
    }();
}
} // anonymous namespace

std::string uuid_to_string(const char* uuid_bytes)
{
    // UUIDs store as char[16].
    // For string representation, the code here expands this to
    // 8-4-4-4-12 hex format, so each byte becomes 2 hex characters.
    return fmt::format(
        "{:02x}{:02x}{:02x}{:02x}-"
        "{:02x}{:02x}-"
        "{:02x}{:02x}-"
        "{:02x}{:02x}-"
        "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        (uint8_t)uuid_bytes[0], // 0
        (uint8_t)uuid_bytes[1], // 1
        (uint8_t)uuid_bytes[2], // 2
        (uint8_t)uuid_bytes[3], // 3
        (uint8_t)uuid_bytes[4], // 4
        (uint8_t)uuid_bytes[5], // 5
        (uint8_t)uuid_bytes[6], // 6
        (uint8_t)uuid_bytes[7], // 7
        (uint8_t)uuid_bytes[8], // 8
        (uint8_t)uuid_bytes[9], // 9
        (uint8_t)uuid_bytes[10], // 10
        (uint8_t)uuid_bytes[11], // 11
        (uint8_t)uuid_bytes[12], // 12
        (uint8_t)uuid_bytes[13], // 13
        (uint8_t)uuid_bytes[14], // 14
        (uint8_t)uuid_bytes[15]); // 15
}

void RegisterNPUDeviceProperties(PyObject* module)
{
    auto m = py::handle(module).cast<py::module>();
    py::class_<aclrtUuid>(m, "_CUuuid")
        .def_property_readonly("bytes", [](const aclrtUuid& uuid) {
            return std::vector<uint8_t>(uuid.bytes, uuid.bytes + 16); // 16
        })
        .def("__str__", [](const aclrtUuid& uuid) {
            return uuid_to_string(uuid.bytes);
        });
    py::class_<NPUDeviceProp>(m, "_NPUDeviceProperties")
        .def_readonly("name", &NPUDeviceProp::name)
        .def_readonly("total_memory", &NPUDeviceProp::totalGlobalMem)
        .def_readonly("cube_core_num", &NPUDeviceProp::cube_core_num)
        .def_readonly("vector_core_num", &NPUDeviceProp::vector_core_num)
        .def_readonly("L2_cache_size", &NPUDeviceProp::L2_cache_size)
        .def_readonly("major", &NPUDeviceProp::major)
        .def_readonly("minor", &NPUDeviceProp::minor)
        .def_readonly("is_multi_gpu_board", &NPUDeviceProp::is_multi_gpu_board)
        .def_readonly("is_integrated", &NPUDeviceProp::is_integrated)
        .def_readonly("multi_processor_count", &NPUDeviceProp::multi_processor_count)
        .def_readonly("max_threads_per_multi_processor", &NPUDeviceProp::max_threads_per_multi_processor)
        .def_readonly("warp_size", &NPUDeviceProp::warp_size)
        .def_readonly("regs_per_multiprocessor", &NPUDeviceProp::regs_per_multiprocessor)
        .def_readonly("gcnArchName", &NPUDeviceProp::gcnArchName)
        .def_readonly("uuid", &NPUDeviceProp::uuid)
        .def("__repr__", [](const NPUDeviceProp &prop) {
            std::ostringstream stream;
            stream << "_NPUDeviceProperties(name='" << prop.name << "', total_memory="
                << prop.totalGlobalMem / (CHANGE_UNIT_SIZE * CHANGE_UNIT_SIZE) << "MB, cube_core_num="
                << prop.cube_core_num << ", vector_core_num=" << prop.vector_core_num
                << ", uuid=" << uuid_to_string(prop.uuid.bytes) << ", L2_cache_size="
                << prop.L2_cache_size / (CHANGE_UNIT_SIZE * CHANGE_UNIT_SIZE) << "MB)";
            return stream.str();
        });
    m.def(
        "_npu_record_memory_history",
        static_cast<void (*)(c10::optional<std::string>,
                             c10::optional<std::string>, std::string, size_t)>(
            torch_npu::_record_memory_history));

    m.def("_npu_isHistoryEnabled",
          []() { return c10_npu::NPUCachingAllocator::isHistoryEnabled(); });
}

std::string GetDeviceName()
{
    const char* device_name = c10_npu::acl::AclrtGetSocName();
    if (device_name == nullptr) {
        ASCEND_LOGE("NPU get device name fail.");
        return "";
    }
    return std::string(device_name);
}

void initDeviceProperty(int64_t deviceid)
{
    const char* device_name;
    size_t device_free;
    size_t device_total;
    int64_t cube_core_num;
    int64_t vector_core_num;
    int64_t L2_cache_size;
    aclrtUuid uuid;

    device_name = c10_npu::acl::AclrtGetSocName();
    if (device_name == nullptr) {
        device_properties[deviceid].name = " ";
        ASCEND_LOGE("NPU get device name fail.");
    } else {
        device_properties[deviceid].name = std::string(device_name);
    }
    NPU_CHECK_ERROR_WITHOUT_UCE(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
    device_properties[deviceid].totalGlobalMem = device_total;

    NPU_CHECK_ERROR_WITHOUT_UCE(aclGetDeviceCapability(deviceid, ACL_DEVICE_INFO_AI_CORE_NUM, &cube_core_num));
    device_properties[deviceid].cube_core_num = cube_core_num;

    NPU_CHECK_ERROR_WITHOUT_UCE(aclGetDeviceCapability(deviceid, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &vector_core_num));
    device_properties[deviceid].vector_core_num = vector_core_num;

    NPU_CHECK_ERROR_WITHOUT_UCE(aclGetDeviceCapability(deviceid, ACL_DEVICE_INFO_L2_SIZE, &L2_cache_size));
    device_properties[deviceid].L2_cache_size = L2_cache_size;

    if (c10_npu::acl::IsExistDeviceGetUuid()) {
        aclError err = c10_npu::acl::AclrtDeviceGetUuid(deviceid, &uuid);
        if (err == ACL_ERROR_NONE) {
            device_properties[deviceid].uuid = uuid;
        } else if (err != ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
            NPU_CHECK_ERROR_WITHOUT_UCE(err);
        }
    }
}

NPUDeviceProp* GetDeviceProperties(int64_t deviceid)
{
    initNPUContextVectors();
    if (deviceid == -1) {
        deviceid = c10_npu::current_device();
    }
    TORCH_CHECK(deviceid >= 0 && deviceid < num_npus,
        "device=", static_cast<int>(deviceid), ", num_npus=", static_cast<int>(num_npus),
        PTA_ERROR(ErrCode::PARAM));
    c10::call_once(device_flags[deviceid], initDeviceProperty, deviceid);
    return &device_properties[deviceid];
}

void BindGetDeviceProperties(PyObject* module)
{
    auto m = py::handle(module).cast<py::module>();
    m.def("_npu_getDeviceProperties", [](int deviceid) -> NPUDeviceProp* {
      return GetDeviceProperties(deviceid);
    }, py::return_value_policy::reference);
    m.def("_npu_getDeviceName", []() -> std::string {
      return GetDeviceName();
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
    NPU_CHECK_ERROR_WITHOUT_UCE(aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total));
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

// We choose to ignore certain blocks that are currently allocated
// when we set the pool to its checkpoint. For those blocks, we need
// to swap out the deleter function of their corresponding blocks
// so that a deallocation is not triggered when they die.
void removeStorageDeleterFns(
    const std::vector<c10::StorageImpl*>& stale_live_storages,
    std::unordered_set<void*> definitely_stale_pointers)
{
    for (c10::StorageImpl* stale_storage : stale_live_storages) {
        auto ptr = stale_storage->data_ptr().get();
        auto allocated_pointer = definitely_stale_pointers.find(ptr);
        TORCH_CHECK(allocated_pointer != definitely_stale_pointers.end());
        auto t = c10_npu::NPUCachingAllocator::get();
        bool succeeded = stale_storage->mutable_data_ptr().compare_exchange_deleter(
            t->raw_deleter(), &c10::detail::deleteNothing);

        TORCH_CHECK(succeeded,
            "Unexpected deleter function on storage, could not swap function", PTA_ERROR(ErrCode::PARAM));
    }
}

void addStorageDeleterFns(
    std::vector<c10::StorageImpl*>& storages_to_add_deleters_to,
    c10_npu::NPUCachingAllocator::CheckpointDelta& delta)
{
    std::unordered_map<void*, c10::StorageImpl*> storages;
    for (auto& storage : storages_to_add_deleters_to) {
        storages[storage->data_ptr().get()] = storage;
    }

    for (auto& data_ptr : delta.dataptrs_allocd) {
        auto storage_pair = storages.find(data_ptr.get());
        if (storage_pair != storages.end()) {
            auto ctx = storage_pair->second->data_ptr().get_context();
            TORCH_CHECK(ctx == nullptr, " Not expecting deleter function", PTA_ERROR(ErrCode::PARAM));
            storage_pair->second->set_data_ptr_noswap(std::move(data_ptr));
        } else {
            data_ptr.release_context();
        }
    }
}

void RegisterNpuPluggableAllocator(PyObject* module)
{
    auto m = py::handle(module).cast<py::module>();

    py::class_<
        c10_npu::NPUCachingAllocator::NPUAllocator,
        std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator>>(
        m, "_npu_NPUAllocator");
    py::class_<
        c10_npu::NPUCachingAllocator::AllocatorState,
        std::shared_ptr<c10_npu::NPUCachingAllocator::AllocatorState>>(
        m, "_npu_NPUAllocator_AllocatorState");

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
            using FuncType = void(void*, c10_npu::NPUStream);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_record_stream_fn(func);
        })
        .def(
        "set_erase_stream_fn",
        [](torch::npu::NPUPluggableAllocator::NPUPluggableAllocator& self,
            uint64_t func_ptr) {
            using FuncType = void(void*, c10_npu::NPUStream);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_erase_stream_fn(func);
        })
        .def(
        "set_get_device_stats_fn",
        [](torch::npu::NPUPluggableAllocator::NPUPluggableAllocator& self,
            uint64_t func_ptr) {
            using FuncType=c10_npu::NPUCachingAllocator::DeviceStats(int);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_get_device_stats_fn(func);
        })
        .def(
        "set_reset_peak_status_fn",
        [](torch::npu::NPUPluggableAllocator::NPUPluggableAllocator& self,
            uint64_t func_ptr) {
            using FuncType = void(int);
            std::function<FuncType> func =
                reinterpret_cast<FuncType*>(func_ptr);
            self.set_reset_peak_status_fn(func);
        });

    m.def(
        "_npu_customAllocator",
        [](uint64_t malloc_ptr, uint64_t free_ptr) {
            using MallocFuncType = void*(size_t, int, aclrtStream);
            using FreeFuncType = void(void*, size_t, int, aclrtStream);
            std::function<MallocFuncType> malloc_fn =
                reinterpret_cast<MallocFuncType*>(malloc_ptr);
            std::function<FreeFuncType> free_fn =
                reinterpret_cast<FreeFuncType*>(free_ptr);
            return torch::npu::NPUPluggableAllocator::createCustomAllocator(
                malloc_fn, free_fn);
        });
    m.def(
        "_npu_beginAllocateCurrentStreamToPool",
        [](c10::DeviceIndex device, c10_npu::MempoolId_t mempool_id) {
            auto stream = c10_npu::getCurrentNPUStream(device);
            TORCH_CHECK(stream, "Expected stream capture to be under way");
            c10_npu::NPUCachingAllocator::beginAllocateToPool(
                device, mempool_id, [stream](aclrtStream target) {
                return target == stream;
            });
        });
    m.def(
        "_npu_beginAllocateToPool",
        [](c10::DeviceIndex device, c10_npu::MempoolId_t mempool_id) {
            c10_npu::NPUCachingAllocator::beginAllocateToPool(
                device, mempool_id, [](aclrtStream) { return true; });
        });
    m.def(
        "_npu_endAllocateCurrentStreamToPool",
        [](c10::DeviceIndex device, c10_npu::MempoolId_t mempool_id) {
            c10_npu::NPUCachingAllocator::endAllocateToPool(device, mempool_id);
        });
    m.def(
        "_npu_releasePool",
        [](c10::DeviceIndex device, c10_npu::MempoolId_t mempool_id) {
            c10_npu::NPUCachingAllocator::releasePool(device, mempool_id);
        });
    m.def(
        "_tensors_data_ptrs_at_indices_equal",
        [](py::list& tensors, py::list& data_ptrs, py::list& indices) {
            for (size_t i = 0, end = indices.size(); i < end; ++i) {
            auto index = indices[i].cast<int64_t>();
            auto t = tensors[index].cast<at::Tensor>();
            auto data_ptr = data_ptrs[index].cast<int64_t>();
            if (reinterpret_cast<int64_t>(t.data_ptr()) != data_ptr) {
                return false;
            }
            }
            return true;
        });
    m.def(
        "_storage_Use_Count",
        [](size_t storage_impl_ptr) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
            return c10::raw::weak_intrusive_ptr::use_count(storage_impl);
        });
    m.def(
        "_npu_getCheckpointState",
        [](c10::DeviceIndex device, c10_npu::MempoolId_t id) {
            return c10_npu::NPUCachingAllocator::getCheckpointState(device, id);
        });
    m.def(
        "_npu_setCheckpointPoolState",
        [](c10::DeviceIndex device,
            std::shared_ptr<c10_npu::NPUCachingAllocator::AllocatorState> pps,
            const std::vector<size_t>& stale_storages_ptr,
            const std::vector<size_t>& storages_to_add_deleters_to_ptr = {}) {
            std::unordered_set<c10::StorageImpl*> ptr_set;
            // iterate on std::vector for determinism
            std::vector<c10::StorageImpl*> ptrs;
            for (size_t ptr_int : stale_storages_ptr) {
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                c10::StorageImpl* ptr = (c10::StorageImpl*)ptr_int;
                if (!ptr_set.count(ptr)) {
                    ptrs.push_back(ptr);
                    ptr_set.insert(ptr);
                }
            }
            auto delta = c10_npu::NPUCachingAllocator::setCheckpointPoolState(device, std::move(pps));
            auto& freed_pointers = delta.ptrs_freed;
    
            std::unordered_set<void*> allocd_set;
            for (auto& data_ptr : delta.dataptrs_allocd) {
                allocd_set.insert(data_ptr.get());
            }
            std::unordered_set<void*> freed_pointer_set;
            size_t definite_freed_count = 0;
            for (void* ptr : freed_pointers) {
                if (!allocd_set.count(ptr)) {
                    definite_freed_count += 1;
                }
                freed_pointer_set.insert((ptr));
            }
            // that block has already been freed,
            // so even those this will error, so too will the allocator
            // when the corresponding tensor dies because there is no
            // live tensor corresponding to it
            TORCH_CHECK(
                ptr_set.size() >= definite_freed_count,
                "Any stale tensors which are being manually freed"
                " must be passed to set checkpoint", PTA_ERROR(ErrCode::PARAM));
    
            removeStorageDeleterFns(ptrs, freed_pointer_set);
            std::vector<c10::StorageImpl*> storages_to_add_deleters_to;
            storages_to_add_deleters_to.reserve(storages_to_add_deleters_to_ptr.size());
            for (size_t ptr_int : storages_to_add_deleters_to_ptr) {
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                storages_to_add_deleters_to.push_back((c10::StorageImpl*)ptr_int);
            }
    
            addStorageDeleterFns(storages_to_add_deleters_to, delta);
            });
    m.def(
        "_free_And_Remove_DeleterFn",
        [](size_t storage_impl_ptr) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
            auto alloc = c10_npu::NPUCachingAllocator::get();
            auto data_ptr = storage_impl->data_ptr().get();
            bool succeeded = storage_impl->mutable_data_ptr().compare_exchange_deleter(
                alloc->raw_deleter(), c10::detail::deleteNothing);
            TORCH_CHECK(succeeded, "Expected standard deleter", PTA_ERROR(ErrCode::PARAM));
            c10_npu::NPUCachingAllocator::raw_delete(data_ptr);
        });
    m.def(
        "_has_Standard_Deleter",
        [](size_t storage_impl_ptr) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
            auto alloc = c10_npu::NPUCachingAllocator::get();
            return (storage_impl->data_ptr().get_deleter() == alloc->raw_deleter());
        });
    m.def(
        "_add_cached_tensor",
        [](const at::Tensor& t) {
            at::caching::add_cached_tensor(t);
        });
    m.def(
        "_remove_cached_tensor",
        [](const at::Tensor& t) {
            at::caching::remove_cached_tensor(t);
        });
    m.def(
        "_construct_NPU_Tensor_From_Storage_And_Metadata",
        [](py::dict& metadata, c10::Storage s) {
            auto dtype_arg = metadata["dtype"].ptr();
            auto meta = c10::scalarTypeToTypeMeta(torch::toScalarType(dtype_arg));

            constexpr c10::DispatchKeySet npu_dks(c10::DispatchKey::PrivateUse1);
            at::Tensor tensor = at::detail::make_tensor_base<c10::TensorImpl>(
                std::move(s), npu_dks, meta);
            if (metadata.contains("npu_format")) {
                at_npu::native::StorageDescHelper::SetDesc(
                    tensor,
                    metadata["size"].cast<std::vector<int64_t>>(),
                    metadata["stride"].cast<std::vector<int64_t>>(),
                    static_cast<aclFormat>(metadata["npu_format"].cast<int64_t>()));
            } else {
                at_npu::native::StorageDescHelper::SetDesc(
                    tensor,
                    metadata["size"].cast<std::vector<int64_t>>(),
                    metadata["stride"].cast<std::vector<int64_t>>());
            }
            tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
                metadata["size"].cast<std::vector<int64_t>>(),
                metadata["stride"].cast<std::vector<int64_t>>());
            tensor.unsafeGetTensorImpl()->set_storage_offset(
                metadata["storage_offset"].cast<int64_t>());
            return tensor;
        });
    m.def(
        "_npu_checkPoolLiveAllocations",
        [](c10::DeviceIndex device, c10_npu::MempoolId_t mempool_id,
            const py::set& expected_live_allocations) {
            std::unordered_set<void*> allocations;
            allocations.reserve(expected_live_allocations.size());
            for (auto& elem : expected_live_allocations) {
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                allocations.insert(reinterpret_cast<void*>(py::cast<size_t>(elem)));
            }
            return c10_npu::NPUCachingAllocator::checkPoolLiveAllocations(device, mempool_id, allocations);
        });
    m.def(
        "_set_cached_tensors_enabled",
        [](bool enabled) {
            at::caching::set_cached_tensors_enabled(enabled);
        });
    m.def(
        "_construct_storage_from_data_pointer",
        [](int64_t data_ptr, c10::Device device, size_t size_bytes) {
            c10::intrusive_ptr<c10::StorageImpl> storage_impl = torch_npu::make_npu_storage_impl(
                c10::StorageImpl::use_byte_size_t(),
                size_bytes,
                at::DataPtr(reinterpret_cast<void*>(data_ptr), device),
                nullptr,
                false);
            return c10::Storage(storage_impl);
        });
    m.def(
        "_weak_ref_tensor",
        [](const at::Tensor& t) {
            void* data_ptr = t.data_ptr();
            std::vector<int64_t> sizes = t.sizes().vec();
            std::vector<int64_t> strides = t.strides().vec();
            auto options = t.options();
            auto new_tensor = at_npu::native::from_blob(data_ptr, sizes, strides, options);

            auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(t)->npu_desc_;
            torch_npu::NPUBridge::GetNpuStorageImpl(new_tensor)->npu_desc_ = dst_desc;
            return new_tensor;
        });
    m.def(
        "_set_storage_access_error_msg",
        [](const at::Tensor& t, std::string s) {
            t.unsafeGetTensorImpl()->release_storage_and_set_meta_custom_data_ptr_error_msg_(s);
        });
    m.def(
        "_set_storage_data_ptr_access_error_msg",
        [](size_t storage_impl_ptr, std::string s) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
            storage_impl->release_data_and_set_meta_custom_data_ptr_error_msg_(s);
        });
    m.def(
        "_tensors_data_ptrs_at_indices_equal",
        [](py::list& tensors, py::list& data_ptrs, py::list& indices) {
            for (auto index : indices) {
                auto t = tensors[index].cast<at::Tensor>();
                auto data_ptr = data_ptrs[index].cast<int64_t>();
                if (reinterpret_cast<int64_t>(t.data_ptr()) != data_ptr) {
                return false;
                }
            }
            return true;
        });
}

static PyObject* THNPModule_initExtension(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    {
        pybind11::gil_scoped_release no_gil;
        c10_npu::NpuSysCtrl::SysStatus status =
            c10_npu::NpuSysCtrl::GetInstance().Initialize();
        if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
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
    at_npu::autograd::generated::initialize_autogenerated_functions(m);
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
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::SetDevice(device));
}

PyObject* THNPModule_setDevice_wrap(PyObject* self, PyObject* arg)
{
    HANDLE_TH_ERRORS
    int device = THPUtils_unpackLong(arg);
    torch_npu::utils::npu_lazy_init();
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::NpuSysCtrl::GetInstance().ExchangeDevice(device));

    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_stopDevice_wrap(PyObject* self, PyObject* arg)
{
    HANDLE_TH_ERRORS
    int device = THPUtils_unpackLong(arg);
    setDefaultStreamsStatus(device, c10_npu::RepoStatus::STOP_EXIT);
    loggerRecovery->info("NPU stop device start, device is %d.", device);
    int ret = c10_npu::acl::AclrtDeviceTaskAbort(device);
    loggerRecovery->info("NPU stop device end, device is %d, ret is %d.", device, ret);
    if (ret == 0) {
        return PyLong_FromLong(0);
    } else {
        return PyLong_FromLong(1);
    }
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_check_uce_in_memory_wrap(PyObject* self, PyObject* arg)
{
    HANDLE_TH_ERRORS
    int device = THPUtils_unpackLong(arg);
    loggerRecovery->info("NPU check_uce_in_memory start, device is %d.", device);
    auto memUceInfo_ = c10_npu::get_mem_uce_info();
    if (memUceInfo_.is_hbm_ecc_error) {
        // HBM ECC error always return 3.
        return PyLong_FromLong(3);
    }
    if (memUceInfo_.retSize == 0) {
        // UCE error size is 0, return 0.
        memUceInfo_.mem_type = 0;
        loggerRecovery->info("NPU check_uce_in_memory end, device is %d, mem_type is 0.", device);
        return PyLong_FromLong(0);
    }
    if (!c10_npu::NPUCachingAllocator::checkUceInMemPool(device)) {
        // UCE error memory is not in PTA memory pool, return 1, can not recover from UCE error.
        memUceInfo_.mem_type = 1;
        loggerRecovery->info("NPU check_uce_in_memory end, device is %d, mem_type is 1.", device);
        return PyLong_FromLong(1);
    } else {
        c10_npu::NPUCachingAllocator::emptyCache(false);
        if (!c10_npu::NPUCachingAllocator::checkUceInMemPool(device)) {
            // UCE error memory is temporary memory in PTA memory pool, return 2, perform step-level re-execution.
            memUceInfo_.mem_type = 2;
            loggerRecovery->info("NPU check_uce_in_memory end, device is %d, mem_type is 2.", device);
            return PyLong_FromLong(2);
        } else {
            // UCE error memory is persistent memory in PTA memory pool, return 3, load the checkpoint (ckpt) from healthy device.
            memUceInfo_.mem_type = 3;
            loggerRecovery->info("NPU check_uce_in_memory end, device is %d, mem_type is 3.", device);
            return PyLong_FromLong(3);
        }
    }

    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_get_uce_addr_wrap(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    auto memUceInfo_ = c10_npu::get_mem_uce_info();

    py::list result;
    for (size_t i = 0; i < memUceInfo_.retSize; ++i) {
        py::dict data;
        data["ptr"] = reinterpret_cast<int64_t>(memUceInfo_.info[i].addr);
        data["size"] = memUceInfo_.info[i].len;
        result.append(data);
    }
    return result.release().ptr();
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_restart_device_wrap(PyObject* self, PyObject* arg)
{
    HANDLE_TH_ERRORS
    int device = THPUtils_unpackLong(arg);
    loggerRecovery->info("NPU restart device start, device is %d.", device);
    auto memUceInfo_ = c10_npu::get_mem_uce_info();
    if (memUceInfo_.retSize > 0) {
        loggerRecovery->info("exec AclrtMemUceRepair start, device is %d, retSize is %d.", memUceInfo_.device, memUceInfo_.retSize);
        NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtMemUceRepair(memUceInfo_.device, memUceInfo_.info, memUceInfo_.retSize));
        loggerRecovery->info("exec AclrtMemUceRepair end, device is %d, retSize is %d.", memUceInfo_.device, memUceInfo_.retSize);
    }

    c10_npu::clear_mem_uce_info();
    setDefaultStreamsStatus(device, c10_npu::RepoStatus::INIT);
    c10_npu::NPUCachingAllocator::cleanEvent();
    loggerRecovery->info("NPU restart device end, device is %d.", device);

    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDevice_wrap(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    int device;
    torch_npu::utils::npu_lazy_init();
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDevice(&device));
    return PyLong_FromLong(device);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDeviceWithoutSet_wrap(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    int device;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDeviceWithoutSet(&device));
    return PyLong_FromLong(device);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_maybeExchangeDevice_wrap(PyObject* self, PyObject* arg)
{
    HANDLE_TH_ERRORS
    int64_t device = THPUtils_unpackLong(arg);
    int current_device = c10_npu::MaybeExchangeDevice(device);
    return PyLong_FromLong(current_device);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_stressDetect_wrap(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    PyObject* value1 = nullptr;
    PyObject* value2 = nullptr;

    if (!PyArg_ParseTuple(args, "OO",  &value1,  &value2)) {
        ASCEND_LOGE("Stress detect failed, argument is invalid.");
        return PyLong_FromLong(1);
    }
    int mode = THPUtils_unpackLong(value1);
    int64_t comm = THPUtils_unpackLong(value2);

    torch_npu::utils::npu_lazy_init();

    int deviceId;
    aclError err = c10_npu::GetDevice(&deviceId);
    if (err != ACL_ERROR_NONE) {
        ASCEND_LOGE("Stress detect failed, error happened in GetDevice, err is %d.", err);
        return PyLong_FromLong(1);
    }

    int ret = StressDetector::perform_stress_detect(deviceId, mode, comm);
    return PyLong_FromLong(ret);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    return PyLong_FromLong(c10_npu::device_count());
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getLocalDevice_wrap(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    return PyLong_FromLong(c10_npu::GetLocalDevice());
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuCanDeviceAccessPeer_wrap(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    PyObject *value_1 = nullptr;
    PyObject *value_2 = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &value_1, &value_2)) {
        throw torch::TypeError("Pybind failed to parse parameters." +
                               PTA_ERROR(ErrCode::TYPE));
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
    TORCH_CHECK(THPUtils_checkLong(device_index), "invalid argument to getDeviceUtilizationRate",
                PTA_ERROR(ErrCode::VALUE));
    int32_t device = static_cast<int32_t>(THPUtils_unpackUInt32(device_index));
    aclrtUtilizationInfo util_info;
    util_info.cubeUtilization = 0;
    util_info.vectorUtilization = 0;
    util_info.utilizationExtend = nullptr;
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtGetDeviceUtilizationRate(device, &util_info));
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
    TORCH_CHECK(util_rate <= 100 && util_rate >= 0, "invalid result to util_rate", PTA_ERROR(ErrCode::VALUE));
    return PyLong_FromLong(util_rate);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getCurrentStream_wrap(
    PyObject * /* unused */, PyObject *device_index)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream", PTA_ERROR(ErrCode::PARAM));
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

PyObject* THNPModule_getCurrentStream_raw(
    PyObject* /* unused */, PyObject* device_index)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(
        THPUtils_checkLong(device_index), "invalid argument to getCurrentStream", PTA_ERROR(ErrCode::PARAM));
    int64_t device = THPUtils_unpackLong(device_index);
    return PyLong_FromVoidPtr(
        c10_npu::getCurrentNPUStream(device).stream());
    END_HANDLE_TH_ERRORS
}

// Note: The torch_npu._C._npu_getCurrentRawStreamNoWait(device) interface does NOT clear the task queue.
// If tasks are dispatched using both the returned aclrtStream and torch_npu's task queue,
// it may cause ordering issues due to lack of synchronization between the two dispatch paths.
// Users must ensure to use only one of these dispatch methods exclusively.
// If mixed usage is unavoidable, ensure there are no data dependencies between tasks
// and that performance is not sensitive to potential execution reordering.
PyObject* THNPModule_getCurrentRawStreamNoWait_wrap(
    PyObject* /* unused */, PyObject* device_index)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(
        THPUtils_checkLong(device_index), "invalid argument to getCurrentStream", PTA_ERROR(ErrCode::PARAM));
    int64_t device = THPUtils_unpackLong(device_index);
    return PyLong_FromVoidPtr(
        c10_npu::getCurrentNPUStreamNoWait(device));
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_getDefaultStream_wrap(PyObject *self /* unused */, PyObject *device_index)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(device_index), "invalid argument to getDefaultStream", PTA_ERROR(ErrCode::PARAM));
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
    constexpr const char* kwlist[] = {
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
    NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::GetDevice(&device));
    if (device != stream.device_index()) {
        THNPModule_setDevice(stream.device_index());
    }
    c10_npu::setCurrentNPUStream(stream);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_eraseStream_wrap(PyObject* self, PyObject* args, PyObject* kwargs)
{
    HANDLE_TH_ERRORS
    PyObject *tensor_obj = nullptr;
    int64_t stream_id = 0;
    int64_t device_index = 0;
    int64_t device_type = 0;

    constexpr const char* kwlist[] = {
        "tensor", "stream_id", "device_index", "device_type", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "OLLL",
        const_cast<char**>(kwlist),
        &tensor_obj,
        &stream_id,
        &device_index,
        &device_type)) {
    }

    if (!THPVariable_Check(tensor_obj)) {
        TORCH_CHECK(false, "tensor is not torch.Tensor.", PTA_ERROR(ErrCode::TYPE));
    }

    // 获取 at::Tensor
    at::Tensor tensor = THPVariable_Unpack(tensor_obj);
    auto stream = c10_npu::NPUStream::unpack3(
        stream_id, device_index, static_cast<c10::DeviceType>(device_type));
    c10_npu::NPUCachingAllocator::eraseStream(tensor.storage().data_ptr(), stream);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_isCurrentStreamCapturing_wrap(
    PyObject* self,
    PyObject* noargs)
{
    HANDLE_TH_ERRORS
    // If there's no npu context, c10_npu::currentStreamCaptureStatus returns
    // CaptureStatus::None without initializing a context.
    if (c10_npu::currentStreamCaptureStatus() == c10_npu::CaptureStatus::None) {
        Py_RETURN_FALSE;
    } else {
        Py_RETURN_TRUE;
    }
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
        static const std::string jit_compile_init_option_name = "jitCompileInit";
        auto init_option_value = c10_npu::option::GetOption(jit_compile_init_option_name);
        if (init_option_value.has_value() && (init_option_value.value() == "disable")) {
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }
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

PyObject* THNPModule_npu_hostEmptyCache(PyObject *_unused, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    at_npu::native::CachingHostAllocator_emptyCache();
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
}

PyObject* THNPModule_npu_ipc_collect(PyObject *_unused, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    torch_npu::ipc::NpuIPCCollect();
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
}

PyObject* THNPModule_emptyVirtAddrCache(PyObject *_unused, PyObject *noargs)
{
    HANDLE_TH_ERRORS
    c10_npu::NPUCachingAllocator::emptyVirtAddrCache();
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
}

PyObject* THNPModule_memoryStats(PyObject *_unused, PyObject *arg)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_allocated", PTA_ERROR(ErrCode::PARAM));
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
    result["requested_bytes"] = statArrayToDict(stats.requested_bytes);
    result["oversize_allocations"] = statToDict(stats.oversize_allocations);
    result["oversize_segments"] = statToDict(stats.oversize_segments);

    return result.release().ptr();
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_resetAccumulatedMemoryStats(PyObject *_unused, PyObject *arg)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to reset_accumulated_memory_stats", PTA_ERROR(ErrCode::PARAM));
    const int device = (int) THPUtils_unpackLong(arg);
    c10_npu::NPUCachingAllocator::resetAccumulatedStats(device);
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
}

PyObject* THNPModule_resetPeakMemoryStats(PyObject *_unused, PyObject *arg)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats", PTA_ERROR(ErrCode::PARAM));
    const int device = (int) THPUtils_unpackLong(arg);
    c10_npu::NPUCachingAllocator::resetPeakStats(device);
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
}

#if defined(__x86_64__)
    using CapturedTraceback = torch::CapturedTraceback;
#elif defined(__aarch64__)
    using CapturedTraceback = torch_npu::CapturedTraceback;
#endif

CapturedTraceback* getFromContext(const std::shared_ptr<c10::GatheredContext>& x)
{
    if (CapturedTraceback* sc = dynamic_cast<CapturedTraceback*>(x.get())) {
        return sc;
    }
    TORCH_CHECK(false, "attempting to gather stack context from the wrong StackContext type.", OPS_ERROR(ErrCode::NOT_FOUND));
}

PyObject* THNPModule_memorySnapshot(PyObject* _unused, PyObject* noargs)
{
    HANDLE_TH_ERRORS

    using c10_npu::NPUCachingAllocator::BlockInfo;
    using c10_npu::NPUCachingAllocator::SegmentInfo;

    py::str device_s = "device";
    py::str address_s = "address";
    py::str total_size_s = "total_size";
    py::str allocated_size_s = "allocated_size";
    py::str active_size_s = "active_size";
    py::str requested_size_s = "requested_size";
    py::str stream_s = "stream";
    py::str segment_type_s = "segment_type";
    py::str segment_pool_id = "segment_pool_id";
    py::str large_s = "large";
    py::str small_s = "small";
    py::str size_s = "size";
    py::str state_s = "state";
    py::str active_allocated_s = "active_allocated";
    py::str active_pending_free_s = "active_pending_free";
    py::str inactive_s = "inactive";
    py::str addr_s = "addr";
    py::str cpp_frames_s = "cpp_frames";
    py::str blocks_s = "blocks";
    py::str is_expandable_s = "is_expandable";
    py::str frames_s = "frames";

    py::list empty_frames;
    std::vector<CapturedTraceback*> to_gather_frames;
    std::vector<py::dict> to_gather_dest;

    auto add_frame_key = [&](const py::dict& d, const std::shared_ptr<c10::GatheredContext>& ctx) {
        if (ctx) {
            auto sc = getFromContext(ctx);
            to_gather_frames.emplace_back(sc);
            to_gather_dest.emplace_back(d);
        } else {
            d[frames_s] = empty_frames;
        }
    };

    const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
        py::dict segmentDict;
        segmentDict[device_s] = segmentInfo.device;
        segmentDict[address_s] = segmentInfo.address;
        segmentDict[total_size_s] = segmentInfo.total_size;
        segmentDict[allocated_size_s] = segmentInfo.allocated_size;
        segmentDict[active_size_s] = segmentInfo.active_size;
        segmentDict[requested_size_s] = segmentInfo.requested_size;
        // we want the python objects to pickle easily so use an int to
        // represent the stream rather than a torch.cuda.stream object
        segmentDict[stream_s] = int64_t(segmentInfo.stream);
        segmentDict[segment_pool_id] = segmentInfo.owner_private_pool_id;
        segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);
        segmentDict[is_expandable_s] = segmentInfo.is_expandable;
        add_frame_key(segmentDict, segmentInfo.context_when_allocated);

        auto address = segmentInfo.address;
        py::list blocks;
        for (const auto& blockInfo : segmentInfo.blocks) {
            py::dict blockDict;
            blockDict[address_s] = address;
            blockDict[size_s] = blockInfo.size;
            blockDict[requested_size_s] = blockInfo.requested_size;
            blockDict[state_s] =
                (blockInfo.allocated ? active_allocated_s : (blockInfo.active ? active_pending_free_s : inactive_s));
            add_frame_key(blockDict, blockInfo.context_when_allocated);
            blocks.append(blockDict);
            address += blockInfo.size;
        }
        segmentDict[blocks_s] = blocks;

        return segmentDict;
    };

    auto snapshot = c10_npu::NPUCachingAllocator::snapshot();
    py::list segments;

    for (const auto& segmentInfo : snapshot.segments) {
        segments.append(segmentInfoToDict(segmentInfo));
    }

    auto workspace_snapshot = c10_npu::NPUWorkspaceAllocator::snapshot();
    for (size_t i = 0; i < workspace_snapshot.segments.size(); i++) {
        segments.append(segmentInfoToDict(workspace_snapshot.segments[i]));
    }

    for (size_t i = 0; i < workspace_snapshot.device_traces.size(); i++) {
        snapshot.device_traces[i].insert(snapshot.device_traces[i].begin(), workspace_snapshot.device_traces[i].begin(),
                                         workspace_snapshot.device_traces[i].end());
    }

    py::list traces;
    py::str action_s = "action";
    py::str alloc_s = "alloc";
    py::str free_requested_s = "free_requested";
    py::str free_completed_s = "free_completed";
    py::str segment_alloc_s = "segment_alloc";
    py::str segment_free_s = "segment_free";
    py::str segment_map_s = "segment_map";
    py::str segment_unmap_s = "segment_unmap";

    py::str snapshot_s = "snapshot";
    py::str workspace_snapshot_s = "workspace_snapshot";
    py::str oom_s = "oom";
    py::str device_free_s = "device_free";

    using namespace c10_npu::NPUCachingAllocator;

    auto action_to_str = [&](TraceEntry::Action action) {
        switch (action) {
            case TraceEntry::ALLOC:
                return alloc_s;
            case TraceEntry::FREE_REQUESTED:
                return free_requested_s;
            case TraceEntry::FREE_COMPLETED:
                return free_completed_s;
            case TraceEntry::SEGMENT_ALLOC:
                return segment_alloc_s;
            case TraceEntry::SEGMENT_FREE:
                return segment_free_s;
            case TraceEntry::OOM:
                return oom_s;
            case TraceEntry::SNAPSHOT:
                return snapshot_s;
            case TraceEntry::WORKSPACE_SNAPSHOT:
                return workspace_snapshot_s;
            case TraceEntry::SEGMENT_UNMAP:
                return segment_unmap_s;
            case TraceEntry::SEGMENT_MAP:
                return segment_map_s;
            default:
                AT_ERROR("invalid TraceEntry action");
        }
        throw std::runtime_error("unreachable");
    };

    for (const auto& traceInfo : snapshot.device_traces) {
        py::list trace;
        for (const auto& te : traceInfo) {
            py::dict trace_entry;
            if (te.context_) {
                // without further compression frames can get really large on dump
                auto sc = getFromContext(te.context_);
                to_gather_frames.emplace_back(sc);
                to_gather_dest.emplace_back(trace_entry);
            }
            trace_entry[action_s] = action_to_str(te.action_);
            trace_entry[te.action_ == TraceEntry::OOM ? device_free_s : addr_s] = te.addr_;
            trace_entry[size_s] = te.size_;
            trace_entry[stream_s] = int64_t(te.stream_);
            trace.append(trace_entry);
        }
        traces.append(trace);
    }

    py::dict result;
    result["segments"] = segments;
    result["device_traces"] = traces;

#if defined(__x86_64__)
    auto frames = torch::py_symbolize(to_gather_frames);
#else
    auto frames = torch_npu::py_symbolize(to_gather_frames);
#endif
    for (auto i : c10::irange(frames.size())) {
        to_gather_dest.at(i)[frames_s] = frames.at(i);
    }
    return result.release().ptr();
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_saveDevMemUsageInfo(PyObject *_unused, PyObject *arg)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to save device mem usage info.", PTA_ERROR(ErrCode::PARAM));
    const int device = (int) THPUtils_unpackLong(arg);
    bool ret = c10_npu::NPUCachingAllocator::saveDevMemUsageInfo(device);
    if (ret) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_attachOutOfMemoryObserver(PyObject* _unused, PyObject* observer)
{
    HANDLE_TH_ERRORS
    Py_XINCREF(observer);
    auto obs = [observer](int64_t device, int64_t alloc, int64_t device_allocated, int64_t device_free) {
        py::gil_scoped_acquire g;
        PyObject* result = PyObject_CallFunction(observer, "LLLL", device, alloc, device_allocated, device_free);
        if (!result) {
            throw py::error_already_set();
        }
        Py_XDECREF(result);
    };
    torch_npu::utils::npu_lazy_init();
    c10_npu::NPUCachingAllocator::attachOutOfMemoryObserver(std::move(obs));
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuCachingAllocator_raw_alloc(PyObject *_unused, PyObject *args)
{
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

PyObject* THNPModule_npuCachingAllocator_raw_delete(PyObject *_unused, PyObject *obj)
{
    HANDLE_TH_ERRORS
    void* mem_ptr = PyLong_AsVoidPtr(obj);
    c10_npu::NPUCachingAllocator::raw_delete(mem_ptr);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npuCachingAllocator_set_allocator_settings(PyObject *_unused, PyObject *arg)
{
    HANDLE_TH_ERRORS
    std::string settings = THPUtils_unpackString(arg);
    c10_npu::NPUCachingAllocator::setAllocatorSettings(settings);
    END_HANDLE_TH_ERRORS
    Py_RETURN_NONE;
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

PyObject* THNPModule_initDump(PyObject* _unused, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    pybind11::gil_scoped_release no_gil;
    NPU_CHECK_ERROR_WITHOUT_UCE(aclmdlInitDump());
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_setDump(PyObject* _unused, PyObject* arg)
{
    HANDLE_TH_ERRORS
    if (!THPUtils_checkString(arg)) {
        THPUtils_setError("npu set dump error, cfg_file must string");
    }
    std::string cfg_file = THPUtils_unpackString(arg);
    {
        pybind11::gil_scoped_release no_gil;
        NPU_CHECK_ERROR_WITHOUT_UCE(aclmdlSetDump(cfg_file.c_str()));
    }
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_finalizeDump(PyObject* _unused, PyObject* noargs)
{
    c10_npu::npuSynchronizeDevice();
    HANDLE_TH_ERRORS
    pybind11::gil_scoped_release no_gil;
    NPU_CHECK_ERROR_WITHOUT_UCE(aclmdlFinalizeDump());
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_setOption_wrap(PyObject* self, PyObject* arg)
{
    HANDLE_TH_ERRORS

    if (!PyDict_Check(arg)) {
        throw torch::TypeError("npu option must be a dict." + PTA_ERROR(ErrCode::TYPE));
    }

    PyObject *key = nullptr;
    PyObject *value = nullptr;
    Py_ssize_t pos = 0;
    std::map<std::string, std::string> option;

    while (PyDict_Next(arg, &pos, &key, &value)) {
        if (key == nullptr || !PyUnicode_Check(key)) {
            throw torch::TypeError("option name is nullptr or is not string." + PTA_ERROR(ErrCode::TYPE));
        }

        if (value == nullptr || !PyUnicode_Check(value)) {
            throw torch::TypeError("option value is nullptr or is not string." + PTA_ERROR(ErrCode::TYPE));
        }

        const char *pKey = PyUnicode_AsUTF8(key);
        const char *pValue = PyUnicode_AsUTF8(value);
        option[pKey] = pValue;
    }

    {
        pybind11::gil_scoped_release no_gil;
        c10_npu::option::SetOption(option);
    }
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
    if (has_overflow) {
        Py_RETURN_TRUE;
    } else {
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
    TORCH_CHECK(THPUtils_checkString(option_type), "invalid argument to option_type,option_type must string!", PTA_ERROR(ErrCode::PARAM));
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
        THPUtils_checkLong(arg), "invalid argument to set_sync_debug_mode, debug_mode type must long", PTA_ERROR(ErrCode::PARAM));
    int64_t debug_mode = THPUtils_unpackLong(arg);
    TORCH_CHECK(
        debug_mode >= 0 && debug_mode <= 2,
        "invalid value of debug_mode, expected one of 0,1,2", PTA_ERROR(ErrCode::VALUE));
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
        /* traceable= */
        false);

    torch::ParsedArgs<1> parsed_args;
    auto _r = parser.parse(args, nullptr, parsed_args);

    at::ScalarType storage_scalar_type;
    bool is_typed_storage = true;
    c10::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
    return THPVariable_Wrap(at_npu::native::set_tensor_with_storage_format(storage));

    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_set_call_state(PyObject* _unused, PyObject* arg)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(
        THPUtils_checkString(arg), "invalid value of call_state, call_state must string", PTA_ERROR(ErrCode::PARAM));
    std::string state = THPUtils_unpackString(arg);
    c10_npu::CallStateMode mode = c10_npu::CallStateMode::L_UNKNOW;
    if (state == "forward") {
        mode = c10_npu::CallStateMode::L_FORWARD;
    } else if (state == "backward") {
        mode = c10_npu::CallStateMode::L_BACKWARD;
    } else {
        TORCH_CHECK(false, "invalid value of call_state, expected one of `forward`, `backward`", PTA_ERROR(ErrCode::PARAM));
    }
    c10_npu::model_state().set_call_state(mode);
    ASCEND_LOGI("NPU set call state success, state is %s.", state.c_str());
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_set_module_train_state(PyObject* _unused, PyObject* arg)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(
        THPUtils_checkString(arg), "invalid value of train_state, train_state must string", PTA_ERROR(ErrCode::PARAM));
    std::string state = THPUtils_unpackString(arg);
    c10_npu::ModelMode mode = c10_npu::ModelMode::L_UNKNOW;
    if (state == "train") {
        mode = c10_npu::ModelMode::L_TRAIN;
    } else if (state == "infer") {
        mode = c10_npu::ModelMode::L_INFER;
    } else {
        TORCH_CHECK(false, "invalid value of train_state, expected one of `train`, `infer`", PTA_ERROR(ErrCode::PARAM));
    }
    c10_npu::model_state().set_model_mode(mode);
    ASCEND_LOGI("NPU set train state success, state is %s.", state.c_str());
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_get_silent_check_version(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    if (c10_npu::opapi::IsExistAclnnSilentCheck()) {
        // silent check v2
        return PyLong_FromLong(2);
    }
    // silent check v1
    return PyLong_FromLong(1);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_aclnn_reselect_static_kernel(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    NPUStatus ret = c10_npu::emptyAllNPUStream();
    TORCH_CHECK(ret == NPU_STATUS_SUCCESS, "Failed to empty NPU task queue, ret:", ret, PTA_ERROR(ErrCode::INTERNAL));

    static const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
    if (task_queue_enable == 2) {
        auto acl_call = []()->int {
            c10_npu::opapi::ReselectStaticKernel();
            return 0;
        };
        at_npu::native::OpCommand::RunOpApiV2("reselect_static_kernel", acl_call);
        NPUStatus ret = c10_npu::emptyAllNPUStream();
        TORCH_CHECK(ret == NPU_STATUS_SUCCESS, "Failed to empty NPU task queue, ret:", ret, PTA_ERROR(ErrCode::INTERNAL));
    } else {
        c10_npu::opapi::ReselectStaticKernel();
    }

    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_set_thread_affinity(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    int core_start;
    int core_end;
    if (!PyArg_ParseTuple(args, "ii", &core_start, &core_end)) {
        throw torch::TypeError("Pybind failed to parse parameters." + PTA_ERROR(ErrCode::TYPE));
    }

    if (core_start == -1) {
        c10_npu::SetThreadAffinity(c10_npu::ThreadType::OTHER_THREAD);
    } else {
        c10_npu::SetThreadAffinity(core_start, core_end);
    }

    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_reset_thread_affinity(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    c10_npu::SetThreadAffinity(c10_npu::ThreadType::MAIN_THREAD);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_set_fft_plan_cache_max_size(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    static torch::PythonArgParser parser(
        {"set_fft_plan_cache_max_size(int64_t size)", },
        false);

    torch::ParsedArgs<1> parsed_args;
    auto _r = parser.parse(args, nullptr, parsed_args);

    int64_t cache_size = _r.toInt64(0);
    TORCH_CHECK(
        cache_size >= 1 && cache_size <= 99,
        "invalid value of cache_size, expected 1 to 99",
        PTA_ERROR(ErrCode::VALUE));
    op_api::setFFTPlanCapacity(cache_size);

    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_get_fft_plan_cache_max_size(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    return PyLong_FromLong(op_api::getFFTPlanCapacity());
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_get_fft_plan_cache_size(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    return PyLong_FromLong(op_api::getFFTPlanSize());
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_npu_clear_fft_plan_cache(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    op_api::clearFFTPlanCache();
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPModule_get_cann_version(PyObject* self, PyObject *args)
{
    HANDLE_TH_ERRORS
    TORCH_CHECK(THPUtils_checkString(args), "invalid value of module, module must be string", PTA_ERROR(ErrCode::PARAM));
    std::string module = THPUtils_unpackString(args);
    std::string version = GetCANNVersion(module);
    return THPUtils_packString(version);
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPModule_is_gte_cann_version(PyObject* self, PyObject *args)
{
    HANDLE_TH_ERRORS
    static torch::PythonArgParser parser(
        {"_is_gte_cann_version(std::string version, std::string module)", },
        false);
    torch::ParsedArgs<2> parsed_args;
    auto _r = parser.parse(args, nullptr, parsed_args);
    string version = _r.string(0);
    string module = _r.string(1);

    bool compareResult = IsGteCANNVersion(version, module);
    return Py_BuildValue("i", int(compareResult));
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPModule_set_device_res_limit(PyObject* self, PyObject *args)
{
    HANDLE_TH_ERRORS
    PyObject* device = nullptr;
    PyObject* type = nullptr;
    PyObject* value = nullptr;

    if (!PyArg_ParseTuple(args, "OOO",  &device,  &type, &value)) {
        throw torch::TypeError("Pybind failed to parse parameters." +
                               PTA_ERROR(ErrCode::TYPE));
    }
    int32_t device_ = THPUtils_unpackLong(device);
    int32_t type_ = THPUtils_unpackLong(type);
    uint32_t value_ =  static_cast<uint32_t>(THPUtils_unpackUInt32(value));
    c10_npu::SetDeviceResLimit(device_, type_, value_);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPModule_get_device_res_limit(PyObject* self, PyObject *args)
{
    HANDLE_TH_ERRORS
    PyObject* device = nullptr;
    PyObject* type = nullptr;

    if (!PyArg_ParseTuple(args, "OO",  &device, &type)) {
        throw torch::TypeError("Pybind failed to parse parameters." +
                               PTA_ERROR(ErrCode::TYPE));
    }
    int32_t device_ = THPUtils_unpackLong(device);
    int32_t type_ = THPUtils_unpackLong(type);
    uint32_t value = c10_npu::GetDeviceResLimit(device_, type_);
    return PyLong_FromUnsignedLong(value);
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPModule_reset_device_res_limit(PyObject* self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int32_t device = THPUtils_unpackLong(args);
    c10_npu::ResetDeviceResLimit(device);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_aclop_start_dump(PyObject* self, PyObject* args)
{
    HANDLE_TH_ERRORS
    uint32_t dump_type = 0x00000001U;
    std::string dump_path = THPUtils_unpackString(args);
    at_npu::native::AclopStartDumpArgs(dump_type, dump_path.c_str());
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_aclop_stop_dump(PyObject* self, PyObject* noargs)
{
    HANDLE_TH_ERRORS
    uint32_t dump_type = 0x00000001U;
    at_npu::native::AclopStopDumpArgs(dump_type);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPModule_set_stream_res_limit(PyObject* self, PyObject *args, PyObject* kwargs)
{
    HANDLE_TH_ERRORS
    int64_t stream_id = 0;
    int64_t device_index = 0;
    int64_t device_type = 0;
    PyObject* type = nullptr;
    PyObject* value = nullptr;

    constexpr const char* kwlist[] = {
        "stream_id", "device_index", "device_type", "type", "value", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "LLLOO",
        const_cast<char**>(kwlist),
        &stream_id,
        &device_index,
        &device_type,
        &type,
        &value)) {
        throw torch::TypeError("Pybind failed to parse parameters." +
                               PTA_ERROR(ErrCode::TYPE));
    }
    auto stream = c10_npu::NPUStream::unpack3(
        stream_id, device_index, static_cast<c10::DeviceType>(device_type));
    int32_t type_ = THPUtils_unpackLong(type);
    uint32_t value_ =  static_cast<uint32_t>(THPUtils_unpackUInt32(value));
    c10_npu::SetStreamResLimit(stream, type_, value_);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPModule_reset_stream_res_limit(PyObject* self, PyObject *args, PyObject* kwargs)
{
    HANDLE_TH_ERRORS
    int64_t stream_id = 0;
    int64_t device_index = 0;
    int64_t device_type = 0;

    constexpr const char* kwlist[] = {
        "stream_id", "device_index", "device_type", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "LLL",
        const_cast<char**>(kwlist),
        &stream_id,
        &device_index,
        &device_type)) {
        throw torch::TypeError("Pybind failed to parse parameters." +
                               PTA_ERROR(ErrCode::TYPE));
    }
    auto stream = c10_npu::NPUStream::unpack3(
        stream_id, device_index, static_cast<c10::DeviceType>(device_type));
    c10_npu::ResetStreamResLimit(stream);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

static PyObject* THNPModule_get_stream_res_limit(PyObject* self, PyObject *args, PyObject* kwargs)
{
    HANDLE_TH_ERRORS
    int64_t stream_id = 0;
    int64_t device_index = 0;
    int64_t device_type = 0;
    PyObject* type = nullptr;

    constexpr const char* kwlist[] = {
        "stream_id", "device_index", "device_type", "type", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "LLLO",
        const_cast<char**>(kwlist),
        &stream_id,
        &device_index,
        &device_type,
        &type)) {
        throw torch::TypeError("Pybind failed to parse parameters." +
                               PTA_ERROR(ErrCode::TYPE));
    }
    auto stream = c10_npu::NPUStream::unpack3(
        stream_id, device_index, static_cast<c10::DeviceType>(device_type));
    int32_t type_ = THPUtils_unpackLong(type);
    uint32_t value = c10_npu::GetStreamResLimit(stream, type_);
    return PyLong_FromUnsignedLong(value);
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_setOpTimeoutMs(PyObject* self, PyObject* arg)
{
    HANDLE_TH_ERRORS
    uint32_t timeout_ms = THPUtils_unpackUInt32(arg);
    NPUStatus ret = c10_npu::emptyAllNPUStream();
    if (ret != NPU_STATUS_SUCCESS) {
        ASCEND_LOGE("MakeSureQueueEmpty fail, ret: %s", ret.c_str());
    }
    uint64_t timeout_us = static_cast<uint64_t>(timeout_ms) * 1000;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtSetOpExecuteTimeOutV2(timeout_us));
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_set_deterministic_level(PyObject* self, PyObject* arg)
{
    HANDLE_TH_ERRORS
    uint32_t level = THPUtils_unpackUInt32(arg);
    c10_npu::SetDeterministicLevel(level);
    Py_RETURN_NONE;
    END_HANDLE_TH_ERRORS
}

PyObject* THNPModule_get_deterministic_level(PyObject* self, PyObject*  noargs)
{
    HANDLE_TH_ERRORS
    uint32_t level = c10_npu::GetDeterministicLevel();
    return THPUtils_packUInt32(level);
    END_HANDLE_TH_ERRORS
}

static struct PyMethodDef THNPModule_methods[] = {
    {"_npu_init", (PyCFunction)THNPModule_initExtension, METH_NOARGS, nullptr},
    {"_npu_set_run_yet_variable_to_false", (PyCFunction)THNPModule_set_run_yet_variable_to_false_wrap, METH_NOARGS, nullptr},
    {"_npu_synchronize", (PyCFunction)THNPModule_npuSynchronize, METH_NOARGS, nullptr},
    {"_npu_setDevice", (PyCFunction)THNPModule_setDevice_wrap, METH_O, nullptr},
    {"_npu_set_op_timeout_ms", (PyCFunction)THNPModule_setOpTimeoutMs, METH_O, nullptr},
    {"_npu_getDevice", (PyCFunction)THNPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_npu_getDeviceWithoutSet", (PyCFunction)THNPModule_getDeviceWithoutSet_wrap, METH_NOARGS, nullptr},
    {"_npu_maybeExchangeDevice", (PyCFunction)THNPModule_maybeExchangeDevice_wrap, METH_O, nullptr},
    {"_npu_stopDevice", (PyCFunction)THNPModule_stopDevice_wrap, METH_O, nullptr},
    {"_npu_restart_device", (PyCFunction)THNPModule_restart_device_wrap, METH_O, nullptr},
    {"_npu_check_uce_in_memory", (PyCFunction)THNPModule_check_uce_in_memory_wrap, METH_O, nullptr},
    {"_npu_get_uce_addr", (PyCFunction)THNPModule_get_uce_addr_wrap, METH_NOARGS, nullptr},
    {"_npu_stress_detect", (PyCFunction)THNPModule_stressDetect_wrap, METH_VARARGS, nullptr},
    {"_npu_getLocalDevice", (PyCFunction)THNPModule_getLocalDevice_wrap, METH_NOARGS, nullptr},
    {"_npu_getDeviceCount", (PyCFunction)THNPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
    {"_npu_canDeviceAccessPeer", (PyCFunction)THNPModule_npuCanDeviceAccessPeer_wrap, METH_VARARGS, nullptr},
    {"_npu_getDeviceUtilizationRate", (PyCFunction)THNPModule_getDeviceUtilizationRate_wrap, METH_O, nullptr},
    {"_npu_getCurrentStream", (PyCFunction)THNPModule_getCurrentStream_wrap, METH_O, nullptr},
    {"_npu_getCurrentRawStream", (PyCFunction)THNPModule_getCurrentStream_raw, METH_O, nullptr},
    {"_npu_getCurrentRawStreamNoWait", (PyCFunction)THNPModule_getCurrentRawStreamNoWait_wrap, METH_O, nullptr},
    {"_npu_getDefaultStream", (PyCFunction)THNPModule_getDefaultStream_wrap, METH_O, nullptr},
    {"_npu_setStream", (PyCFunction)THNPModule_setStream_wrap,  METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_npu_eraseStream", (PyCFunction)THNPModule_npu_eraseStream_wrap, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_npu_isCurrentStreamCapturing", (PyCFunction)THNPModule_isCurrentStreamCapturing_wrap, METH_NOARGS, nullptr},
    {"_npu_is_jit_compile_false", (PyCFunction)THNPModule_is_jit_compile_false_wrap, METH_NOARGS, nullptr},
    {"_npu_setMemoryFraction", (PyCFunction) THNPModule_setMemoryFraction, METH_VARARGS, nullptr},
    {"_npu_emptyCache", (PyCFunction) THNPModule_emptyCache, METH_NOARGS, nullptr},
    {"_npu_hostEmptyCache", (PyCFunction) THNPModule_npu_hostEmptyCache, METH_NOARGS, nullptr},
    {"_npu_ipc_collect", (PyCFunction) THNPModule_npu_ipc_collect, METH_NOARGS, nullptr},
    {"_npu_emptyVirtAddrCache", (PyCFunction) THNPModule_emptyVirtAddrCache, METH_NOARGS, nullptr},
    {"_npu_memoryStats", (PyCFunction) THNPModule_memoryStats, METH_O, nullptr},
    {"_npu_resetAccumulatedMemoryStats", (PyCFunction) THNPModule_resetAccumulatedMemoryStats, METH_O, nullptr},
    {"_npu_resetPeakMemoryStats", (PyCFunction) THNPModule_resetPeakMemoryStats, METH_O,  nullptr},
    {"_npu_memorySnapshot", (PyCFunction) THNPModule_memorySnapshot, METH_NOARGS, nullptr},
    {"_npu_saveDevMemUsageInfo", (PyCFunction) THNPModule_saveDevMemUsageInfo, METH_O, nullptr},
    {"_npu_attach_out_of_memory_observer", THNPModule_attachOutOfMemoryObserver, METH_O, nullptr},
    {"_npu_npuCachingAllocator_raw_alloc", (PyCFunction)THNPModule_npuCachingAllocator_raw_alloc, METH_VARARGS, nullptr},
    {"_npu_npuCachingAllocator_raw_delete", (PyCFunction)THNPModule_npuCachingAllocator_raw_delete, METH_O, nullptr},
    {"_npu_npuCachingAllocator_set_allocator_settings", (PyCFunction)THNPModule_npuCachingAllocator_set_allocator_settings, METH_O, nullptr},
    {"_npu_getAllocatorBackend", (PyCFunction)THNPModule_getAllocatorBackend, METH_NOARGS, nullptr},
    {"_npu_lock_mutex",   (PyCFunction)THNPModule_npuLockMutex,   METH_NOARGS,  nullptr},
    {"_npu_unlock_mutex", (PyCFunction)THNPModule_npuUnlockMutex, METH_NOARGS,  nullptr},
    {"_npu_initDump", (PyCFunction)THNPModule_initDump, METH_NOARGS, nullptr},
    {"_npu_setDump", (PyCFunction)THNPModule_setDump, METH_O, nullptr},
    {"_npu_finalizeDump", (PyCFunction)THNPModule_finalizeDump, METH_NOARGS, nullptr},
    {"_npu_setOption", (PyCFunction)THNPModule_setOption_wrap, METH_O, nullptr},
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
    {"_npu_set_call_state", (PyCFunction)THNPModule_npu_set_call_state, METH_O, nullptr},
    {"_npu_set_module_train_state", (PyCFunction)THNPModule_npu_set_module_train_state, METH_O, nullptr},
    {"_get_silent_check_version", (PyCFunction)THNPModule_npu_get_silent_check_version, METH_NOARGS, nullptr},
    {"_aclnn_reselect_static_kernel", (PyCFunction)THNPModule_aclnn_reselect_static_kernel, METH_NOARGS, nullptr},
    {"_npu_set_thread_affinity", (PyCFunction)THNPModule_npu_set_thread_affinity, METH_VARARGS, nullptr},
    {"_npu_reset_thread_affinity", (PyCFunction)THNPModule_npu_reset_thread_affinity, METH_NOARGS, nullptr},
    {"_npu_set_fft_plan_cache_max_size", (PyCFunction)THNPModule_npu_set_fft_plan_cache_max_size, METH_VARARGS, nullptr},
    {"_npu_get_fft_plan_cache_max_size", (PyCFunction)THNPModule_npu_get_fft_plan_cache_max_size, METH_NOARGS, nullptr},
    {"_npu_get_fft_plan_cache_size", (PyCFunction)THNPModule_npu_get_fft_plan_cache_size, METH_NOARGS, nullptr},
    {"_npu_clear_fft_plan_cache", (PyCFunction)THNPModule_npu_clear_fft_plan_cache, METH_NOARGS, nullptr},
    {"_get_cann_version", (PyCFunction)THNPModule_get_cann_version, METH_O, nullptr},
    {"_is_gte_cann_version", (PyCFunction)THNPModule_is_gte_cann_version, METH_VARARGS, nullptr},
    {"_npu_get_device_res_limit", (PyCFunction)THNPModule_get_device_res_limit, METH_VARARGS, nullptr},
    {"_npu_set_device_res_limit", (PyCFunction)THNPModule_set_device_res_limit, METH_VARARGS, nullptr},
    {"_npu_reset_device_res_limit", (PyCFunction)THNPModule_reset_device_res_limit, METH_O, nullptr},
    {"_aclop_start_dump", (PyCFunction)THNPModule_aclop_start_dump, METH_O, nullptr},
    {"_aclop_stop_dump", (PyCFunction)THNPModule_aclop_stop_dump, METH_NOARGS, nullptr},
    {"_npu_set_stream_res_limit", (PyCFunction)THNPModule_set_stream_res_limit, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_npu_reset_stream_res_limit", (PyCFunction)THNPModule_reset_stream_res_limit, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_npu_get_stream_res_limit", (PyCFunction)THNPModule_get_stream_res_limit, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_npu_set_deterministic_level", (PyCFunction)THNPModule_set_deterministic_level, METH_O, nullptr},
    {"_npu_get_deterministic_level", (PyCFunction)THNPModule_get_deterministic_level, METH_NOARGS, nullptr},
    {nullptr}};

TORCH_NPU_API PyMethodDef* THNPModule_get_methods()
{
    return THNPModule_methods;
}

// Data Parallel Commands
void initCommMethods()
{
    auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
    if (!torch_C_module) {
        throw python_error();
    }
    auto m = py::handle(torch_C_module).cast<py::module>();
    m.def(
        "_broadcast_coalesced",
        [](std::vector<at::Tensor>& tensors,
            std::vector<int64_t> devices,
            size_t buffer_size) {
            return torch_npu::data_parallel::broadcast_coalesced(tensors, devices, buffer_size);
        },
        py::arg("tensors"),
        py::arg("devices"),
        py::arg("buffer_size"),
        py::call_guard<py::gil_scoped_release>())
        .def(
        "_broadcast",
        [](at::Tensor& tensor, std::vector<int64_t> devices) {
            return torch_npu::data_parallel::broadcast(tensor, devices);
        },
        py::arg("tensor"),
        py::arg("devices"),
        py::call_guard<py::gil_scoped_release>())
        .def(
        "_broadcast_out",
        [](at::Tensor& tensor, std::vector<at::Tensor>& out_tensors) {
            return torch_npu::data_parallel::broadcast_out(tensor, out_tensors);
        },
        py::arg("tensor"),
        py::arg("out"),
        py::call_guard<py::gil_scoped_release>())
        .def(
        "_scatter",
        [](at::Tensor& tensor,
            std::vector<int64_t>& devices,
            c10::optional<std::vector<int64_t>> chunk_sizes,
            int64_t dim,
            c10::optional<py::object> py_streams) {
            c10::optional<std::vector<c10::optional<c10_npu::NPUStream>>>
                streams;
            if (py_streams) {
            py::handle handle = *py_streams;
            streams = THNPUtils_PySequence_to_NPUStreamList(handle.ptr());
            }
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return torch_npu::data_parallel::scatter(tensor, devices, chunk_sizes, dim, streams);
        },
        py::arg("tensor"),
        py::arg("devices"),
        py::arg("chunk_sizes"),
        py::arg("dim"),
        py::arg("streams"))
        .def(
        "_scatter_out",
        [](at::Tensor& tensor,
            std::vector<at::Tensor>& out_tensors,
            int64_t dim,
            c10::optional<py::object> py_streams) {
            c10::optional<std::vector<c10::optional<c10_npu::NPUStream>>>
                streams;
            if (py_streams) {
            py::handle handle = *py_streams;
            streams = THNPUtils_PySequence_to_NPUStreamList(handle.ptr());
            }
            // Note: We're holding the GIL up to here.
            pybind11::gil_scoped_release no_gil;
            return torch_npu::data_parallel::scatter_out(tensor, out_tensors, dim, streams);
        },
        py::arg("tensor"),
        py::arg("out"),
        py::arg("dim"),
        py::arg("streams"))
        .def(
        "_gather",
        [](std::vector<at::Tensor>& tensors,
            int64_t dim,
            c10::optional<int32_t> destination_index) {
            return torch_npu::data_parallel::gather(tensors, dim, destination_index);
        },
        py::arg("tensors"),
        py::arg("dim"),
        py::arg("destination_index"),
        py::call_guard<py::gil_scoped_release>())
        .def(
        "_gather_out",
        [](std::vector<at::Tensor>& tensors,
            at::Tensor& out_tensor,
            int64_t dim) { return torch_npu::data_parallel::gather_out(tensors, out_tensor, dim); },
        py::arg("tensors"),
        py::arg("out"),
        py::arg("dim"),
        py::call_guard<py::gil_scoped_release>());
}
