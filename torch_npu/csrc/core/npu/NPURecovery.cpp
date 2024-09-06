#include <shared_mutex>
#ifndef BUILD_LIBTORCH
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#endif

#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPURecovery.h"

namespace c10_npu {
namespace {

static bool NPU_DATA_UNSAFE_FLAG = false;
std::shared_mutex rw_mutex;

}

bool get_npu_data_unsafe_flag()
{
    std::shared_lock<std::shared_mutex> lock(rw_mutex);
    return NPU_DATA_UNSAFE_FLAG;
}

void set_npu_data_unsafe_flag(bool flag)
{
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    NPU_DATA_UNSAFE_FLAG = flag;
    ASCEND_LOGI("Set npu data unsafe flag to %d", flag);
    return;
}

void check_npu_tensor_is_safe(const at::Tensor& self)
{
    if (!torch_npu::utils::is_npu(self)) {
        return;
    }
    bool is_safe = NPUCachingAllocator::checkBlockIsSafe(self.storage().data_ptr());
    TORCH_CHECK(is_safe, "There is unsafe data in the input tensor.", PTA_ERROR(ErrCode::VALUE));
    return;
}

void check_npu_tensor_is_safe(const c10::optional<at::Tensor>& self)
{
    if (!self.has_value() or !torch_npu::utils::is_npu(self.value())) {
        return;
    }
    bool is_safe = NPUCachingAllocator::checkBlockIsSafe(self.value().storage().data_ptr());
    TORCH_CHECK(is_safe, "There is unsafe data in the input tensor.", PTA_ERROR(ErrCode::VALUE));
    return;
}

void check_npu_tensor_is_safe(const at::TensorList& self)
{
    for (const auto& s : self) {
        check_npu_tensor_is_safe(s);
    }
    return;
}

void check_npu_tensor_is_safe(const at::ITensorListRef& self)
{
    auto materialized = self.materialize();
    for (const auto& materialized_t : materialized) {
        check_npu_tensor_is_safe(materialized_t.get());
    }
    return;
}

void check_npu_tensor_is_safe(const c10::List<c10::optional<at::Tensor>>& self)
{
    for (const auto &s : self) {
        check_npu_tensor_is_safe(s);
    }
    return;
}

void update_npu_tensor_is_safe(const at::Tensor& self)
{
    if (!torch_npu::utils::is_npu(self)) {
        return;
    }
    return NPUCachingAllocator::updateBlockToSafe(self.storage().data_ptr());
}

void update_npu_tensor_is_safe(const at::TensorList& self)
{
    for (const auto& s : self) {
        update_npu_tensor_is_safe(s);
    }
    return;
}

void check_and_update_npu_tensor_for_copy(const at::Tensor& dst, const at::Tensor& src)
{
    check_npu_tensor_is_safe(src);
    update_npu_tensor_is_safe(dst);
    return;
}

void check_and_update_npu_tensor_for_copy(const at::TensorList& dsts, const at::TensorList& srcs)
{
    check_npu_tensor_is_safe(srcs);
    update_npu_tensor_is_safe(dsts);
    return;
}

#ifndef BUILD_LIBTORCH
void bind_npu_recovery_functions(PyObject* module)
{
    auto m = py::handle(module).cast<py::module>();
    m.def("_check_npu_data_ptr", [](const c10::Storage obj) -> bool {
        return c10_npu::NPUCachingAllocator::checkBlockIsSafe(obj.data_ptr());
    });
    m.def("_mark_all_npu_data_ptr_unsafe", [](int device) -> void {
        return c10_npu::NPUCachingAllocator::markAllBlockUnsafe(device);
    });
    m.def("_update_npu_data_ptr", [](const c10::Storage obj) -> void {
        return c10_npu::NPUCachingAllocator::updateBlockToSafe(obj.data_ptr());
    });
    m.def("_set_npu_data_unsafe_flag", [](bool flag) -> void {
        return set_npu_data_unsafe_flag(flag);
    });
    m.def("_get_npu_data_unsafe_flag", []() -> bool {
      return get_npu_data_unsafe_flag();
    });
    m.def("_recovery_all_npu_stream", [](int device) -> void {
        return c10_npu::recovery_all_npu_streams(device);
    });
}
#endif

}
