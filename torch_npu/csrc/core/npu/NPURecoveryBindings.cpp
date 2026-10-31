#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/core/npu/NPURecoveryBindings.h"

#include <torch/csrc/utils/pybind.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPURecovery.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace c10_npu {

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

} // namespace c10_npu
#endif
