#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void TORCH_NPU_API THNPMemPool_init(PyObject* module) {
    auto torch_C_m = py::handle(module).cast<py::module>();
    shared_ptr_class_<::c10_npu::MemPool>(torch_C_m, "_MemPool")
        .def(py::init<c10_npu::NPUCachingAllocator::NPUAllocator*, bool>())
        .def_property_readonly("id", &::c10_npu::MemPool::id)
        .def_property_readonly("allocator", &::c10_npu::MemPool::allocator);
    shared_ptr_class_<::c10_npu::MemPoolContext>(torch_C_m, "_MemPoolContext")
        .def(py::init<c10_npu::MemPool*>())
        .def_static(
            "active_pool", &::c10_npu::MemPoolContext::getActiveMemPool);
}
