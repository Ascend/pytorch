#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_npu/csrc/core/npu/MemPool.h"
#include "torch_npu/csrc/utils/LazyInit.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void TORCH_NPU_API THNPMemPool_init(PyObject* module) {
    auto torch_C_m = py::handle(module).cast<py::module>();
    shared_ptr_class_<::c10_npu::MemPool>(torch_C_m, "_MemPool")
        .def(py::init(
            [](std::shared_ptr<c10_npu::NPUCachingAllocator::NPUAllocator> allocator,
               bool is_user_created,
               bool use_on_oom,
               bool no_split) {
                torch_npu::utils::npu_lazy_init(); // init npu before construct mempool
                return std::make_shared<::c10_npu::MemPool>(allocator, is_user_created, use_on_oom, no_split);
            }))
        .def_property_readonly("id", &::c10_npu::MemPool::id)
        .def("use_count", &::c10_npu::MemPool::use_count);
}
