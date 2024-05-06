#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
// test   in  .setup with relative path
#include <tmp.h>

using namespace at;

Tensor tanh_add(Tensor x, Tensor y)
{
    return x.tanh() + y.tanh();
}

Tensor npu_add(const Tensor &self_, const Tensor &other_)
{
    TORCH_INTERNAL_ASSERT(self_.device().type() == c10::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(other_.device().type() == c10::DeviceType::PrivateUse1);
    return at::add(self_, other_, 1);
}

bool check_storage_sizes(const Tensor &tensor, const c10::IntArrayRef &sizes)
{
    auto tensor_sizes = at_npu::native::get_npu_storage_sizes(tensor);
    if (tensor_sizes.size() == sizes.size()) {
        return std::equal(tensor_sizes.begin(), tensor_sizes.end(), sizes.begin());
    }
    return false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tanh_add", &tanh_add, "tanh(x) + tanh(y)");
    m.def("npu_add", &npu_add, "x + y");
    m.def("check_storage_sizes", &check_storage_sizes, "check_storage_sizes");
}
