#include <torch/extension.h>

// test   in  .setup with relative path
#include <tmp.h>

using namespace at;

Tensor tanh_add(Tensor x, Tensor y) {
  return x.tanh() + y.tanh();
}

Tensor npu_add(const Tensor& self_, const Tensor& other_) {
  TORCH_INTERNAL_ASSERT(self_.device().type() == c10::DeviceType::PrivateUse1);
  TORCH_INTERNAL_ASSERT(other_.device().type() == c10::DeviceType::PrivateUse1);
  return at::add(self_, other_, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tanh_add", &tanh_add, "tanh(x) + tanh(y)");
  m.def("npu_add", &npu_add, "x + y");
}
