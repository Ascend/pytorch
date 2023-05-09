#include <torch/extension.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>

at::Tensor bscpp_add(const at::Tensor &self, const at::Tensor &other);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bscpp_add", &bscpp_add, "add two tensors by BiShengCPP.");
}
