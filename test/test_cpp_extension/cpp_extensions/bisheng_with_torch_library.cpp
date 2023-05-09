#include <torch/extension.h>

at::Tensor bscpp_add(const at::Tensor &self, const at::Tensor &other);

TORCH_LIBRARY(cpp_bisheng, m) {
  m.def("bscpp_add", &bscpp_add);
}
