#include "torch_npu/csrc/utils/TensorMethods.h"

namespace torch_npu {
namespace utils {


PyMethodDef* tensor_functions() {
  return TorchTensorMethods;
}

std::tuple<at::Tensor, c10::optional<at::Device>, c10::optional<at::ScalarType>, bool, bool, c10::optional<at::MemoryFormat>> parse_to_conversion(torch::PythonArgs& r, bool allow_copy) {
  if (r.idx == 0) {
    if (!allow_copy && !r.isNone(4))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(r.tensor(0), r.deviceOptional(1), r.scalartypeOptional(2), r.toBool(3), r.toBool(4), r.memoryformatOptional(5));
  } else if (r.idx == 1) {
    if (!allow_copy && !r.isNone(4))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(r.tensor(0), c10::nullopt, r.scalartype(1), r.toBool(2), r.toBool(3), r.memoryformatOptional(4));
  } else {
    auto tensor = r.tensor(1);
    if (!allow_copy && !r.isNone(5))
      throw std::runtime_error(".to() does not accept copy argument");
    return std::make_tuple(
        std::move(r.tensor(0)),
        tensor.device(),
        tensor.scalar_type(),
        r.toBool(2),
        r.toBool(3),
        r.memoryformatOptional(4)
    );
  }
}

}
}