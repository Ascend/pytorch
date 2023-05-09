#include <torch/csrc/utils/tensor_new.h>

#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace torch_npu {
namespace utils {

// Initializes the Python tensor type objects: torch.npu.FloatTensor,
// torch.npu.DoubleTensor, etc. and binds them in their containing modules.
void _initialize_python_bindings();

PyMethodDef* npu_extension_functions();

}
}
