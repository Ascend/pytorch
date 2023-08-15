#include <torch/csrc/dynamo/compiled_autograd.h>

#include "torch_npu/csrc/framework/autograd/FunctionsManual.h"
#ifdef USE_GEN_HEADER
#include "op_plugin/OpInterface.h"
#else
#include "op_plugin/ops/OpInterface.h"
#endif

// ${generated_comment}

// The manual function definitions that used to be here are now in torch/csrc/autograd/FunctionsManual.cpp
// This speeds up re-compilation and allow to share these implementations so that they can be
// used for forward mode AD formulas as well.

using namespace at_npu::autograd::generated::details;
using namespace op_plugin;
using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace at_npu { namespace autograd { namespace generated {

${autograd_function_definitions}

}}} // namespace at_npu::autograd::generated
