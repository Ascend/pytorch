#include "common/logger.h"

#include "shape/fx_symbol_env.h"
#include "exe_graph/fxrt_kernel_export.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/tensor.h"

namespace {

// Inputs: [0] FxSymbolEnv*, [1..] gert::Tensor* for graph feeds (index = feed_index).
ge::graphStatus BindSymbolEnvKernel(gert::KernelContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *env = context->GetInputValue<fx_converter::FxSymbolEnv *>(0);
  if (env == nullptr) {
    RT_GLOG(ERROR) << "BindSymbolEnv: null env";
    return ge::GRAPH_FAILED;
  }
  env->ClearValues();
  for (const auto &seed : env->seeds) {
    const size_t tensor_slot = static_cast<size_t>(seed.feed_index) + 1U;  // skip env at 0
    if (tensor_slot >= context->GetInputNum()) {
      RT_GLOG(ERROR) << "BindSymbolEnv: feed_index out of range for " << seed.name;
      return ge::GRAPH_FAILED;
    }
    const auto *tensor = context->GetInputPointer<gert::Tensor>(tensor_slot);
    if (tensor == nullptr) {
      RT_GLOG(ERROR) << "BindSymbolEnv: null tensor for " << seed.name;
      return ge::GRAPH_FAILED;
    }
    const auto &shape = tensor->GetOriginShape();
    if (seed.dim_index < 0 || static_cast<size_t>(seed.dim_index) >= shape.GetDimNum()) {
      RT_GLOG(ERROR) << "BindSymbolEnv: dim_index out of range for " << seed.name;
      return ge::GRAPH_FAILED;
    }
    const int64_t v = shape.GetDim(static_cast<size_t>(seed.dim_index));
    if (v <= 0) {
      RT_GLOG(ERROR) << "BindSymbolEnv: non-positive dim for " << seed.name;
      return ge::GRAPH_FAILED;
    }
    env->SetValue(seed.name, v);
  }
  env->bound = true;
  return ge::GRAPH_SUCCESS;
}

}  // namespace

FXRT_EXPORT_KERNEL_RUN(BindSymbolEnv, BindSymbolEnvKernel)
