// Fused InferShape: one Main node computes every symbolic output shape in the graph.
//
// The generated function evaluates every rule from the graph feeds' origin shapes. The
// per-op InferShape nodes' data-input edges were pure ordering, never data.
//
// Unlike GE's InferShape kernel this does not call TransformAllOutputsShape (which needs a
// ComputeNodeInfo). Every converter tensor desc is ND, so origin shape == storage shape and
// both halves are written here directly. A non-ND format would need the transform back.

#include "shape/fx_fused_infer_shape.h"
#include "shape/fx_infer_func.h"

#include <cstdint>
#include <vector>

#include "common/logger.h"
#include "exe_graph/fxrt_kernel_export.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"

namespace {

// Inputs: [0] FxInferFuncHandle*, [1..] graph feeds. Outputs follow the fused slot order.
ge::graphStatus FxFusedInferShapeKernel(gert::KernelContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto *handle = context->GetInputValue<const fx_converter::FxInferFuncHandle *>(0U);
  if (handle == nullptr || handle->func == nullptr) {
    RT_GLOG(ERROR) << "FxFusedInferShape: null infer function";
    return ge::GRAPH_FAILED;
  }
  std::vector<const gert::Tensor *> inputs;
  inputs.reserve(context->GetInputNum() - 1U);
  for (size_t i = 1U; i < context->GetInputNum(); ++i) {
    const auto *tensor = context->GetInputPointer<gert::Tensor>(i);
    if (tensor == nullptr) {
      RT_GLOG(ERROR) << "FxFusedInferShape: null feed tensor at " << (i - 1U);
      return ge::GRAPH_FAILED;
    }
    inputs.push_back(tensor);
  }
  std::vector<gert::Tensor *> outputs;
  outputs.reserve(context->GetOutputNum());
  for (size_t i = 0U; i < context->GetOutputNum(); ++i) {
    auto *tensor = context->GetOutputPointer<gert::Tensor>(i);
    if (tensor == nullptr) {
      auto *chain = context->GetOutput(i);
      if (chain == nullptr) {
        RT_GLOG(ERROR) << "FxFusedInferShape: null output chain at slot " << i;
        return ge::GRAPH_FAILED;
      }
      tensor = new (std::nothrow) gert::Tensor();
      if (tensor == nullptr) {
        RT_GLOG(ERROR) << "FxFusedInferShape: failed to allocate output tensor at slot " << i;
        return ge::GRAPH_FAILED;
      }
      chain->SetWithDefaultDeleter(tensor);
    }
    outputs.push_back(tensor);
  }
  return handle->func(inputs.data(), inputs.size(), outputs.data(), outputs.size()) == 0
             ? ge::GRAPH_SUCCESS
             : ge::GRAPH_FAILED;
}

// Downstream consumers read the shape; dtype and format come from BuildTensorAttr.
ge::graphStatus BuildFxFusedInferShapeOutputs(gert::KernelContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  for (size_t index = 0; index < context->GetOutputNum(); ++index) {
    auto *chain = context->GetOutput(index);
    if (chain == nullptr) {
      return ge::GRAPH_FAILED;
    }
    auto *tensor = new (std::nothrow) gert::Tensor();
    if (tensor == nullptr) {
      return ge::GRAPH_FAILED;
    }
    chain->SetWithDefaultDeleter(tensor);
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

FXRT_EXPORT_KERNEL_RUN(FxFusedInferShape, FxFusedInferShapeKernel)
FXRT_EXPORT_KERNEL_OUTPUTS(FxFusedInferShape, BuildFxFusedInferShapeOutputs)
