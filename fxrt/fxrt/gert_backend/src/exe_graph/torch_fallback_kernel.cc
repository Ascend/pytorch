#include <new>

#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/fxrt_kernel_export.h"
#include "lowering/torch_fallback/torch_fallback_runtime.h"

namespace {
// Same tail as the former ExecuteCustomOp node: allocator, stream, call pointer,
// attached-stream provider, root graph id, node id, vector core num.
constexpr size_t kTailInputs = 7U;
constexpr size_t kCallSlot = 2U;

ge::graphStatus CreateFxTorchFallbackOutputs(gert::KernelContext *context) {
  auto *extended = reinterpret_cast<gert::ExtendedKernelContext *>(context);
  if (context == nullptr || extended == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const size_t count = extended->GetComputeNodeOutputNum();
  for (size_t index = 0U; index < count; ++index) {
    auto *chain = context->GetOutput(index);
    const auto *desc = extended->GetOutputDesc(index);
    if (chain == nullptr || desc == nullptr) {
      return ge::GRAPH_FAILED;
    }
    auto *tensor = new (std::nothrow) gert::Tensor(gert::StorageShape(), desc->GetFormat(), desc->GetDataType());
    if (tensor == nullptr) {
      return ge::GRAPH_FAILED;
    }
    chain->SetWithDefaultDeleter(tensor);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CopyTemplateShapes(gert::KernelContext *context, size_t input_count, size_t output_count) {
  const size_t start = input_count + kTailInputs;
  for (size_t index = 0U; index < output_count; ++index) {
    const auto *shape = context->GetInputPointer<gert::Tensor>(start + index);
    auto *output = context->GetOutputPointer<gert::Tensor>(index);
    if (shape == nullptr || output == nullptr) {
      return ge::GRAPH_FAILED;
    }
    output->MutableStorageShape() = shape->GetStorageShape();
    output->MutableOriginShape() = shape->GetOriginShape();
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RunFallback(gert::KernelContext *context, bool copy_shapes) {
  auto *eager = reinterpret_cast<gert::EagerOpExecutionContext *>(context);
  if (context == nullptr || eager == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const size_t input_count = eager->GetComputeNodeInputNum();
  const size_t output_count = eager->GetComputeNodeOutputNum();
  auto *call = context->GetInputValue<fx_converter::TorchFallbackCall *>(input_count + kCallSlot);
  if (copy_shapes && CopyTemplateShapes(context, input_count, output_count) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return fx_converter::RunTorchFallback(call, eager);
}

ge::graphStatus RunDeferred(gert::KernelContext *context) { return RunFallback(context, false); }

ge::graphStatus RunWithShape(gert::KernelContext *context) { return RunFallback(context, true); }
}  // namespace

FXRT_EXPORT_KERNEL_RUN(FxTorchFallback, RunDeferred);
FXRT_EXPORT_KERNEL_OUTPUTS(FxTorchFallback, CreateFxTorchFallbackOutputs);
FXRT_EXPORT_KERNEL_RUN(FxTorchFallbackWithInferShape, RunWithShape);
FXRT_EXPORT_KERNEL_OUTPUTS(FxTorchFallbackWithInferShape, CreateFxTorchFallbackOutputs);
