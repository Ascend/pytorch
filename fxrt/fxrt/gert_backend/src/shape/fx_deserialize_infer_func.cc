/* Copyright 2026 Huawei Technologies Co., Ltd */

#include "shape/fx_infer_func.h"

#include <limits>

#include "common/logger.h"
#include "exe_graph/fxrt_kernel_export.h"
#include "exe_graph/runtime/kernel_context.h"

namespace {

ge::graphStatus FxDeserializeInferFuncKernel(gert::KernelContext *context) {
  if (context == nullptr || context->GetInputNum() != 1U || context->GetOutputNum() != 1U) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *data = context->GetInputValue<const uint8_t *>(0U);
  if (data == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t body_size = 0U;
  for (size_t i = 0U; i < sizeof(body_size); ++i) {
    body_size |= static_cast<uint64_t>(data[8U + i]) << (i * 8U);
  }
  if (body_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 16U)) {
    return ge::GRAPH_FAILED;
  }
  const fx_converter::FxSerializedBuffer serialized{data, body_size + 16U};
  fx_converter::FxInferFuncHandle *handle = nullptr;
  if (fx_converter::DeserializeInferFunc(serialized, &handle) != ge::GRAPH_SUCCESS || handle == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *output = context->GetOutput(0U);
  if (output == nullptr) {
    return ge::GRAPH_FAILED;
  }
  output->SetWithDefaultDeleter(handle);
  return ge::GRAPH_SUCCESS;
}

}  // namespace

FXRT_EXPORT_KERNEL_RUN(FxDeserializeInferFunc, FxDeserializeInferFuncKernel)
