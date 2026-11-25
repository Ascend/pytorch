#include <cstdint>

#include "exe_graph/fxrt_kernel_export.h"
#include "exe_graph/runtime/kernel_context.h"
#include "graph/error_codes.h"

namespace {
ge::graphStatus GetStorageOffset(gert::KernelContext *context) {
  if (context == nullptr || context->GetInputNum() != 2U || context->GetOutputNum() != 1U) {
    return ge::GRAPH_PARAM_INVALID;
  }
  // Input 1 is the live Data feed dependency; its tensor carries no offset in
  // the ES facade. Input 0 points into the session-owned host metadata instead.
  const auto *slot = context->GetInputValue<const int64_t *>(0U);
  auto *offset = context->GetOutputPointer<int64_t>(0U);
  if (slot == nullptr || offset == nullptr || *slot < 0) {
    return ge::GRAPH_PARAM_INVALID;
  }
  *offset = *slot;
  return ge::GRAPH_SUCCESS;
}
}  // namespace

FXRT_EXPORT_KERNEL_RUN(FxGetStorageOffset, GetStorageOffset)
