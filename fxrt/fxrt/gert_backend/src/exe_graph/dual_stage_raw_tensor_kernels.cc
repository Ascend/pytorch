#include "exe_graph/dual_stage_raw_tensor_kernels.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "exe_graph/fxrt_kernel_export.h"
#include "acl/acl_rt.h"
#include "base/runtime/runtime_attrs_def.h"
#include "common/logger.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "register/op_impl_registry.h"

namespace gert {
namespace kernel {
namespace {

struct OpContextShadow {
  size_t tensor_count = 0U;
  Chain framework_data{};
  std::vector<uintptr_t> context_storage;
  std::vector<uintptr_t> compute_info_storage;
  KernelContext *context = nullptr;
};

bool OpIoCounts(const fx_converter::DualStageTensorLayout *layout, size_t tensor_count, size_t *op_in,
                size_t *op_out) {
  const size_t outs = layout->output_count == 0U ? 1U : static_cast<size_t>(layout->output_count);
  if (outs == 0U || outs > tensor_count) {
    return false;
  }
  *op_out = outs;
  *op_in = tensor_count - outs;
  return true;
}

bool BuildShadowContext(KernelContext *actual, size_t extend_count, OpContextShadow *shadow) {
  if (actual == nullptr || shadow == nullptr) {
    return false;
  }
  const auto *layout = actual->GetInputPointer<fx_converter::DualStageTensorLayout>(0);
  if (layout == nullptr || layout->tensor_count == 0U ||
      layout->tensor_count > fx_converter::kMaxDualStageTensorCount) {
    return false;
  }

  const size_t tensor_count = layout->tensor_count;
  shadow->tensor_count = tensor_count;
  const size_t actual_index = 1U + tensor_count;
  if (actual_index + extend_count != actual->GetInputNum()) {
    return false;
  }
  for (size_t i = 1U; i < actual->GetInputNum(); ++i) {
    if (actual->GetInput(i) == nullptr) {
      return false;
    }
  }

  const size_t input_count = tensor_count + extend_count;
  const size_t output_count = actual->GetOutputNum();
  size_t op_in_count = 0U;
  size_t op_out_count = 0U;
  if (tensor_count == 0U || !OpIoCounts(layout, tensor_count, &op_in_count, &op_out_count)) {
    return false;
  }
  size_t total_compute_info_size = 0U;
  const auto *source_compute_info = reinterpret_cast<const ComputeNodeInfo *>(actual->GetContext()->compute_node_info);
  if (source_compute_info == nullptr ||
      ComputeNodeInfo::CalcSize(0U, 0U, op_in_count, op_out_count, total_compute_info_size) != ge::GRAPH_SUCCESS) {
    return false;
  }
  const auto *source_attrs = reinterpret_cast<const uint8_t *>(source_compute_info->GetAttrs());
  if (source_attrs == nullptr) {
    return false;
  }
  size_t attrs_size = sizeof(RuntimeAttrsDef);
  if (source_compute_info->GetIrOutputsNum() > 0U) {
    const auto *source_output_instances =
        reinterpret_cast<const uint8_t *>(source_compute_info->GetOutputInstanceInfo(0U));
    if (source_output_instances == nullptr || source_output_instances < source_attrs + sizeof(RuntimeAttrs)) {
      return false;
    }
    attrs_size = static_cast<size_t>(source_output_instances - source_attrs);
  }
  std::vector<uint8_t> generated_attrs;
  if (source_compute_info->GetIrOutputsNum() == 0U) {
    generated_attrs.resize(attrs_size, 0U);
    auto *attrs_def = reinterpret_cast<RuntimeAttrsDef *>(generated_attrs.data());
    attrs_def->attr_num = 0U;
    source_attrs = generated_attrs.data();
  }
  if (layout->packed_attr_bytes != 0U) {
    source_attrs = reinterpret_cast<const uint8_t *>(layout) + sizeof(*layout);
    attrs_size = layout->packed_attr_bytes;
  } else if (layout->attr_count > 0U) {
    if (layout->attr_count > 2U) {
      return false;
    }
    attrs_size = sizeof(RuntimeAttrsDef) + layout->attr_count * sizeof(size_t) + layout->attr_count * sizeof(bool);
    generated_attrs.resize(attrs_size, 0U);
    auto *attrs_def = reinterpret_cast<RuntimeAttrsDef *>(generated_attrs.data());
    attrs_def->attr_num = layout->attr_count;
    const size_t value_offset = sizeof(RuntimeAttrsDef) + layout->attr_count * sizeof(size_t);
    for (size_t i = 0; i < layout->attr_count; ++i) {
      attrs_def->offset[i] = value_offset + i * sizeof(bool);
      generated_attrs[attrs_def->offset[i]] = layout->bool_attrs[i] != 0U ? 1U : 0U;
    }
    source_attrs = generated_attrs.data();
  }
  if (attrs_size > SIZE_MAX - total_compute_info_size) {
    return false;
  }
  total_compute_info_size += attrs_size;
  shadow->compute_info_storage.assign(
      (total_compute_info_size + sizeof(uintptr_t) - 1U) / sizeof(uintptr_t), 0U);
  auto *compute_info = reinterpret_cast<ComputeNodeInfo *>(shadow->compute_info_storage.data());
  compute_info->Init(0U, 0U, op_in_count, op_out_count, attrs_size, "fx_execute_op", "ExecuteOp");
  auto *attrs = reinterpret_cast<uint8_t *>(compute_info->MutableAttrs());
  if (attrs == nullptr) {
    return false;
  }
  std::memcpy(attrs, source_attrs, attrs_size);
  const size_t value_count = input_count + output_count;
  const size_t context_bytes = offsetof(KernelRunContext, values) + value_count * sizeof(AsyncAnyValue *);
  shadow->context_storage.assign((context_bytes + sizeof(uintptr_t) - 1U) / sizeof(uintptr_t), 0U);
  auto *run = reinterpret_cast<KernelRunContext *>(shadow->context_storage.data());
  const auto *actual_run = actual->GetContext();
  run->input_size = input_count;
  run->output_size = output_count;
  run->compute_node_info = compute_info;
  run->kernel_extend_info = actual_run->kernel_extend_info;
  run->output_start = &run->values[input_count];
  for (size_t i = 0; i < tensor_count; ++i) {
    run->values[i] = reinterpret_cast<AsyncAnyValue *>(actual->MutableInput(1U + i));
  }
  for (size_t i = 0; i < extend_count; ++i) {
    run->values[tensor_count + i] = reinterpret_cast<AsyncAnyValue *>(actual->MutableInput(actual_index + i));
  }
  for (size_t i = 0; i < output_count; ++i) {
    run->values[input_count + i] = reinterpret_cast<AsyncAnyValue *>(actual->GetOutput(i));
  }
  shadow->context = reinterpret_cast<KernelContext *>(run);
  return true;
}

// Only explicit per-op limits touch ACL resources. The default path preserves
// the caller's stream limits without querying them or taking any lock.
class StreamCoreLimitScope {
 public:
  explicit StreamCoreLimitScope(aclrtStream stream) : stream_(stream) {}

  ge::graphStatus Apply(int32_t cube, int32_t vector) {
    const int32_t requested[2] = {cube, vector};
    for (size_t i = 0U; i < 2U; ++i) {
      if (requested[i] <= 0) {
        continue;
      }
      if (aclrtGetStreamResLimit(stream_, types_[i], &saved_[i]) != ACL_SUCCESS ||
          aclrtSetStreamResLimit(stream_, types_[i], static_cast<uint32_t>(requested[i])) != ACL_SUCCESS) {
        return ge::GRAPH_FAILED;
      }
      changed_[i] = true;
    }
    if ((changed_[0] || changed_[1]) && aclrtUseStreamResInCurrentThread(stream_) != ACL_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus Restore() {
    bool failed = false;
    bool changed = false;
    for (size_t i = 0U; i < 2U; ++i) {
      if (changed_[i]) {
        changed = true;
        failed = aclrtSetStreamResLimit(stream_, types_[i], saved_[i]) != ACL_SUCCESS || failed;
        changed_[i] = false;
      }
    }
    if (changed) {
      failed = aclrtUseStreamResInCurrentThread(stream_) != ACL_SUCCESS || failed;
    }
    return failed ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
  }

 private:
  aclrtStream stream_;
  const aclrtDevResLimitType types_[2] = {ACL_RT_DEV_RES_CUBE_CORE, ACL_RT_DEV_RES_VECTOR_CORE};
  uint32_t saved_[2] = {};
  bool changed_[2] = {};
};

ge::graphStatus FxExecuteOpPrepare(KernelContext *context) {
  constexpr size_t kExtendCount = static_cast<size_t>(OpExecutePrepareInputExtendIndex::kNum);
  OpContextShadow shadow;
  if (!BuildShadowContext(context, kExtendCount, &shadow)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t function_index = 1U + shadow.tensor_count +
                                static_cast<size_t>(OpExecutePrepareInputExtendIndex::kFwkData);
  const auto func = context->GetInputValue<OpImplRegisterV2::OpExecPrepareFunc>(function_index);
  const auto stream = context->GetInputValue<aclrtStream>(
      1U + shadow.tensor_count + static_cast<size_t>(OpExecutePrepareInputExtendIndex::kStream));
  if (func == nullptr || stream == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  // Function pointers belong to the converter protocol. OPP callbacks retain
  // the public extension layout, with a null framework-reserved slot.
  auto *run = const_cast<KernelRunContext *>(shadow.context->GetContext());
  run->values[shadow.tensor_count + static_cast<size_t>(OpExecutePrepareInputExtendIndex::kFwkData)] =
      reinterpret_cast<AsyncAnyValue *>(&shadow.framework_data);
  const auto *layout = context->GetInputPointer<fx_converter::DualStageTensorLayout>(0U);
  StreamCoreLimitScope core_limits(stream);
  auto status = core_limits.Apply(layout->cube_core_limit, layout->vector_core_limit);
  if (status == ge::GRAPH_SUCCESS) {
    status = func(reinterpret_cast<OpExecutePrepareContext *>(shadow.context));
  }
  const auto reset_status = core_limits.Restore();
  if (status != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "[FxExecuteOpPrepare] prepare returned " << status;
  }
  return status == ge::GRAPH_SUCCESS ? reset_status : status;
}

ge::graphStatus FxExecuteOpLaunch(KernelContext *context) {
  constexpr size_t kExtendCount = static_cast<size_t>(OpExecuteLaunchInputIndex::kNum);
  OpContextShadow shadow;
  if (!BuildShadowContext(context, kExtendCount, &shadow)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t function_index = 1U + shadow.tensor_count +
                                static_cast<size_t>(OpExecuteLaunchInputIndex::kFwkData);
  const auto func = context->GetInputValue<OpImplRegisterV2::OpExecLaunchFunc>(function_index);
  if (func == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  auto *run = const_cast<KernelRunContext *>(shadow.context->GetContext());
  run->values[shadow.tensor_count + static_cast<size_t>(OpExecuteLaunchInputIndex::kFwkData)] =
      reinterpret_cast<AsyncAnyValue *>(&shadow.framework_data);
  const auto status = func(reinterpret_cast<OpExecuteLaunchContext *>(shadow.context));
  if (status != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "[FxExecuteOpLaunch] launch returned " << status;
  }
  return status;
}

FXRT_EXPORT_KERNEL_RUN(FxExecuteOpPrepare, FxExecuteOpPrepare)
FXRT_EXPORT_KERNEL_RUN(FxExecuteOpLaunch, FxExecuteOpLaunch)

}  // namespace
}  // namespace kernel
}  // namespace gert
