#include "opp/mutation_dual_stage_impl.h"

#include <new>
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"
#include "register/op_impl_registry.h"

namespace fx_converter {
namespace {
struct MutationParams { aclOpExecutor *executor = nullptr; };

template <bool Fill>
ge::graphStatus MutationPrepare(gert::OpExecutePrepareContext *context) {
  if (context == nullptr) return ge::GRAPH_PARAM_INVALID;
  const auto *target = context->GetInputTensor(0U);
  if (target == nullptr) return ge::GRAPH_PARAM_INVALID;
  auto *self = ToAclTensor(target);
  if (self == nullptr) return ge::GRAPH_PARAM_INVALID;
  auto *params = new (std::nothrow) MutationParams();
  if (params == nullptr) { DestroyAclTensor(self); return ge::GRAPH_FAILED; }
  uint64_t workspace = 0U;
  aclnnStatus status;
  if constexpr (Fill) {
    const auto *attrs = context->GetAttrs();
    const auto *value = attrs == nullptr ? nullptr : attrs->GetFloat(0U);
    if (value == nullptr) { delete params; DestroyAclTensor(self); return ge::GRAPH_PARAM_INVALID; }
    float scalar_value = *value;
    auto *scalar = aclnn_api::aclCreateScalar(&scalar_value, ACL_FLOAT);
    if (scalar == nullptr) { delete params; DestroyAclTensor(self); return ge::GRAPH_FAILED; }
    status = aclnn_api::Call(aclnn_api::aclnnInplaceFillScalarGetWorkspaceSize, self,
        static_cast<const aclScalar *>(scalar), &workspace, &params->executor);
    DestroyAclScalar(scalar);
  } else {
    auto *source = ToAclTensor(context->GetInputTensor(1U));
    if (source == nullptr) { delete params; DestroyAclTensor(self); return ge::GRAPH_PARAM_INVALID; }
    status = aclnn_api::Call(aclnn_api::aclnnInplaceCopyGetWorkspaceSize, self,
        static_cast<const aclTensor *>(source), &workspace, &params->executor);
    DestroyAclTensor(source);
  }
  DestroyAclTensor(self);
  if (status != kAclnnSuccess || params->executor == nullptr) { delete params; return ge::GRAPH_FAILED; }
  if (context->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    delete params; return ge::GRAPH_FAILED;
  }
  return context->SetWorkspaceSizes({workspace});
}

template <bool Fill>
ge::graphStatus MutationLaunch(gert::OpExecuteLaunchContext *context) {
  if (context == nullptr) return ge::GRAPH_PARAM_INVALID;
  auto *params = static_cast<MutationParams *>(context->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) return ge::GRAPH_PARAM_INVALID;
  const auto *sizes = context->GetWorkspaceSizes();
  const uint64_t size = sizes == nullptr || sizes->GetSize() == 0 ? 0U : sizes->GetData()[0U];
  void *workspace = nullptr;
  if (size != 0U) {
    const auto *addresses = context->GetWorkspaceAddrs();
    if (addresses == nullptr || addresses->GetSize() == 0U || addresses->GetData()[0U] == nullptr) {
      return ge::GRAPH_PARAM_INVALID;
    }
    workspace = addresses->GetData()[0U]->GetAddr();
  }
  void *address = Fill ? aclnn_api::aclnnInplaceFillScalar : aclnn_api::aclnnInplaceCopy;
  const auto status = aclnn_api::Call(address, workspace, size, params->executor, context->GetStream());
  return status == kAclnnSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

template <bool Fill>
bool Install(const gert::OpImplSpaceRegistryV2Ptr &registry, const char *type) {
  if (registry == nullptr) return false;
  auto *functions = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl(type));
  if (functions == nullptr) return false;
  functions->op_execute_prepare_func = MutationPrepare<Fill>;
  functions->op_execute_launch_func = MutationLaunch<Fill>;
  return true;
}
}  // namespace
bool InstallFxFillDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) { return Install<true>(registry, "Fills"); }
bool InstallFxCopyDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) { return Install<false>(registry, "Assign"); }
}  // namespace fx_converter
