#include "opp/where_dual_stage_impl.h"

#include <array>
#include <new>

#include "common/logger.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"

namespace fx_converter {
namespace {
struct WhereParams {
  aclOpExecutor *executor = nullptr;
};

ge::graphStatus WherePrepare(gert::OpExecutePrepareContext *context) {
  if (context == nullptr || context->GetComputeNodeInputNum() != 3U) return ge::GRAPH_PARAM_INVALID;
  std::array<aclTensor *, 4U> tensors{};
  bool valid = true;
  for (size_t i = 0; i < 3U; ++i) {
    tensors[i] = ToAclTensor(context->GetInputTensor(i));
    valid = valid && tensors[i] != nullptr;
  }
  tensors[3U] = ToAclTensor(context->GetOutputTensor(0U));
  valid = valid && tensors[3U] != nullptr;
  auto *params = new (std::nothrow) WhereParams();
  uint64_t workspace_size = 0U;
  const auto status = valid && params != nullptr
      ? aclnn_api::Call(aclnn_api::aclnnSWhereGetWorkspaceSize, static_cast<const aclTensor *>(tensors[0U]),
          static_cast<const aclTensor *>(tensors[1U]), static_cast<const aclTensor *>(tensors[2U]), tensors[3U],
          &workspace_size, &params->executor)
      : static_cast<aclnnStatus>(-1);
  for (auto *tensor : tensors) DestroyAclTensor(tensor);
  if (status != kAclnnSuccess || params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxWherePrepare: aclnnSWhereGetWorkspaceSize failed, status=" << status;
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (context->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    delete params;
    return ge::GRAPH_FAILED;
  }
  return context->SetWorkspaceSizes({workspace_size});
}

ge::graphStatus WhereLaunch(gert::OpExecuteLaunchContext *context) {
  if (context == nullptr) return ge::GRAPH_PARAM_INVALID;
  auto *params = static_cast<WhereParams *>(context->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) return ge::GRAPH_PARAM_INVALID;
  const auto *sizes = context->GetWorkspaceSizes();
  const uint64_t size = sizes == nullptr || sizes->GetSize() == 0U ? 0U : sizes->GetData()[0U];
  void *workspace = nullptr;
  if (size != 0U) {
    const auto *addresses = context->GetWorkspaceAddrs();
    if (addresses == nullptr || addresses->GetSize() == 0U || addresses->GetData()[0U] == nullptr) {
      return ge::GRAPH_PARAM_INVALID;
    }
    workspace = addresses->GetData()[0U]->GetAddr();
  }
  return aclnn_api::Call(aclnn_api::aclnnSWhere, workspace, size, params->executor, context->GetStream()) == kAclnnSuccess
      ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
}  // namespace

bool InstallFxWhereDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) return false;
  auto *functions = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("SelectV2"));
  if (functions == nullptr) return false;
  functions->op_execute_prepare_func = WherePrepare;
  functions->op_execute_launch_func = WhereLaunch;
  return true;
}
}  // namespace fx_converter
