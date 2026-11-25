// Native advanced-index ABI. CANN's Index .o requires indexed_sizes/strides
// and a dynamic input group; it cannot consume ATen's Tensor[] ABI directly.
#include "opp/index_dual_stage_impl.h"

#include <new>
#include <vector>

#include "common/logger.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"

namespace fx_converter {
namespace {
struct IndexParams {
  aclOpExecutor *executor = nullptr;
};

ge::graphStatus IndexPrepare(gert::OpExecutePrepareContext *context) {
  if (context == nullptr || context->GetComputeNodeInputNum() < 2U) {
    return ge::GRAPH_PARAM_INVALID;
  }
  auto *self = ToAclTensor(context->GetInputTensor(0U));
  auto *output = ToAclTensor(context->GetOutputTensor(0U));
  std::vector<aclTensor *> indices;
  bool valid = self != nullptr && output != nullptr;
  for (size_t i = 1U; i < context->GetComputeNodeInputNum(); ++i) {
    auto *index = ToAclTensor(context->GetInputTensor(i));
    valid = valid && index != nullptr;
    indices.push_back(index);
  }
  auto *list = valid ? aclnn_api::aclCreateTensorList(indices.data(), indices.size()) : nullptr;
  auto *params = new (std::nothrow) IndexParams();
  uint64_t workspace_size = 0U;
  const auto status = list != nullptr && params != nullptr
      ? aclnn_api::Call(aclnn_api::aclnnIndexGetWorkspaceSize, static_cast<const aclTensor *>(self),
          static_cast<const aclTensorList *>(list), output, &workspace_size, &params->executor)
      : static_cast<aclnnStatus>(-1);
  // TensorList owns its aclTensor descriptors after aclCreateTensorList succeeds.
  if (list != nullptr) {
    aclnn_api::aclDestroyTensorList(list);
  } else {
    for (auto *index : indices) DestroyAclTensor(index);
  }
  DestroyAclTensor(self);
  DestroyAclTensor(output);
  if (status != kAclnnSuccess || params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxIndexPrepare: aclnnIndexGetWorkspaceSize failed, status=" << status;
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (context->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    delete params;
    return ge::GRAPH_FAILED;
  }
  return context->SetWorkspaceSizes({workspace_size});
}

ge::graphStatus IndexLaunch(gert::OpExecuteLaunchContext *context) {
  if (context == nullptr) return ge::GRAPH_PARAM_INVALID;
  auto *params = static_cast<IndexParams *>(context->GetOpApiParams());
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
  return aclnn_api::Call(aclnn_api::aclnnIndex, workspace, size, params->executor, context->GetStream()) == kAclnnSuccess
      ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
}  // namespace

bool InstallFxIndexDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) return false;
  auto *functions = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("Index"));
  if (functions == nullptr) return false;
  functions->op_execute_prepare_func = IndexPrepare;
  functions->op_execute_launch_func = IndexLaunch;
  return true;
}
}  // namespace fx_converter
