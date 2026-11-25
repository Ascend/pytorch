#include "opp/bitwise_not_dual_stage_impl.h"

#include <new>

#include "common/logger.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"
#include "register/op_impl_registry.h"

namespace fx_converter {
namespace {

struct BitwiseNotOpApiParams {
  aclOpExecutor *executor = nullptr;
};

struct BitwiseNotAclArgs {
  aclTensor *input = nullptr;
  aclTensor *output = nullptr;

  ~BitwiseNotAclArgs() {
    DestroyAclTensor(input);
    DestroyAclTensor(output);
  }
};

bool IsBitwiseNotDtype(ge::DataType dtype) {
  return dtype == ge::DT_BOOL || dtype == ge::DT_INT8 || dtype == ge::DT_UINT8 || dtype == ge::DT_INT16 ||
         dtype == ge::DT_INT32 || dtype == ge::DT_INT64;
}

ge::graphStatus BitwiseNotPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) return ge::GRAPH_FAILED;
  const auto *input = ctx->GetInputTensor(0);
  const auto *output = ctx->GetOutputTensor(0);
  if (input == nullptr || output == nullptr) {
    RT_GLOG(ERROR) << "FxBitwiseNotPrepare: missing input/output tensor";
    return ge::GRAPH_FAILED;
  }
  if (!IsBitwiseNotDtype(input->GetDataType()) || input->GetDataType() != output->GetDataType() ||
      TensorLogicalShape(input) != TensorLogicalShape(output)) {
    RT_GLOG(ERROR) << "FxBitwiseNotPrepare: unsupported dtype or mismatched output dtype/shape";
    return ge::GRAPH_FAILED;
  }
  BitwiseNotAclArgs args;
  args.input = ToAclTensor(input);
  args.output = ToAclTensor(output);
  if (args.input == nullptr || args.output == nullptr) {
    RT_GLOG(ERROR) << "FxBitwiseNotPrepare: aclCreateTensor failed";
    return ge::GRAPH_FAILED;
  }
  auto *params = new (std::nothrow) BitwiseNotOpApiParams();
  if (params == nullptr) return ge::GRAPH_FAILED;
  uint64_t workspace_size = 0;
  const aclnnStatus status =
    aclnn_api::Call(aclnn_api::aclnnBitwiseNotGetWorkspaceSize, static_cast<const aclTensor *>(args.input),
        args.output, &workspace_size, &params->executor);
  if (status != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxBitwiseNotPrepare: aclnnBitwiseNotGetWorkspaceSize failed, status=" << status;
    delete params;
    return ge::GRAPH_FAILED;
  }
  // Same one-shot params ownership as the existing Add dual-stage adapter.
  // ACL descriptors are released on return, after GetWorkspaceSize captured them.
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxBitwiseNotPrepare: SetOpApiParams failed";
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetWorkspaceSizes({workspace_size}) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxBitwiseNotPrepare: SetWorkspaceSizes failed";
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BitwiseNotLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) return ge::GRAPH_FAILED;
  auto *params = reinterpret_cast<BitwiseNotOpApiParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxBitwiseNotLaunch: missing OpApiParams/executor";
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  const auto *sizes = ctx->GetWorkspaceSizes();
  if (sizes != nullptr && sizes->GetSize() > 0) workspace_size = static_cast<uint64_t>(sizes->GetData()[0]);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    const auto *addresses = ctx->GetWorkspaceAddrs();
    if (addresses == nullptr || addresses->GetSize() == 0 || addresses->GetData()[0] == nullptr ||
        addresses->GetData()[0]->GetAddr() == nullptr) {
      RT_GLOG(ERROR) << "FxBitwiseNotLaunch: missing workspace address";
      return ge::GRAPH_FAILED;
    }
    workspace = addresses->GetData()[0]->GetAddr();
  }
  // Zero workspace is valid and still requires the launch call.
  const aclnnStatus status = aclnn_api::Call(aclnn_api::aclnnBitwiseNot, workspace, workspace_size,
      params->executor, ctx->GetStream());
  if (status != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxBitwiseNotLaunch: aclnnBitwiseNot failed, status=" << status;
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

bool InstallFxBitwiseNotDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) return false;
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("Invert"));
  if (funcs == nullptr) {
    RT_GLOG(ERROR) << "InstallFxBitwiseNotDualStage: GetOpImpl(Invert) returned null";
    return false;
  }
  funcs->op_execute_prepare_func = BitwiseNotPrepare;
  funcs->op_execute_launch_func = BitwiseNotLaunch;
  RT_GLOG(INFO) << "Fx dual-stage installed: type=Invert prepare=" << reinterpret_cast<const void *>(BitwiseNotPrepare)
                << " launch=" << reinterpret_cast<const void *>(BitwiseNotLaunch);
  return true;
}

}  // namespace fx_converter
