/**
 * Fx-owned dual-stage Add: wrap aclnnAddGetWorkspaceSize / aclnnAdd and patch
 * OpImplSpaceRegistryV2 so the shared ExecuteOpPrepare/Launch RT kernels can run Add.
 */

#include "opp/add_dual_stage_impl.h"

#include "common/logger.h"

#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "register/op_impl_registry.h"

namespace fx_converter {
namespace {

struct AddOpApiParams {
  aclOpExecutor *executor = nullptr;
};

struct AddAclArgs {
  aclTensor *x1 = nullptr;
  aclTensor *x2 = nullptr;
  aclTensor *y = nullptr;
  aclScalar *alpha = nullptr;
  float alpha_storage = 1.0f;
  int64_t alpha_int = 1;

  ~AddAclArgs() { Release(); }

  void Release() {
    DestroyAclTensor(x1);
    DestroyAclTensor(x2);
    DestroyAclTensor(y);
    DestroyAclScalar(alpha);
    x1 = x2 = y = nullptr;
    alpha = nullptr;
  }

  bool Valid() const { return x1 != nullptr && x2 != nullptr && y != nullptr && alpha != nullptr; }
};

bool FetchAddTensors(gert::OpExecutePrepareContext *ctx, const gert::Tensor **x1, const gert::Tensor **x2,
                     const gert::Tensor **y) {
  *x1 = ctx->GetInputTensor(0);
  *x2 = ctx->GetInputTensor(1);
  *y = ctx->GetOutputTensor(0);
  if (*x1 == nullptr || *x2 == nullptr || *y == nullptr) {
    RT_GLOG(ERROR) << "FxAddPrepare: missing input/output tensor";
    return false;
  }
  return true;
}

bool BuildAddAclArgs(gert::OpExecutePrepareContext *ctx, const gert::Tensor *x1, const gert::Tensor *x2,
                     const gert::Tensor *y, AddAclArgs *args) {
  args->x1 = ToAclTensor(x1);
  args->x2 = ToAclTensor(x2);
  args->y = ToAclTensor(y);
  const auto *attrs = ctx->GetAttrs();
  const float *alpha = attrs == nullptr ? nullptr : attrs->GetFloat(0);
  const int64_t *alpha_int = attrs == nullptr ? nullptr : attrs->GetInt(1);
  if (alpha != nullptr) args->alpha_storage = *alpha;
  if (alpha_int != nullptr) args->alpha_int = *alpha_int;
  const auto dtype = y->GetDataType();
  const bool floating = dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
  args->alpha =
    floating ? aclnn_api::aclCreateScalar(&args->alpha_storage, ACL_FLOAT) : aclnn_api::aclCreateScalar(&args->alpha_int, ACL_INT64);
  if (!args->Valid()) {
    RT_GLOG(ERROR) << "FxAddPrepare: aclCreateTensor/Scalar failed";
    args->Release();
    return false;
  }
  return true;
}

bool RunAddGetWorkspace(AddAclArgs *args, AddOpApiParams *params, uint64_t *workspace_size) {
  const aclnnStatus st =
      aclnn_api::Call(aclnn_api::aclnnAddGetWorkspaceSize, static_cast<const aclTensor *>(args->x1),
          static_cast<const aclTensor *>(args->x2), static_cast<const aclScalar *>(args->alpha), args->y,
          workspace_size, &params->executor);
  args->Release();
  if (st != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxAddPrepare: aclnnAddGetWorkspaceSize failed, status=" << st;
    return false;
  }
  return true;
}

bool CommitPrepareOutputs(gert::OpExecutePrepareContext *ctx, AddOpApiParams *params, uint64_t workspace_size) {
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxAddPrepare: SetOpApiParams failed";
    delete params;
    return false;
  }
  if (ctx->SetWorkspaceSizes({workspace_size}) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxAddPrepare: SetWorkspaceSizes failed";
    return false;
  }
  return true;
}

ge::graphStatus AddPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Tensor *x1 = nullptr;
  const gert::Tensor *x2 = nullptr;
  const gert::Tensor *y = nullptr;
  if (!FetchAddTensors(ctx, &x1, &x2, &y)) {
    return ge::GRAPH_FAILED;
  }

  AddAclArgs args;
  if (!BuildAddAclArgs(ctx, x1, x2, y, &args)) {
    return ge::GRAPH_FAILED;
  }

  auto *params = new (std::nothrow) AddOpApiParams();
  if (params == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  if (!RunAddGetWorkspace(&args, params, &workspace_size)) {
    delete params;
    return ge::GRAPH_FAILED;
  }
  return CommitPrepareOutputs(ctx, params, workspace_size) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

bool ResolveWorkspace(gert::OpExecuteLaunchContext *ctx, void **workspace, uint64_t *workspace_size) {
  *workspace = nullptr;
  *workspace_size = 0;
  const auto *ws_sizes = ctx->GetWorkspaceSizes();
  if (ws_sizes != nullptr && ws_sizes->GetSize() > 0) {
    *workspace_size = static_cast<uint64_t>(ws_sizes->GetData()[0]);
  }
  if (*workspace_size == 0) {
    return true;
  }
  const auto *ws_addrs = ctx->GetWorkspaceAddrs();
  if (ws_addrs == nullptr || ws_addrs->GetSize() == 0 || ws_addrs->GetData()[0] == nullptr) {
    RT_GLOG(ERROR) << "FxAddLaunch: missing workspace addr";
    return false;
  }
  *workspace = ws_addrs->GetData()[0]->GetAddr();
  return true;
}

ge::graphStatus AddLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *params = reinterpret_cast<AddOpApiParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxAddLaunch: missing OpApiParams/executor";
    return ge::GRAPH_FAILED;
  }

  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  if (!ResolveWorkspace(ctx, &workspace, &workspace_size)) {
    return ge::GRAPH_FAILED;
  }

  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnAdd, workspace, workspace_size, params->executor,
      ctx->GetStream());
  if (st != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxAddLaunch: aclnnAdd failed, status=" << st;
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

bool InstallFxAddDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) {
    return false;
  }
  // GetOpImpl is const; puncture patches the live entry in-place (same pattern as GE UT stubs).
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("Add"));
  if (funcs == nullptr) {
    RT_GLOG(ERROR) << "InstallFxAddDualStage: GetOpImpl(Add) returned null";
    return false;
  }
  funcs->op_execute_prepare_func = AddPrepare;
  funcs->op_execute_launch_func = AddLaunch;
  RT_GLOG(INFO) << "Fx dual-stage installed: type=Add prepare=" << reinterpret_cast<const void *>(AddPrepare)
                << " launch=" << reinterpret_cast<const void *>(AddLaunch);
  return true;
}

}  // namespace fx_converter
