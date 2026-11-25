/**
 * Fx-owned dual-stage GatherElements: wrap aclnnGatherGetWorkspaceSize / aclnnGather.
 * Inputs: x, index (device). Attr0 dim (Int). Covers bool / int64 index that Path C V2 bins miss.
 */

#include "opp/gather_dual_stage_impl.h"

#include "common/logger.h"

#include "exe_graph/graph_builder_utils.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "register/op_impl_registry.h"

namespace fx_converter {
namespace {

struct GatherOpApiParams {
  aclOpExecutor *executor = nullptr;
};

struct GatherAclArgs {
  aclTensor *self = nullptr;
  aclTensor *index = nullptr;
  aclTensor *out = nullptr;

  ~GatherAclArgs() { Release(); }

  void Release() {
    DestroyAclTensor(self);
    DestroyAclTensor(index);
    DestroyAclTensor(out);
    self = index = out = nullptr;
  }

  bool Valid() const { return self != nullptr && index != nullptr && out != nullptr; }
};

bool FetchGatherTensors(gert::OpExecutePrepareContext *ctx, const gert::Tensor **self, const gert::Tensor **index,
                        const gert::Tensor **out) {
  *self = ctx->GetInputTensor(0);
  *index = ctx->GetInputTensor(1);
  *out = ctx->GetOutputTensor(0);
  if (*self == nullptr || *index == nullptr || *out == nullptr) {
    RT_GLOG(ERROR) << "FxGatherPrepare: missing input/output tensor";
    return false;
  }
  return true;
}

bool ReadGatherDim(gert::OpExecutePrepareContext *ctx, int64_t *dim) {
  *dim = 0;
  const gert::RuntimeAttrs *attrs = ctx->GetAttrs();
  if (attrs == nullptr) {
    return true;
  }
  const int64_t *dim_attr = attrs->GetInt(0);
  if (dim_attr != nullptr) {
    *dim = *dim_attr;
  }
  return true;
}

bool BuildGatherAclArgs(const gert::Tensor *self, const gert::Tensor *index, const gert::Tensor *out,
                        GatherAclArgs *args) {
  args->self = ToAclTensor(self);
  // aten.gather output shape is the logical index shape. Reconstruct a
  // stride-0 descriptor when the compact broadcast index stayed a ViewAlias.
  args->index = ToAclTensorBroadcastTo(index, ShapeDims(*out));
  args->out = ToAclTensorWithDtype(out, self->GetDataType());
  if (!args->Valid()) {
    RT_GLOG(ERROR) << "FxGatherPrepare: aclCreateTensor failed";
    args->Release();
    return false;
  }
  return true;
}

bool RunGatherGetWorkspace(const gert::Tensor *self_t, const gert::Tensor *index_t,
                        const gert::Tensor *out_t, GatherAclArgs *args, int64_t dim,
                        GatherOpApiParams *params, uint64_t *workspace_size) {
  const aclnnStatus st =
      aclnn_api::Call(aclnn_api::aclnnGatherGetWorkspaceSize, static_cast<const aclTensor *>(args->self), dim,
          static_cast<const aclTensor *>(args->index), args->out, workspace_size, &params->executor);
  args->Release();
  if (st != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxGatherPrepare: aclnnGatherGetWorkspaceSize failed, status=" << st;
    return false;
  }
  return true;
}

bool CommitPrepareOutputs(gert::OpExecutePrepareContext *ctx, GatherOpApiParams *params, uint64_t workspace_size) {
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxGatherPrepare: SetOpApiParams failed";
    delete params;
    return false;
  }
  if (ctx->SetWorkspaceSizes({workspace_size}) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxGatherPrepare: SetWorkspaceSizes failed";
    return false;
  }
  return true;
}

ge::graphStatus GatherPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Tensor *self = nullptr;
  const gert::Tensor *index = nullptr;
  const gert::Tensor *out = nullptr;
  if (!FetchGatherTensors(ctx, &self, &index, &out)) {
    return ge::GRAPH_FAILED;
  }
  int64_t dim = 0;
  (void)ReadGatherDim(ctx, &dim);
  if (dim < 0) {
    dim += static_cast<int64_t>(self->GetStorageShape().GetDimNum());
  }

  GatherAclArgs args;
  if (!BuildGatherAclArgs(self, index, out, &args)) {
    return ge::GRAPH_FAILED;
  }

  auto *params = new (std::nothrow) GatherOpApiParams();
  if (params == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  if (!RunGatherGetWorkspace(self, index, out, &args, dim, params, &workspace_size)) {
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
    RT_GLOG(ERROR) << "FxGatherLaunch: missing workspace addr";
    return false;
  }
  *workspace = ws_addrs->GetData()[0]->GetAddr();
  return true;
}

ge::graphStatus GatherLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *params = reinterpret_cast<GatherOpApiParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxGatherLaunch: missing OpApiParams/executor";
    return ge::GRAPH_FAILED;
  }
  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  if (!ResolveWorkspace(ctx, &workspace, &workspace_size)) {
    return ge::GRAPH_FAILED;
  }
  // workspace_size==0 is valid (small bool gather). Skipping the launch left the
  // output uninitialized (ETA mask gather dim=-1). aclnnGather accepts a null
  // workspace when GetWorkspaceSize returned 0, same as aclnnAdd / aclnnReduceSum.
  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnGather, workspace, workspace_size, params->executor,
      ctx->GetStream());
  if (st != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxGatherLaunch: aclnnGather failed, status=" << st;
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

bool InstallFxGatherDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) {
    return false;
  }
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("GatherElements"));
  if (funcs == nullptr) {
    RT_GLOG(ERROR) << "InstallFxGatherDualStage: GetOpImpl(GatherElements) returned null";
    return false;
  }
  funcs->op_execute_prepare_func = GatherPrepare;
  funcs->op_execute_launch_func = GatherLaunch;
  RT_GLOG(INFO) << "Fx dual-stage installed: type=GatherElements prepare="
                << reinterpret_cast<const void *>(GatherPrepare)
                << " launch=" << reinterpret_cast<const void *>(GatherLaunch);
  return true;
}

}  // namespace fx_converter
