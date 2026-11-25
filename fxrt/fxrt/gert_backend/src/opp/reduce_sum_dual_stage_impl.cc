/**
 * Fx-owned dual-stage ReduceSum: wrap aclnnReduceSumGetWorkspaceSize / aclnnReduceSum.
 * Inputs: x (device), axes (host INT64 vector). Attr0 keep_dims; dtype from out tensor.
 */

#include "opp/reduce_sum_dual_stage_impl.h"

#include <vector>

#include "common/logger.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "register/op_impl_registry.h"

namespace fx_converter {
namespace {

struct ReduceSumOpApiParams {
  aclOpExecutor *executor = nullptr;
};

struct ReduceSumAclArgs {
  aclTensor *self = nullptr;
  aclTensor *out = nullptr;
  aclIntArray *dims = nullptr;

  ~ReduceSumAclArgs() { Release(); }

  void Release() {
    DestroyAclTensor(self);
    DestroyAclTensor(out);
    if (dims != nullptr) {
      (void)aclnn_api::aclDestroyIntArray(dims);
      dims = nullptr;
    }
    self = out = nullptr;
  }

  bool Valid() const { return self != nullptr && out != nullptr && dims != nullptr; }
};

bool FetchReduceSumTensors(gert::OpExecutePrepareContext *ctx, const gert::Tensor **self,
                           const gert::Tensor **axes, const gert::Tensor **out) {
  *self = ctx->GetInputTensor(0);
  *axes = ctx->GetInputTensor(1);
  *out = ctx->GetOutputTensor(0);
  if (*self == nullptr || *axes == nullptr || *out == nullptr) {
    RT_GLOG(ERROR) << "FxReduceSumPrepare: missing input/output tensor";
    return false;
  }
  return true;
}

bool ReadKeepDims(gert::OpExecutePrepareContext *ctx, bool *keep_dims) {
  *keep_dims = false;
  const gert::RuntimeAttrs *attrs = ctx->GetAttrs();
  if (attrs == nullptr) {
    return true;
  }
  // gert.source_op attrs write keep_dims as attr0 (GE ReduceSum has only that attr).
  const bool *kd = attrs->GetBool(0);
  if (kd != nullptr) {
    *keep_dims = *kd;
  }
  return true;
}

bool AxesToInt64Vector(const gert::Tensor *axes, std::vector<int64_t> *dims) {
  if (axes == nullptr || dims == nullptr) {
    return false;
  }
  const auto &shape = axes->GetStorageShape();
  size_t n = 0;
  if (shape.GetDimNum() == 0) {
    n = 1;
  } else if (shape.GetDimNum() == 1) {
    n = static_cast<size_t>(shape.GetDim(0));
  } else {
    RT_GLOG(ERROR) << "FxReduceSumPrepare: axes must be rank-0/1 INT64";
    return false;
  }
  dims->resize(n);
  if (n == 0) return true;
  if (axes->GetAddr() == nullptr) return false;
  if (axes->GetDataType() == ge::DT_INT32) {
    const auto *src = static_cast<const int32_t *>(axes->GetAddr());
    for (size_t i = 0; i < n; ++i) {
      (*dims)[i] = src[i];
    }
  } else {
    const auto *src = static_cast<const int64_t *>(axes->GetAddr());
    for (size_t i = 0; i < n; ++i) {
      (*dims)[i] = src[i];
    }
  }
  return true;
}

bool BuildReduceSumAclArgs(const gert::Tensor *self, const gert::Tensor *axes, const gert::Tensor *out,
                           ReduceSumAclArgs *args) {
  std::vector<int64_t> dim_vals;
  if (!AxesToInt64Vector(axes, &dim_vals)) {
    return false;
  }
  args->self = ToAclTensor(self);
  args->out = ToAclTensor(out);
  args->dims = aclnn_api::aclCreateIntArray(dim_vals.data(), dim_vals.size());
  if (!args->Valid()) {
    RT_GLOG(ERROR) << "FxReduceSumPrepare: aclCreateTensor/IntArray failed";
    args->Release();
    return false;
  }
  return true;
}

bool RunReduceSumGetWorkspace(ReduceSumAclArgs *args, bool keep_dims, aclDataType dtype,
                              ReduceSumOpApiParams *params, uint64_t *workspace_size) {
  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnReduceSumGetWorkspaceSize,
      static_cast<const aclTensor *>(args->self), static_cast<const aclIntArray *>(args->dims), keep_dims, dtype,
      args->out, workspace_size, &params->executor);
  args->Release();
  if (st != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxReduceSumPrepare: aclnnReduceSumGetWorkspaceSize failed, status=" << st;
    return false;
  }
  return true;
}

bool CommitPrepareOutputs(gert::OpExecutePrepareContext *ctx, ReduceSumOpApiParams *params,
                          uint64_t workspace_size) {
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxReduceSumPrepare: SetOpApiParams failed";
    delete params;
    return false;
  }
  if (ctx->SetWorkspaceSizes({workspace_size}) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxReduceSumPrepare: SetWorkspaceSizes failed";
    return false;
  }
  return true;
}

ge::graphStatus ReduceSumPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Tensor *self = nullptr;
  const gert::Tensor *axes = nullptr;
  const gert::Tensor *out = nullptr;
  if (!FetchReduceSumTensors(ctx, &self, &axes, &out)) {
    return ge::GRAPH_FAILED;
  }
  bool keep_dims = false;
  (void)ReadKeepDims(ctx, &keep_dims);

  ReduceSumAclArgs args;
  if (!BuildReduceSumAclArgs(self, axes, out, &args)) {
    return ge::GRAPH_FAILED;
  }

  auto *params = new (std::nothrow) ReduceSumOpApiParams();
  if (params == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  const aclDataType dtype = ToAclDataType(out->GetDataType());
  if (!RunReduceSumGetWorkspace(&args, keep_dims, dtype, params, &workspace_size)) {
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
    RT_GLOG(ERROR) << "FxReduceSumLaunch: missing workspace addr";
    return false;
  }
  *workspace = ws_addrs->GetData()[0]->GetAddr();
  return true;
}

ge::graphStatus ReduceSumLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *params = reinterpret_cast<ReduceSumOpApiParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxReduceSumLaunch: missing OpApiParams/executor";
    return ge::GRAPH_FAILED;
  }
  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  if (!ResolveWorkspace(ctx, &workspace, &workspace_size)) {
    return ge::GRAPH_FAILED;
  }
  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnReduceSum, workspace, workspace_size, params->executor,
      ctx->GetStream());
  if (st != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxReduceSumLaunch: aclnnReduceSum failed, status=" << st;
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

bool InstallFxReduceSumDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) {
    return false;
  }
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("ReduceSum"));
  if (funcs == nullptr) {
    RT_GLOG(ERROR) << "InstallFxReduceSumDualStage: GetOpImpl(ReduceSum) returned null";
    return false;
  }
  funcs->op_execute_prepare_func = ReduceSumPrepare;
  funcs->op_execute_launch_func = ReduceSumLaunch;
  RT_GLOG(INFO) << "Fx dual-stage installed: type=ReduceSum prepare="
                << reinterpret_cast<const void *>(ReduceSumPrepare)
                << " launch=" << reinterpret_cast<const void *>(ReduceSumLaunch);
  return true;
}

}  // namespace fx_converter
