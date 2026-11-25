/**
 * Fx-owned dual-stage Concat: wrap aclnnCatGetWorkspaceSize / aclnnCat.
 * Inputs are the packed tensor list. Attr0 is concat_dim (aclnnCat's dim, negative allowed).
 */

#include "opp/concat_dual_stage_impl.h"

#include <new>
#include <vector>

#include "common/logger.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"
#include "register/op_impl_registry.h"

namespace fx_converter {
namespace {

struct ConcatOpApiParams {
  aclOpExecutor *executor = nullptr;
};

struct ConcatAclArgs {
  std::vector<aclTensor *> tensors;
  aclTensorList *list = nullptr;
  aclTensor *out = nullptr;

  ~ConcatAclArgs() { Release(); }

  void Release() {
    if (list != nullptr) {
      aclnn_api::aclDestroyTensorList(list);
      list = nullptr;
      tensors.clear();
    } else {
      for (aclTensor *tensor : tensors) {
        DestroyAclTensor(tensor);
      }
      tensors.clear();
    }
    DestroyAclTensor(out);
    out = nullptr;
  }

  bool Valid() const { return list != nullptr && out != nullptr; }
};

bool ReadConcatDim(gert::OpExecutePrepareContext *ctx, int64_t *dim) {
  const gert::RuntimeAttrs *attrs = ctx->GetAttrs();
  const int64_t *value = (attrs == nullptr) ? nullptr : attrs->GetInt(0);
  if (value == nullptr) {
    RT_GLOG(ERROR) << "FxConcatPrepare: missing concat_dim attr";
    return false;
  }
  *dim = *value;
  return true;
}

bool CollectConcatTensors(gert::OpExecutePrepareContext *ctx, ConcatAclArgs *args) {
  const size_t input_num = ctx->GetComputeNodeInputNum();
  if (input_num < 2U) {
    RT_GLOG(ERROR) << "FxConcatPrepare: expected at least 2 inputs, got " << input_num;
    return false;
  }
  args->tensors.reserve(input_num);
  for (size_t i = 0; i < input_num; ++i) {
    aclTensor *tensor = ToAclTensor(ctx->GetInputTensor(i));
    if (tensor == nullptr) {
      RT_GLOG(ERROR) << "FxConcatPrepare: input " << i << " is null";
      return false;
    }
    args->tensors.push_back(tensor);
  }
  args->list = aclnn_api::aclCreateTensorList(args->tensors.data(), args->tensors.size());
  args->out = ToAclTensor(ctx->GetOutputTensor(0));
  if (!args->Valid()) {
    RT_GLOG(ERROR) << "FxConcatPrepare: aclCreateTensorList/Tensor failed";
    return false;
  }
  return true;
}

bool RunConcatGetWorkspace(ConcatAclArgs *args, int64_t dim, ConcatOpApiParams *params, uint64_t *workspace_size) {
  const aclnnStatus status =
      aclnn_api::Call(aclnn_api::aclnnCatGetWorkspaceSize, static_cast<const aclTensorList *>(args->list), dim,
          args->out, workspace_size, &params->executor);
  args->Release();
  if (status != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxConcatPrepare: aclnnCatGetWorkspaceSize failed, status=" << status;
    return false;
  }
  return true;
}

ge::graphStatus ConcatPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  int64_t dim = 0;
  ConcatAclArgs args;
  if (!ReadConcatDim(ctx, &dim) || !CollectConcatTensors(ctx, &args)) {
    return ge::GRAPH_FAILED;
  }
  auto *params = new (std::nothrow) ConcatOpApiParams();
  if (params == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  if (!RunConcatGetWorkspace(&args, dim, params, &workspace_size)) {
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    delete params;
    return ge::GRAPH_FAILED;
  }
  return ctx->SetWorkspaceSizes({workspace_size});
}

ge::graphStatus ConcatLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  auto *params = static_cast<ConcatOpApiParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxConcatLaunch: missing OpApiParams/executor";
    return ge::GRAPH_PARAM_INVALID;
  }
  const auto *sizes = ctx->GetWorkspaceSizes();
  const uint64_t size = (sizes == nullptr || sizes->GetSize() == 0U) ? 0U : sizes->GetData()[0U];
  void *workspace = nullptr;
  if (size != 0U) {
    const auto *addresses = ctx->GetWorkspaceAddrs();
    if (addresses == nullptr || addresses->GetSize() == 0U || addresses->GetData()[0U] == nullptr) {
      RT_GLOG(ERROR) << "FxConcatLaunch: missing workspace addr";
      return ge::GRAPH_PARAM_INVALID;
    }
    workspace = addresses->GetData()[0U]->GetAddr();
  }
  const aclnnStatus status = aclnn_api::Call(aclnn_api::aclnnCat, workspace, size, params->executor,
      ctx->GetStream());
  if (status != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxConcatLaunch: aclnnCat failed, status=" << status;
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

bool InstallFxConcatDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) {
    return false;
  }
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("Concat"));
  if (funcs == nullptr) {
    RT_GLOG(ERROR) << "InstallFxConcatDualStage: GetOpImpl(Concat) returned null";
    return false;
  }
  funcs->op_execute_prepare_func = ConcatPrepare;
  funcs->op_execute_launch_func = ConcatLaunch;
  RT_GLOG(INFO) << "Fx dual-stage installed: type=Concat prepare=" << reinterpret_cast<const void *>(ConcatPrepare)
                << " launch=" << reinterpret_cast<const void *>(ConcatLaunch);
  return true;
}

}  // namespace fx_converter
