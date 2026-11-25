#include "opp/flash_attention_score_dual_stage_impl.h"

#include <cstring>

#include "common/logger.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"

#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "register/op_impl_registry.h"


namespace fx_converter {
namespace {

struct AttentionParams {
  aclOpExecutor *executor = nullptr;
};

struct AttentionTensors {
  aclTensor *query = nullptr;
  aclTensor *key = nullptr;
  aclTensor *value = nullptr;
  aclTensor *softmax_max = nullptr;
  aclTensor *softmax_sum = nullptr;
  aclTensor *softmax_out = nullptr;
  aclTensor *attention = nullptr;

  void Release() {
    DestroyAclTensor(query);
    DestroyAclTensor(key);
    DestroyAclTensor(value);
    DestroyAclTensor(softmax_max);
    DestroyAclTensor(softmax_sum);
    DestroyAclTensor(softmax_out);
    DestroyAclTensor(attention);
    query = key = value = softmax_max = softmax_sum = softmax_out = attention = nullptr;
  }

  bool Valid() const {
    return query != nullptr && key != nullptr && value != nullptr && softmax_max != nullptr &&
           softmax_sum != nullptr && softmax_out != nullptr && attention != nullptr;
  }
};

bool FetchAttentionTensors(gert::OpExecutePrepareContext *ctx, AttentionTensors *args) {
  args->query = ToAclTensor(ctx->GetInputTensor(0));
  args->key = ToAclTensor(ctx->GetInputTensor(1));
  args->value = ToAclTensor(ctx->GetInputTensor(2));
  args->softmax_max = ToAclTensor(ctx->GetOutputTensor(0));
  args->softmax_sum = ToAclTensor(ctx->GetOutputTensor(1));
  args->softmax_out = ToAclTensor(ctx->GetOutputTensor(2));
  args->attention = ToAclTensor(ctx->GetOutputTensor(3));
  if (!args->Valid()) {
    RT_GLOG(ERROR) << "FxFlashAttentionPrepare: missing query/key/value or one of the four outputs";
    args->Release();
    return false;
  }
  return true;
}

bool ReadAttentionAttrs(const gert::RuntimeAttrs *attrs, double *scale, double *keep, int64_t *head, int64_t *pre,
                        int64_t *next, int64_t *inner, int64_t *sparse, const char **layout) {
  if (attrs == nullptr || attrs->GetFloat(0) == nullptr || attrs->GetFloat(1) == nullptr ||
      attrs->GetInt(2) == nullptr || attrs->GetStr(7) == nullptr) {
    RT_GLOG(ERROR) << "FxFlashAttentionPrepare: missing scale/keep/head/layout attrs";
    return false;
  }
  *scale = *attrs->GetFloat(0);
  *keep = *attrs->GetFloat(1);
  *head = *attrs->GetInt(2);
  *pre = attrs->GetInt(3) != nullptr ? *attrs->GetInt(3) : 2147483647;
  *next = attrs->GetInt(4) != nullptr ? *attrs->GetInt(4) : 2147483647;
  *inner = attrs->GetInt(5) != nullptr ? *attrs->GetInt(5) : 0;
  *sparse = attrs->GetInt(6) != nullptr ? *attrs->GetInt(6) : 0;
  *layout = attrs->GetStr(7);
  return *head > 0 && *layout != nullptr && *layout[0] != '\0';
}

ge::graphStatus AttentionPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  AttentionTensors args;
  if (!FetchAttentionTensors(ctx, &args)) {
    return ge::GRAPH_FAILED;
  }
  double scale = 1.0;
  double keep = 1.0;
  int64_t head = 0;
  int64_t pre = 2147483647;
  int64_t next = 2147483647;
  int64_t inner = 0;
  int64_t sparse = 0;
  const char *layout = nullptr;
  if (!ReadAttentionAttrs(ctx->GetAttrs(), &scale, &keep, &head, &pre, &next, &inner, &sparse, &layout)) {
    args.Release();
    return ge::GRAPH_FAILED;
  }
  char layout_buf[8] = {};
  std::strncpy(layout_buf, layout, sizeof(layout_buf) - 1U);
  auto *params = new (std::nothrow) AttentionParams();
  if (params == nullptr) {
    args.Release();
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  const aclnnStatus status = aclnn_api::Call(aclnn_api::aclnnFlashAttentionScoreGetWorkspaceSize,
      static_cast<const aclTensor *>(args.query), static_cast<const aclTensor *>(args.key),
      static_cast<const aclTensor *>(args.value), static_cast<const aclTensor *>(nullptr),
      static_cast<const aclTensor *>(nullptr), static_cast<const aclTensor *>(nullptr),
      static_cast<const aclTensor *>(nullptr), static_cast<const aclIntArray *>(nullptr), scale, keep, pre, next,
      head, static_cast<char *>(layout_buf), inner, sparse, static_cast<const aclTensor *>(args.softmax_max),
      static_cast<const aclTensor *>(args.softmax_sum), static_cast<const aclTensor *>(args.softmax_out),
      static_cast<const aclTensor *>(args.attention), &workspace_size, &params->executor);
  args.Release();
  if (status != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxFlashAttentionPrepare: GetWorkspaceSize failed, status=" << status;
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetWorkspaceSizes({workspace_size}) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *params = reinterpret_cast<AttentionParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxFlashAttentionLaunch: missing executor";
    return ge::GRAPH_FAILED;
  }
  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  const auto *sizes = ctx->GetWorkspaceSizes();
  const auto *addrs = ctx->GetWorkspaceAddrs();
  if (sizes != nullptr && sizes->GetSize() > 0) {
    workspace_size = static_cast<uint64_t>(sizes->GetData()[0]);
  }
  if (workspace_size != 0) {
    if (addrs == nullptr || addrs->GetSize() == 0 || addrs->GetData()[0] == nullptr ||
        addrs->GetData()[0]->GetAddr() == nullptr) {
      RT_GLOG(ERROR) << "FxFlashAttentionLaunch: missing workspace addr";
      return ge::GRAPH_FAILED;
    }
    workspace = addrs->GetData()[0]->GetAddr();
  }
  const aclnnStatus status = aclnn_api::Call(aclnn_api::aclnnFlashAttentionScore, workspace, workspace_size,
      params->executor, ctx->GetStream());
  if (status != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxFlashAttentionLaunch: aclnnFlashAttentionScore failed, status=" << status;
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

extern "C" gert::OpImplKernelRegistry::OpImplFunctionsV2 *FxCreateOrGetOpImpl(
    gert::OpImplSpaceRegistryV2 *registry, const char *op_type)
    __asm__("_ZN4gert21OpImplSpaceRegistryV217CreateOrGetOpImplEPKc");

bool InstallFxFlashAttentionScoreDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) {
    return false;
  }
  if (registry->GetOpImpl("FlashAttentionScore") == nullptr &&
      FxCreateOrGetOpImpl(registry.get(), "FlashAttentionScore") == nullptr) {
    RT_GLOG(ERROR) << "InstallFxFlashAttentionScoreDualStage: CreateOrGetOpImpl returned null";
    return false;
  }
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("FlashAttentionScore"));
  if (funcs == nullptr) {
    return false;
  }
  funcs->op_execute_prepare_func = AttentionPrepare;
  funcs->op_execute_launch_func = AttentionLaunch;
  return true;
}

}  // namespace fx_converter
