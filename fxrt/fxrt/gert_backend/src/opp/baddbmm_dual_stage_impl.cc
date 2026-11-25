/**
 * Fx-owned dual-stage Baddbmm: wrap aclnnBaddbmmGetWorkspaceSize / aclnnBaddbmm.
 *
 * OPP has no "Baddbmm" OpImpl entry, so the catalog row mounts this wrapper on
 * "BatchMatMulV2" -- a distinct entry from the "BatchMatMul" that
 * InstallFxBatchMatMulDualStage already replaces.
 *
 * `self` broadcasts over batch1@batch2 and may arrive as an expanded (stride-0)
 * view, so every tensor goes through the runtime TensorV2 layout rather than a
 * densified copy.
 *
 * cubeMathType must follow torch.npu.matmul.allow_hf32 exactly like the MatMul
 * wrapper; torch_npu's baddbmm routes through get_cube_math_type_with_passthrough(),
 * whose non-passthrough branch is the same allow_hf32 answer.
 */

#include "opp/baddbmm_dual_stage_impl.h"

#include <cstdint>
#include <new>

#include "common/logger.h"
#include "opp/aclnn_tensor_utils.h"
#include "opp/aclnn_api.h"

#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "register/op_impl_registry.h"

namespace fx_converter {
namespace {

// Attr order mirrors the catalog row in baddbmm_op_lowering.cc.
enum BaddbmmAttrIndex : size_t {
  kBeta = 0,
  kAlpha,
};

struct BaddbmmOpApiParams {
  aclOpExecutor *executor = nullptr;
};

struct BaddbmmAclArgs {
  aclTensor *self = nullptr;
  aclTensor *batch1 = nullptr;
  aclTensor *batch2 = nullptr;
  aclTensor *out = nullptr;

  ~BaddbmmAclArgs() { Release(); }

  void Release() {
    DestroyAclTensor(self);
    DestroyAclTensor(batch1);
    DestroyAclTensor(batch2);
    DestroyAclTensor(out);
    self = batch1 = batch2 = out = nullptr;
  }

  bool Valid() const { return self != nullptr && batch1 != nullptr && batch2 != nullptr && out != nullptr; }
};

// aclnn cubeMathType: KEEP_DTYPE=0, USE_HF32=3; allow_hf32[1] is the matmul switch.
constexpr int8_t kCubeKeepDtype = 0;
constexpr int8_t kCubeUseHf32 = 3;

int8_t CubeMathTypeFromPrepare(gert::OpExecutePrepareContext *ctx) {
  const char *hf32 = ctx != nullptr ? ctx->GetAllowHf32() : nullptr;
  if (hf32 != nullptr && hf32[1] == '1') {
    return kCubeUseHf32;
  }
  return kCubeKeepDtype;
}

bool ResolveWorkspace(gert::OpExecuteLaunchContext *ctx, void **workspace, uint64_t *workspace_size) {
  *workspace = nullptr;
  *workspace_size = 0;
  const auto *ws_sizes = ctx->GetWorkspaceSizes();
  if (ws_sizes != nullptr && ws_sizes->GetSize() > 0) {
    *workspace_size = static_cast<uint64_t>(ws_sizes->GetData()[0]);
  }
  if (*workspace_size == 0) return true;
  const auto *ws_addrs = ctx->GetWorkspaceAddrs();
  if (ws_addrs == nullptr || ws_addrs->GetSize() == 0 || ws_addrs->GetData()[0] == nullptr) {
    RT_GLOG(ERROR) << "FxBaddbmmLaunch: missing workspace addr";
    return false;
  }
  *workspace = ws_addrs->GetData()[0]->GetAddr();
  return true;
}

ge::graphStatus BaddbmmPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) return ge::GRAPH_FAILED;
  const gert::Tensor *self = ctx->GetInputTensor(0);
  const gert::Tensor *batch1 = ctx->GetInputTensor(1);
  const gert::Tensor *batch2 = ctx->GetInputTensor(2);
  const gert::Tensor *out = ctx->GetOutputTensor(0);
  if (self == nullptr || batch1 == nullptr || batch2 == nullptr || out == nullptr) {
    RT_GLOG(ERROR) << "FxBaddbmmPrepare: missing input/output tensor";
    return ge::GRAPH_FAILED;
  }
  BaddbmmAclArgs args;
  args.self = ToAclTensor(self);
  args.batch1 = ToAclTensor(batch1);
  args.batch2 = ToAclTensor(batch2);
  args.out = ToAclTensor(out);
  if (!args.Valid()) {
    RT_GLOG(ERROR) << "FxBaddbmmPrepare: aclCreateTensor failed";
    return ge::GRAPH_FAILED;
  }

  const auto *attrs = ctx->GetAttrs();
  const float *beta_attr = attrs == nullptr ? nullptr : attrs->GetFloat(kBeta);
  const float *alpha_attr = attrs == nullptr ? nullptr : attrs->GetFloat(kAlpha);
  // beta/alpha have catalog defaults of 1, so a missing slot still means 1.0.
  float beta_value = beta_attr != nullptr ? *beta_attr : 1.0f;
  float alpha_value = alpha_attr != nullptr ? *alpha_attr : 1.0f;
  aclScalar *beta = aclnn_api::aclCreateScalar(&beta_value, ACL_FLOAT);
  aclScalar *alpha = aclnn_api::aclCreateScalar(&alpha_value, ACL_FLOAT);
  if (beta == nullptr || alpha == nullptr) {
    DestroyAclScalar(beta);
    DestroyAclScalar(alpha);
    RT_GLOG(ERROR) << "FxBaddbmmPrepare: aclCreateScalar failed";
    return ge::GRAPH_FAILED;
  }

  auto *params = new (std::nothrow) BaddbmmOpApiParams();
  if (params == nullptr) {
    DestroyAclScalar(beta);
    DestroyAclScalar(alpha);
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnBaddbmmGetWorkspaceSize,
      static_cast<const aclTensor *>(args.self), static_cast<const aclTensor *>(args.batch1),
      static_cast<const aclTensor *>(args.batch2), static_cast<const aclScalar *>(beta),
      static_cast<const aclScalar *>(alpha), args.out, CubeMathTypeFromPrepare(ctx), &workspace_size,
      &params->executor);
  DestroyAclScalar(beta);
  DestroyAclScalar(alpha);
  args.Release();
  if (st != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxBaddbmmPrepare: aclnnBaddbmmGetWorkspaceSize failed, status=" << st;
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxBaddbmmPrepare: SetOpApiParams failed";
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetWorkspaceSizes({workspace_size}) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxBaddbmmPrepare: SetWorkspaceSizes failed";
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaddbmmLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) return ge::GRAPH_FAILED;
  auto *params = reinterpret_cast<BaddbmmOpApiParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxBaddbmmLaunch: missing OpApiParams/executor";
    return ge::GRAPH_FAILED;
  }
  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  if (!ResolveWorkspace(ctx, &workspace, &workspace_size)) return ge::GRAPH_FAILED;
  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnBaddbmm, workspace, workspace_size, params->executor,
      ctx->GetStream());
  if (st != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxBaddbmmLaunch: aclnnBaddbmm failed, status=" << st;
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

bool InstallFxBaddbmmDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) return false;
  // Mount point: OPP's BatchMatMulV2 entry. Nothing under "Baddbmm" exists to patch,
  // and BatchMatMulV2 is distinct from the BatchMatMul entry the bmm row already owns.
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("BatchMatMulV2"));
  if (funcs == nullptr) {
    RT_GLOG(ERROR) << "InstallFxBaddbmmDualStage: GetOpImpl(BatchMatMulV2) returned null";
    return false;
  }
  funcs->op_execute_prepare_func = BaddbmmPrepare;
  funcs->op_execute_launch_func = BaddbmmLaunch;
  RT_GLOG(INFO) << "Fx dual-stage installed: type=BatchMatMulV2(baddbmm) prepare="
                << reinterpret_cast<const void *>(BaddbmmPrepare) << " launch="
                << reinterpret_cast<const void *>(BaddbmmLaunch);
  return true;
}

}  // namespace fx_converter
