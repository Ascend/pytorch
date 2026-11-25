/**
 * Fx-owned dual-stage MatMul / BatchMatMul.
 *
 * OPP ExecuteOpPrepare maps MatMul onto aclnnMm and ignores CNI transpose_x / adj,
 * so a physical [K,M] view into mm yields zeros (K-axis mismatch, output untouched).
 * This wrapper:
 *   MatMul:        aclnnMatmul on a last-two transposed ND view
 *   BatchMatMul:   aclnnBatchMatMul on a last-two transposed ND view
 * Attrs come from CNI. Trust them when they already match last-two K/M/N.
 * If they disagree, override only when exactly one (t1,t2) pair matches
 * (rectangular, uniquely determined). Square / dynamic / multi-hit cases
 * fail prepare instead of silently picking the first candidate.
 */

#include "opp/matmul_dual_stage_impl.h"

#include <algorithm>
#include <string>
#include <vector>

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

// aclnn cubeMathType: KEEP_DTYPE=0, USE_HF32=3. Must follow torch.npu.matmul.allow_hf32
// via OpExecuteOptions (SharedExecOptions / const_exec_options); hardcoding KEEP_DTYPE
// leaves converter in full fp32 while ETA --hf32=true runs eager in hf32 (~1e-2 mismatch).
constexpr int8_t kCubeKeepDtype = 0;
constexpr int8_t kCubeUseHf32 = 3;

enum class MatMulApi : uint8_t { kMm = 0, kBmm = 1, kMatmul = 2 };

int8_t CubeMathTypeFromPrepare(gert::OpExecutePrepareContext *ctx) {
  const char *hf32 = ctx != nullptr ? ctx->GetAllowHf32() : nullptr;
  // allow_hf32[0]=conv, [1]=matmul; '1' enables.
  if (hf32 != nullptr && hf32[1] == '1') {
    return kCubeUseHf32;
  }
  return kCubeKeepDtype;
}

struct MatMulOpApiParams {
  aclOpExecutor *executor = nullptr;
  MatMulApi api = MatMulApi::kMm;
};

struct MatMulAclArgs {
  aclTensor *x1 = nullptr;
  aclTensor *x2 = nullptr;
  aclTensor *y = nullptr;
  std::string diagnostic;

  ~MatMulAclArgs() { Release(); }

  void Release() {
    DestroyAclTensor(x1);
    DestroyAclTensor(x2);
    DestroyAclTensor(y);
    x1 = x2 = y = nullptr;
  }

  bool Valid() const { return x1 != nullptr && x2 != nullptr && y != nullptr; }
};

bool FetchMmTensors(gert::OpExecutePrepareContext *ctx, const gert::Tensor **x1, const gert::Tensor **x2,
                    const gert::Tensor **y) {
  *x1 = ctx->GetInputTensor(0);
  *x2 = ctx->GetInputTensor(1);
  *y = ctx->GetOutputTensor(0);
  if (*x1 == nullptr || *x2 == nullptr || *y == nullptr) {
    RT_GLOG(ERROR) << "FxMatMulPrepare: missing input/output tensor";
    return false;
  }
  return true;
}

bool ReadBoolAttr(const gert::RuntimeAttrs *attrs, size_t index) {
  if (attrs == nullptr) {
    return false;
  }
  const bool *flag = attrs->GetBool(index);
  return flag != nullptr && *flag;
}

int64_t DimOr(const std::vector<int64_t> &shape, size_t i) {
  return i < shape.size() ? shape[i] : 0;
}

bool LastTwoLayoutMatches(const std::vector<int64_t> &x1, const std::vector<int64_t> &x2,
                          const std::vector<int64_t> &y, bool trans_x1, bool trans_x2) {
  if (x1.size() < 2U || x2.size() < 2U || y.size() < 2U || x1.size() != y.size() || x2.size() != y.size()) {
    return false;
  }
  for (size_t i = 0; i + 2U < y.size(); ++i) {
    if (DimOr(x1, i) != y[i]) {
      return false;
    }
    if (x2[i] != y[i] && x2[i] != 1) {
      return false;
    }
  }
  const size_t r2 = x1.size() - 2U;
  const size_t r1 = x1.size() - 1U;
  const int64_t m = trans_x1 ? x1[r1] : x1[r2];
  const int64_t k1 = trans_x1 ? x1[r2] : x1[r1];
  const int64_t k2 = trans_x2 ? x2[r1] : x2[r2];
  const int64_t n = trans_x2 ? x2[r2] : x2[r1];
  return k1 == k2 && m == y[r2] && n == y[r1];
}

bool UniqueMatchingTranspose(const std::vector<int64_t> &x1, const std::vector<int64_t> &x2,
                             const std::vector<int64_t> &y, bool *out_x1, bool *out_x2) {
  const bool candidates[4][2] = {{false, false}, {true, false}, {false, true}, {true, true}};
  int hits = 0;
  bool hit_x1 = false;
  bool hit_x2 = false;
  for (const auto &cand : candidates) {
    if (!LastTwoLayoutMatches(x1, x2, y, cand[0], cand[1])) {
      continue;
    }
    ++hits;
    hit_x1 = cand[0];
    hit_x2 = cand[1];
  }
  if (hits != 1 || out_x1 == nullptr || out_x2 == nullptr) {
    return false;
  }
  *out_x1 = hit_x1;
  *out_x2 = hit_x2;
  return true;
}

std::string ShapeStr(const std::vector<int64_t> &s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i != 0) {
      out += ",";
    }
    out += std::to_string(s[i]);
  }
  out += "]";
  return out;
}

std::vector<int64_t> MatMulShape(const gert::Tensor *tensor) {
  return TensorLogicalShape(tensor);
}

bool ReadAndResolveTranspose(gert::OpExecutePrepareContext *ctx, const gert::Tensor *x1, const gert::Tensor *x2,
                             const gert::Tensor *y, bool *trans_x1, bool *trans_x2, bool *feed_view_x1,
                             bool *feed_view_x2, size_t attr_x1_index = 0, size_t attr_x2_index = 1) {
  const bool attr_x1 = ReadBoolAttr(ctx->GetAttrs(), attr_x1_index);
  const bool attr_x2 = ReadBoolAttr(ctx->GetAttrs(), attr_x2_index);
  *feed_view_x1 = false;
  *feed_view_x2 = false;
  const auto s1 = MatMulShape(x1);
  const auto s2 = MatMulShape(x2);
  const auto sy = ShapeDims(y);
  // ACLNN Matmul already implements vector promotion and batch broadcasting.
  // The transpose recovery below is only needed for the equal-rank matrix case.
  if (!attr_x1 && !attr_x2 &&
      (s1.size() < 2U || s2.size() < 2U || s1.size() != sy.size() || s2.size() != sy.size() ||
       !std::equal(s1.begin(), s1.end() - 2, s2.begin()))) {
    *trans_x1 = false;
    *trans_x2 = false;
    return true;
  }
  // ES datasets carry only addresses and shapes. For TensorV1, a matching
  // untransposed shape with a CNI transpose attr identifies a transposed feed:
  // its logical dims are already correct, but its stride has been lost.
  const bool untransposed_shapes_match = LastTwoLayoutMatches(s1, s2, sy, false, false);
  const bool v1_feed_x1 = attr_x1 && x1->GetVersion() == gert::kTensorV1 && untransposed_shapes_match;
  const bool v1_feed_x2 = attr_x2 && x2->GetVersion() == gert::kTensorV1 && untransposed_shapes_match;
  if (v1_feed_x1 || v1_feed_x2) {
    *trans_x1 = attr_x1;
    *trans_x2 = attr_x2;
    *feed_view_x1 = v1_feed_x1;
    *feed_view_x2 = v1_feed_x2;
    return true;
  }
  const bool shape_trans_x1 = attr_x1 && !IsTensorLastTwoTransposed(x1);
  const bool shape_trans_x2 = attr_x2 && !IsTensorLastTwoTransposed(x2);
  if (LastTwoLayoutMatches(s1, s2, sy, shape_trans_x1, shape_trans_x2)) {
    *trans_x1 = attr_x1;
    *trans_x2 = attr_x2;
    return true;
  }
  if (UniqueMatchingTranspose(s1, s2, sy, trans_x1, trans_x2)) {
    RT_GLOG(INFO) << "FxMatMulPrepare: override transpose_x1/x2 " << attr_x1 << "/" << attr_x2 << " -> " << *trans_x1
                  << "/" << *trans_x2 << " x1=" << ShapeStr(s1) << " x2=" << ShapeStr(s2) << " y=" << ShapeStr(sy);
    return true;
  }
  RT_GLOG(ERROR) << "FxMatMulPrepare: transpose_x1/x2 " << attr_x1 << "/" << attr_x2
                 << " is not uniquely determined by shapes x1=" << ShapeStr(s1) << " x2=" << ShapeStr(s2)
                 << " y=" << ShapeStr(sy);
  return false;
}

bool BuildMmFamilyArgs(const gert::Tensor *x1, const gert::Tensor *x2, const gert::Tensor *y, bool trans_x1,
                       bool trans_x2, bool feed_view_x1, bool feed_view_x2, MatMulAclArgs *args, const char *tag) {
  // TensorV1 from an ES dataset retains a transposed feed's logical dims but
  // loses its stride. Recreate that runtime view from the persisted CNI attr.
  const auto make_input = [](const gert::Tensor *tensor, bool transposed, bool feed_view) {
    if (!transposed || IsTensorLastTwoTransposed(tensor)) {
      return ToAclTensor(tensor);
    }
    if (feed_view) {
      return ToAclTensorAssumingLastTwoTransposed(tensor);
    }
    return ToAclTensorTransposedLastTwo(tensor);
  };
  args->x1 = make_input(x1, trans_x1, feed_view_x1);
  args->x2 = make_input(x2, trans_x2, feed_view_x2);
  args->y = ToAclTensor(y);
  args->diagnostic = "x1=" + ShapeStr(MatMulShape(x1)) + "(v" +
                     std::to_string(static_cast<int>(x1->GetVersion())) + ",trans=" +
                     std::to_string(trans_x1) + ") x2=" + ShapeStr(MatMulShape(x2)) + "(v" +
                     std::to_string(static_cast<int>(x2->GetVersion())) + ",trans=" +
                     std::to_string(trans_x2) + ") y=" + ShapeStr(ShapeDims(y));
  if (!args->Valid()) {
    RT_GLOG(ERROR) << tag << ": aclCreateTensor failed";
    args->Release();
    return false;
  }
  return true;
}

bool RunMatmulGetWorkspace(MatMulAclArgs *args, MatMulOpApiParams *params, uint64_t *workspace_size,
                           int8_t cube_math_type) {
  params->api = MatMulApi::kMatmul;
  const aclnnStatus st =
      aclnn_api::Call(aclnn_api::aclnnMatmulGetWorkspaceSize, static_cast<const aclTensor *>(args->x1),
          static_cast<const aclTensor *>(args->x2), args->y, cube_math_type, workspace_size, &params->executor);
  args->Release();
  if (st != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxMatmulPrepare: aclnnMatmulGetWorkspaceSize failed, status=" << st << " "
                   << args->diagnostic;
    return false;
  }
  return true;
}

bool RunMmGetWorkspace(MatMulAclArgs *args, MatMulOpApiParams *params, uint64_t *workspace_size,
                       int8_t cube_math_type) {
  params->api = MatMulApi::kMm;
  const aclnnStatus st =
      aclnn_api::Call(aclnn_api::aclnnMmGetWorkspaceSize, static_cast<const aclTensor *>(args->x1),
          static_cast<const aclTensor *>(args->x2), args->y, cube_math_type, workspace_size, &params->executor);
  args->Release();
  if (st != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxMatMulPrepare: aclnnMmGetWorkspaceSize failed, status=" << st;
    return false;
  }
  return true;
}

bool RunBmmGetWorkspace(MatMulAclArgs *args, MatMulOpApiParams *params, uint64_t *workspace_size,
                        int8_t cube_math_type) {
  params->api = MatMulApi::kBmm;
  const aclnnStatus st =
      aclnn_api::Call(aclnn_api::aclnnBatchMatMulGetWorkspaceSize, static_cast<const aclTensor *>(args->x1),
          static_cast<const aclTensor *>(args->x2), args->y, cube_math_type, workspace_size, &params->executor);
  args->Release();
  if (st != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxBatchMatMulPrepare: aclnnBatchMatMulGetWorkspaceSize failed, status=" << st;
    return false;
  }
  return true;
}

bool CommitPrepareOutputs(gert::OpExecutePrepareContext *ctx, MatMulOpApiParams *params, uint64_t workspace_size) {
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxMatMulPrepare: SetOpApiParams failed";
    delete params;
    return false;
  }
  if (ctx->SetWorkspaceSizes({workspace_size}) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxMatMulPrepare: SetWorkspaceSizes failed";
    return false;
  }
  return true;
}

ge::graphStatus MatMulPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Tensor *x1 = nullptr;
  const gert::Tensor *x2 = nullptr;
  const gert::Tensor *y = nullptr;
  if (!FetchMmTensors(ctx, &x1, &x2, &y)) {
    return ge::GRAPH_FAILED;
  }
  bool trans_x1 = false;
  bool trans_x2 = false;
  bool feed_view_x1 = false;
  bool feed_view_x2 = false;
  if (!ReadAndResolveTranspose(ctx, x1, x2, y, &trans_x1, &trans_x2, &feed_view_x1, &feed_view_x2)) {
    return ge::GRAPH_FAILED;
  }

  MatMulAclArgs args;
  if (!BuildMmFamilyArgs(x1, x2, y, trans_x1, trans_x2, feed_view_x1, feed_view_x2, &args, "FxMatMulPrepare")) {
    return ge::GRAPH_FAILED;
  }
  auto *params = new (std::nothrow) MatMulOpApiParams();
  if (params == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  if (!RunMatmulGetWorkspace(&args, params, &workspace_size, CubeMathTypeFromPrepare(ctx))) {
    delete params;
    return ge::GRAPH_FAILED;
  }
  return CommitPrepareOutputs(ctx, params, workspace_size) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus BatchMatMulPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Tensor *x1 = nullptr;
  const gert::Tensor *x2 = nullptr;
  const gert::Tensor *y = nullptr;
  if (!FetchMmTensors(ctx, &x1, &x2, &y)) {
    return ge::GRAPH_FAILED;
  }
  bool trans_x1 = false;
  bool trans_x2 = false;
  bool feed_view_x1 = false;
  bool feed_view_x2 = false;
  if (!ReadAndResolveTranspose(ctx, x1, x2, y, &trans_x1, &trans_x2, &feed_view_x1, &feed_view_x2)) {
    return ge::GRAPH_FAILED;
  }

  MatMulAclArgs args;
  if (!BuildMmFamilyArgs(x1, x2, y, trans_x1, trans_x2, feed_view_x1, feed_view_x2, &args,
                         "FxBatchMatMulPrepare")) {
    return ge::GRAPH_FAILED;
  }
  auto *params = new (std::nothrow) MatMulOpApiParams();
  if (params == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  if (!RunBmmGetWorkspace(&args, params, &workspace_size, CubeMathTypeFromPrepare(ctx))) {
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
    RT_GLOG(ERROR) << "FxMatMulLaunch: missing workspace addr";
    return false;
  }
  *workspace = ws_addrs->GetData()[0]->GetAddr();
  return true;
}

ge::graphStatus MatMulFamilyLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *params = reinterpret_cast<MatMulOpApiParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxMatMulLaunch: missing OpApiParams/executor";
    return ge::GRAPH_FAILED;
  }
  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  if (!ResolveWorkspace(ctx, &workspace, &workspace_size)) {
    return ge::GRAPH_FAILED;
  }
  aclnnStatus st = kAclnnSuccess;
  if (params->api == MatMulApi::kBmm) {
    st = aclnn_api::Call(aclnn_api::aclnnBatchMatMul, workspace, workspace_size, params->executor,
        ctx->GetStream());
  } else if (params->api == MatMulApi::kMatmul) {
    st = aclnn_api::Call(aclnn_api::aclnnMatmul, workspace, workspace_size, params->executor, ctx->GetStream());
  } else {
    st = aclnn_api::Call(aclnn_api::aclnnMm, workspace, workspace_size, params->executor, ctx->GetStream());
  }
  if (st != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxMatMulLaunch: aclnn launch failed, status=" << st
                   << " api=" << static_cast<int>(params->api);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// Catalog attr order for addmm: beta, alpha, transpose_x1, transpose_x2.
constexpr size_t kAddmmBetaAttr = 0;
constexpr size_t kAddmmAlphaAttr = 1;
constexpr size_t kAddmmTransX1Attr = 2;
constexpr size_t kAddmmTransX2Attr = 3;

struct AddmmAclArgs {
  aclTensor *bias = nullptr;
  aclTensor *mat1 = nullptr;
  aclTensor *mat2 = nullptr;
  aclTensor *y = nullptr;
  aclScalar *beta = nullptr;
  aclScalar *alpha = nullptr;
  float beta_storage = 1.0f;
  float alpha_storage = 1.0f;

  ~AddmmAclArgs() { Release(); }

  void Release() {
    DestroyAclTensor(bias);
    DestroyAclTensor(mat1);
    DestroyAclTensor(mat2);
    DestroyAclTensor(y);
    DestroyAclScalar(beta);
    DestroyAclScalar(alpha);
    bias = mat1 = mat2 = y = nullptr;
    beta = alpha = nullptr;
  }

  bool Valid() const {
    return bias != nullptr && mat1 != nullptr && mat2 != nullptr && y != nullptr && beta != nullptr &&
           alpha != nullptr;
  }
};

bool ReadAddmmScale(const gert::RuntimeAttrs *attrs, size_t index, float *scale) {
  if (attrs == nullptr || scale == nullptr) {
    return false;
  }
  const float *value = attrs->GetFloat(index);
  if (value == nullptr) {
    RT_GLOG(ERROR) << "FxAddmmPrepare: missing scale attr at index " << index;
    return false;
  }
  *scale = *value;
  return true;
}

bool BuildAddmmArgs(const gert::Tensor *mat1, const gert::Tensor *mat2, const gert::Tensor *bias,
                    const gert::Tensor *y, bool trans_x1, bool trans_x2, bool feed_view_x1, bool feed_view_x2,
                    float beta, float alpha, AddmmAclArgs *args) {
  MatMulAclArgs mm;
  if (!BuildMmFamilyArgs(mat1, mat2, y, trans_x1, trans_x2, feed_view_x1, feed_view_x2, &mm, "FxAddmmPrepare")) {
    return false;
  }
  args->mat1 = mm.x1;
  args->mat2 = mm.x2;
  args->y = mm.y;
  mm.x1 = mm.x2 = mm.y = nullptr;
  args->bias = ToAclTensor(bias);
  args->beta_storage = beta;
  args->alpha_storage = alpha;
  args->beta = aclnn_api::aclCreateScalar(&args->beta_storage, ACL_FLOAT);
  args->alpha = aclnn_api::aclCreateScalar(&args->alpha_storage, ACL_FLOAT);
  if (!args->Valid()) {
    RT_GLOG(ERROR) << "FxAddmmPrepare: aclCreateTensor/Scalar failed";
    args->Release();
    return false;
  }
  return true;
}

struct AddmmOpApiParams {
  aclOpExecutor *executor = nullptr;
};

ge::graphStatus AddmmPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  // Inputs arrive in aten.addmm IR order -- (self, mat1, mat2) -- because
  // AddmmAclnnLowering passes them through without the catalog row's
  // "mat1,mat2,bias" reordering. That is also exactly aclnnAddmm's own order.
  const gert::Tensor *bias = ctx->GetInputTensor(0);
  const gert::Tensor *mat1 = ctx->GetInputTensor(1);
  const gert::Tensor *mat2 = ctx->GetInputTensor(2);
  const gert::Tensor *y = ctx->GetOutputTensor(0);
  if (mat1 == nullptr || mat2 == nullptr || bias == nullptr || y == nullptr) {
    RT_GLOG(ERROR) << "FxAddmmPrepare: missing input/output tensor";
    return ge::GRAPH_FAILED;
  }
  float beta = 1.0f;
  float alpha = 1.0f;
  if (!ReadAddmmScale(ctx->GetAttrs(), kAddmmBetaAttr, &beta) ||
      !ReadAddmmScale(ctx->GetAttrs(), kAddmmAlphaAttr, &alpha)) {
    return ge::GRAPH_FAILED;
  }

  bool trans_x1 = false;
  bool trans_x2 = false;
  bool feed_view_x1 = false;
  bool feed_view_x2 = false;
  if (!ReadAndResolveTranspose(ctx, mat1, mat2, y, &trans_x1, &trans_x2, &feed_view_x1, &feed_view_x2,
                               kAddmmTransX1Attr, kAddmmTransX2Attr)) {
    return ge::GRAPH_FAILED;
  }
  AddmmAclArgs args;
  if (!BuildAddmmArgs(mat1, mat2, bias, y, trans_x1, trans_x2, feed_view_x1, feed_view_x2, beta, alpha, &args)) {
    return ge::GRAPH_FAILED;
  }
  auto *params = new (std::nothrow) AddmmOpApiParams();
  if (params == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnAddmmGetWorkspaceSize,
      static_cast<const aclTensor *>(args.bias), static_cast<const aclTensor *>(args.mat1),
      static_cast<const aclTensor *>(args.mat2), static_cast<const aclScalar *>(args.beta),
      static_cast<const aclScalar *>(args.alpha), args.y, CubeMathTypeFromPrepare(ctx), &workspace_size,
      &params->executor);
  args.Release();
  if (st != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxAddmmPrepare: aclnnAddmmGetWorkspaceSize failed, status=" << st;
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxAddmmPrepare: SetOpApiParams failed";
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetWorkspaceSizes({workspace_size}) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxAddmmPrepare: SetWorkspaceSizes failed";
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddmmLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *params = reinterpret_cast<AddmmOpApiParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxAddmmLaunch: missing OpApiParams/executor";
    return ge::GRAPH_FAILED;
  }
  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  if (!ResolveWorkspace(ctx, &workspace, &workspace_size)) {
    return ge::GRAPH_FAILED;
  }
  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnAddmm, workspace, workspace_size, params->executor,
      ctx->GetStream());
  if (st != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxAddmmLaunch: aclnnAddmm failed, status=" << st;
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

bool PatchPrepareLaunch(const gert::OpImplSpaceRegistryV2Ptr &registry, const char *opp_type,
                        gert::OpImplRegisterV2::OpExecPrepareFunc prepare,
                        gert::OpImplRegisterV2::OpExecLaunchFunc launch) {
  if (registry == nullptr) {
    return false;
  }
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl(opp_type));
  if (funcs == nullptr) {
    RT_GLOG(ERROR) << "InstallFxMatMulDualStage: GetOpImpl(" << opp_type << ") returned null";
    return false;
  }
  funcs->op_execute_prepare_func = prepare;
  funcs->op_execute_launch_func = launch;
  RT_GLOG(INFO) << "Fx dual-stage installed: type=" << opp_type
                << " prepare=" << reinterpret_cast<const void *>(prepare)
                << " launch=" << reinterpret_cast<const void *>(launch);
  return true;
}

}  // namespace

bool InstallFxMatMulDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  return PatchPrepareLaunch(registry, "MatMul", MatMulPrepare, MatMulFamilyLaunch);
}

bool InstallFxBatchMatMulDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  return PatchPrepareLaunch(registry, "BatchMatMul", BatchMatMulPrepare, MatMulFamilyLaunch);
}

// Mount point: OPP's MatMulV2 entry (see InstallFxAddmmDualStage). It carries a
// native OPTIONAL_INPUT(bias), which is exactly the addmm shape, and unlike a
// synthesized entry it has an op proto and shape inference behind it.
bool InstallFxAddmmDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) {
    return false;
  }
  // MatMulV2 is a real OPP type carrying a native OPTIONAL_INPUT(bias). Never
  // mint an entry named "Addmm": OPP registers no such op proto, so a synthesized
  // entry has no shape inference and OM2 never binds the launch output.
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("MatMulV2"));
  if (funcs == nullptr) {
    RT_GLOG(ERROR) << "InstallFxAddmmDualStage: GetOpImpl(MatMulV2) returned null";
    return false;
  }
  funcs->op_execute_prepare_func = AddmmPrepare;
  funcs->op_execute_launch_func = AddmmLaunch;
  RT_GLOG(INFO) << "Fx dual-stage installed: type=MatMulV2(addmm) prepare="
                << reinterpret_cast<const void *>(AddmmPrepare)
                << " launch=" << reinterpret_cast<const void *>(AddmmLaunch);
  return true;
}

}  // namespace fx_converter
