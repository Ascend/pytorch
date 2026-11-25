/**
 * Fx-owned dual-stage Convolution: wrap aclnnConvolutionGetWorkspaceSize / aclnnConvolution.
 *
 * torch_npu lowers aten.convolution to aclnnConvolution, so mode3 must call the
 * same kernel to stay bit-exact with mode1. A matmul rewrite of a k=1 conv is NOT
 * equivalent on Ascend (measured max abs diff 5.4e-04, different cube tiling).
 *
 * OPP has no "Convolution" OpImpl entry, so the catalog row mounts this wrapper on
 * the existing "Conv2D" entry (same pattern Add/MatMul use to patch in place).
 *
 * Format: aclnnConvolution documents input/weight/output as NCL | NCHW | NCDHW --
 * ND is NOT accepted (it returns ACLNN_ERR_PARAM_INVALID). A 3-D Conv1d therefore
 * passes NCL directly, with stride/padding/dilation of length (rank - 2) == 1,
 * which is exactly what aten hands us. No dimension promotion is involved.
 */

#include "opp/convolution_dual_stage_impl.h"

#include <cstdint>
#include <string>
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

struct ConvOpApiParams {
  aclOpExecutor *executor = nullptr;
};

// Attr order mirrors the catalog row in convolution_op_lowering.cc.
enum ConvAttrIndex : size_t {
  kStride = 0,
  kPadding,
  kDilation,
  kTransposed,
  kOutputPadding,
  kGroups,
};

void DestroyIntArray(aclIntArray *a) {
  if (a != nullptr) aclnn_api::aclDestroyIntArray(a);
}

struct ConvAclArgs {
  aclTensor *input = nullptr;
  aclTensor *weight = nullptr;
  aclTensor *bias = nullptr;
  aclTensor *output = nullptr;
  aclIntArray *stride = nullptr;
  aclIntArray *padding = nullptr;
  aclIntArray *dilation = nullptr;
  aclIntArray *output_padding = nullptr;

  ~ConvAclArgs() { Release(); }

  void Release() {
    DestroyAclTensor(input);
    DestroyAclTensor(weight);
    DestroyAclTensor(bias);
    DestroyAclTensor(output);
    DestroyIntArray(stride);
    DestroyIntArray(padding);
    DestroyIntArray(dilation);
    DestroyIntArray(output_padding);
    input = weight = bias = output = nullptr;
    stride = padding = dilation = output_padding = nullptr;
  }
};

std::vector<int64_t> ReadListIntAttr(const gert::RuntimeAttrs *attrs, size_t index,
                                     const std::vector<int64_t> &fallback) {
  const auto *list = attrs == nullptr ? nullptr : attrs->GetListInt(index);
  if (list == nullptr || list->GetSize() == 0U) return fallback;
  std::vector<int64_t> out;
  out.reserve(list->GetSize());
  for (size_t i = 0; i < list->GetSize(); ++i) out.push_back(list->GetData()[i]);
  return out;
}

// The accepted lowering subset encodes uniform spatial attributes as a
// singleton. Expand those defaults to the rank required by public ACLNN.
bool NormalizeSpatialAttr(std::vector<int64_t> *values, size_t spatial_rank) {
  if (values->size() == 1U) values->resize(spatial_rank, values->front());
  return values->size() == spatial_rank;
}

std::vector<int64_t> ContiguousStridesOf(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Rank 3 -> NCL, rank 4 -> NCHW, rank 5 -> NCDHW. aclnnConvolution documents exactly
// these for input/weight/output and rejects ND with PARAM_INVALID.
aclFormat ConvFormatForRank(size_t rank) {
  switch (rank) {
    case 3U:
      return ACL_FORMAT_NCL;
    case 4U:
      return ACL_FORMAT_NCHW;
    case 5U:
      return ACL_FORMAT_NCDHW;
    default:
      return ACL_FORMAT_ND;
  }
}

aclTensor *ConvAclTensor(const gert::Tensor *tensor, bool is_bias) {
  const auto shape = TensorLogicalShape(tensor);
  const auto strides = ContiguousStridesOf(shape);
  const aclFormat format = is_bias ? ACL_FORMAT_ND : ConvFormatForRank(shape.size());
  return aclnn_api::aclCreateTensor(shape.data(), shape.size(), ToAclDataType(tensor->GetDataType()), strides.data(), 0,
                         format, shape.data(), shape.size(), const_cast<void *>(tensor->GetAddr()));
}

bool BuildConvAclArgs(gert::OpExecutePrepareContext *ctx, const gert::Tensor *input, const gert::Tensor *weight,
                      const gert::Tensor *bias, const gert::Tensor *output, ConvAclArgs *args,
                      bool *transposed, int64_t *groups) {
  args->input = ConvAclTensor(input, false);
  args->weight = ConvAclTensor(weight, false);
  // Bias is an optional GE input; an absent one must stay absent for aclnn.
  args->bias = bias == nullptr ? nullptr : ConvAclTensor(bias, true);
  args->output = ConvAclTensor(output, false);
  if (args->input == nullptr || args->weight == nullptr || args->output == nullptr ||
      (bias != nullptr && args->bias == nullptr)) {
    RT_GLOG(ERROR) << "FxConvPrepare: aclCreateTensor failed";
    args->Release();
    return false;
  }
  const auto *attrs = ctx->GetAttrs();
  std::vector<int64_t> stride = ReadListIntAttr(attrs, kStride, {1});
  std::vector<int64_t> padding = ReadListIntAttr(attrs, kPadding, {0});
  std::vector<int64_t> dilation = ReadListIntAttr(attrs, kDilation, {1});
  const size_t input_rank = TensorLogicalShape(input).size();
  if (input_rank < 3U || input_rank > 5U ||
      !NormalizeSpatialAttr(&stride, input_rank - 2U) ||
      !NormalizeSpatialAttr(&padding, input_rank - 2U) ||
      !NormalizeSpatialAttr(&dilation, input_rank - 2U)) {
    RT_GLOG(ERROR) << "FxConvPrepare: invalid spatial attribute length for input rank " << input_rank;
    return false;
  }
  // aclnn ignores outputPadding for a non-transposed convolution; pass it only when it
  // applies so an unexpected list length cannot trip validation.
  const bool *transposed_attr = attrs == nullptr ? nullptr : attrs->GetBool(kTransposed);
  const bool is_transposed = transposed_attr != nullptr && *transposed_attr;
  std::vector<int64_t> output_padding =
      is_transposed ? ReadListIntAttr(attrs, kOutputPadding, {0}) : std::vector<int64_t>{};

  args->stride = aclnn_api::aclCreateIntArray(stride.data(), stride.size());
  args->padding = aclnn_api::aclCreateIntArray(padding.data(), padding.size());
  args->dilation = aclnn_api::aclCreateIntArray(dilation.data(), dilation.size());
  args->output_padding = aclnn_api::aclCreateIntArray(output_padding.data(), output_padding.size());
  if (args->stride == nullptr || args->padding == nullptr || args->dilation == nullptr ||
      args->output_padding == nullptr) {
    RT_GLOG(ERROR) << "FxConvPrepare: aclCreateIntArray failed";
    args->Release();
    return false;
  }
  const int64_t *groups_attr = attrs == nullptr ? nullptr : attrs->GetInt(kGroups);
  *transposed = is_transposed;
  *groups = groups_attr != nullptr ? *groups_attr : 1;
  return true;
}

// aclnn cubeMathType: KEEP_DTYPE=0, USE_HF32=3. It must follow
// torch.npu.conv.allow_hf32 through OpExecuteOptions, the same way the MatMul
// wrapper does; hardcoding KEEP_DTYPE leaves the converter in full fp32 while eager
// runs hf32, which shows up as a ~5e-04 mismatch against torch_npu's convolution.
constexpr int8_t kCubeKeepDtype = 0;
constexpr int8_t kCubeUseHf32 = 3;

int8_t CubeMathTypeFromPrepare(gert::OpExecutePrepareContext *ctx) {
  const char *hf32 = ctx != nullptr ? ctx->GetAllowHf32() : nullptr;
  // allow_hf32[0]=conv, [1]=matmul; '1' enables.
  if (hf32 != nullptr && hf32[0] == '1') {
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
    RT_GLOG(ERROR) << "FxConvLaunch: missing workspace addr";
    return false;
  }
  *workspace = ws_addrs->GetData()[0]->GetAddr();
  return true;
}

ge::graphStatus ConvPrepare(gert::OpExecutePrepareContext *ctx) {
  if (ctx == nullptr) return ge::GRAPH_FAILED;
  const gert::Tensor *input = ctx->GetInputTensor(0);
  const gert::Tensor *weight = ctx->GetInputTensor(1);
  // Prepare input chains contain the compute inputs followed by outputs.
  // With no bias, physical slot 2 already holds output 0. Check compute arity
  // before accessing that slot; the raw shadow has no optional-IR instance map.
  const size_t input_count = ctx->GetComputeNodeInputNum();
  if (input_count < 2U || input_count > 3U) {
    RT_GLOG(ERROR) << "FxConvPrepare: unexpected compute input count " << input_count;
    return ge::GRAPH_FAILED;
  }
  const gert::Tensor *bias = input_count == 3U ? ctx->GetInputTensor(2) : nullptr;
  const gert::Tensor *output = ctx->GetOutputTensor(0);
  if (input == nullptr || weight == nullptr || output == nullptr) {
    RT_GLOG(ERROR) << "FxConvPrepare: missing input/weight/output tensor";
    return ge::GRAPH_FAILED;
  }
  ConvAclArgs args;
  bool transposed = false;
  int64_t groups = 1;
  if (!BuildConvAclArgs(ctx, input, weight, bias, output, &args, &transposed, &groups)) {
    return ge::GRAPH_FAILED;
  }
  auto *params = new (std::nothrow) ConvOpApiParams();
  if (params == nullptr) {
    args.Release();
    return ge::GRAPH_FAILED;
  }
  uint64_t workspace_size = 0;
  const int8_t cube_math_type = CubeMathTypeFromPrepare(ctx);
  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnConvolutionGetWorkspaceSize,
      static_cast<const aclTensor *>(args.input), static_cast<const aclTensor *>(args.weight),
      static_cast<const aclTensor *>(args.bias), static_cast<const aclIntArray *>(args.stride),
      static_cast<const aclIntArray *>(args.padding), static_cast<const aclIntArray *>(args.dilation), transposed,
      static_cast<const aclIntArray *>(args.output_padding), groups, args.output, cube_math_type, &workspace_size,
      &params->executor);
  args.Release();
  if (st != kAclnnSuccess || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxConvPrepare: aclnnConvolutionGetWorkspaceSize failed, status="
                   << static_cast<int>(st) << " executor=" << static_cast<void *>(params->executor);
    delete params;
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetOpApiParamsWithDefaultDeleter(params) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxConvPrepare: SetOpApiParams failed";
    return ge::GRAPH_FAILED;
  }
  if (ctx->SetWorkspaceSizes({workspace_size}) != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "FxConvPrepare: SetWorkspaceSizes failed";
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConvLaunch(gert::OpExecuteLaunchContext *ctx) {
  if (ctx == nullptr) return ge::GRAPH_FAILED;
  auto *params = reinterpret_cast<ConvOpApiParams *>(ctx->GetOpApiParams());
  if (params == nullptr || params->executor == nullptr) {
    RT_GLOG(ERROR) << "FxConvLaunch: missing OpApiParams/executor";
    return ge::GRAPH_FAILED;
  }
  void *workspace = nullptr;
  uint64_t workspace_size = 0;
  if (!ResolveWorkspace(ctx, &workspace, &workspace_size)) return ge::GRAPH_FAILED;
  const aclnnStatus st = aclnn_api::Call(aclnn_api::aclnnConvolution, workspace, workspace_size, params->executor,
      ctx->GetStream());
  if (st != kAclnnSuccess) {
    RT_GLOG(ERROR) << "FxConvLaunch: aclnnConvolution failed, status=" << st;
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

bool InstallFxConvolutionDualStage(const gert::OpImplSpaceRegistryV2Ptr &registry) {
  if (registry == nullptr) return false;
  // Mount point: OPP's Conv2D entry. Nothing under "Convolution" exists to patch,
  // and CreateOrGetOpImpl (which could mint a fresh entry) is private.
  auto *funcs = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registry->GetOpImpl("Conv2D"));
  if (funcs == nullptr) {
    RT_GLOG(ERROR) << "InstallFxConvolutionDualStage: GetOpImpl(Conv2D) returned null";
    return false;
  }
  funcs->op_execute_prepare_func = ConvPrepare;
  funcs->op_execute_launch_func = ConvLaunch;
  RT_GLOG(INFO) << "Fx dual-stage installed: type=Conv2D(convolution) prepare="
                << reinterpret_cast<const void *>(ConvPrepare) << " launch="
                << reinterpret_cast<const void *>(ConvLaunch);
  return true;
}

}  // namespace fx_converter
