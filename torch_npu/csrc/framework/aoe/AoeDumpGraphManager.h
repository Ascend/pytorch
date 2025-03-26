#ifndef __NATIVE_NPU_TOOLS_AOEUTILS__
#define __NATIVE_NPU_TOOLS_AOEUTILS__

#include <unordered_set>
#include <third_party/acl/inc/acl/acl_op_compiler.h>
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {
namespace aoe {

class AoeDumpGraphManager  {
public:
  void SetDumpGraphPath(const std::string& dump_path);
  std::string GetDumpGraphPath() const;

  aclGraphDumpOption* CreateGraphDumpOption();
  void DestropyGraphDumpOption();

  void EnableAoe();
  bool IsAoeEnabled() const;
  bool IsInWhitelist(const std::string &opName) const;

  bool aoe_enable = false;
  // to save graph for autotune, default path is ./
  std::string autotune_graphdumppath = "./";
  aclGraphDumpOption* AclGraphDumpOption = nullptr;
  std::unordered_set<std::string> white_list_ = {
      "Abs",
      "AccumulateNV2",
      "Add",
      "AddN",
      "Adds",
      "BatchNorm3D",
      "BiasAdd",
      "BiasAddGrad",
      "BNInfer",
      "BNTrainingReduce",
      "BNTrainingUpdate",
      "BNTrainingUpdateGrad",
      "BNTrainingUpdateV2",
      "BNTrainingUpdateV3",
      "BroadcastToD",
      "ClipByNormNoDivSum",
      "ClipByValue",
      "DiagPartD",
      "Div",
      "DivNoNan",
      "DynamicRNN",
      "Elu",
      "EluGrad",
      "Equal",
      "Exp",
      "ExpandD",
      "Floor",
      "FloorDiv",
      "Gelu",
      "Greater",
      "GreaterEqual",
      "InstanceNorm",
      "L2Loss",
      "LambUpdateWithLrV2",
      "LayerNormBetaGammaBackpropV2",
      "LeakyRelu",
      "Less",
      "LessEqual",
      "Log",
      "Log1p",
      "LogicalAnd",
      "LogicalNot",
      "LogicalOr",
      "LpNorm",
      "MatrixDiagD",
      "Maximum",
      "Minimum",
      "Mul",
      "Muls",
      "Neg",
      "NotEqual",
      "Pow",
      "PRelu",
      "RealDiv",
      "ReduceAllD",
      "ReduceMaxD",
      "ReduceMeanD",
      "ReduceMinD",
      "ReduceSumD",
      "Relu",
      "ReluV2",
      "Round",
      "Rsqrt",
      "RsqrtGrad",
      "Sigmoid",
      "Sign",
      "SoftmaxCrossEntropyWithLogits",
      "Softplus",
      "SplitVD",
      "SqrtGrad",
      "Square",
      "SquaredDifference",
      "SquareSumV1",
      "SquareSumV2",
      "StridedSliceD",
      "Sub",
      "Tanh",
      "TileD",
      "Unpack",
      "AvgPool",
      "Deconvolution",
      "Conv2D",
      "Conv2DCompress",
      "Conv2DBackpropInput",
      "Conv2DBackpropFilter",
      "GEMM",
      "MatMul",
      "MatMulV2",
      "MatMulV2Compress",
      "BatchMatMul",
      "BatchMatMulV2",
      "FullyConnection",
      "FullyConnectionCompress",
      "DepthwiseConv2D",
      "DepthwiseConv2DBackpropInput",
      "DepthwiseConv2DBackpropFilter",
      "Conv3D",
      "Conv3DBackpropInput",
      "Conv3DBackpropFilter",
      "AvgPool3DGrad",
      "Pooling",
      "Conv2DTranspose",
      "Conv3DTranspose",
      "ReluGradV2",
      "AvgPoolV2",
      "Conv1D",
      "DeformableConv2D",
      "AvgPool3D",
      "LogSoftmaxGrad",
      "LogSoftmaxV2",
      "SoftmaxGrad",
      "SoftmaxV2",
    };
};

AoeDumpGraphManager& aoe_manager();

} // namespace aoe
} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_TOOLS_AOEUTILS__