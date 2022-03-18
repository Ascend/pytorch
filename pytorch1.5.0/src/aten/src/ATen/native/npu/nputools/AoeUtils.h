// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NATIVE_NPU_TOOLS_AOEUTILS__
#define __NATIVE_NPU_TOOLS_AOEUTILS__

#include <unordered_set>
#include <third_party/acl/inc/acl/acl_op_compiler.h>
#include <c10/npu/NPUException.h>

namespace at {
namespace native {
namespace npu {
namespace aoe {

class AoeDumpGraphManager  {
public:
  void SetDumpGraphPath(const std::string& dump_path);
  std::string GetDumpGraphPath() const;

  aclGraphDumpOption* CreateGraphDumpOption();
  void DestropyGraphDumpOption();

  void EnableAoe();
  bool IsAoeEnabled() const;
  bool IsInWhiltelist(const std::string &opName) const;

  bool aoe_enable = false;
  // to save graph for autotune, default path is ./
  std::string autotune_graphdumppath = "./";
  aclGraphDumpOption* AclGraphDumpOption = NULL;
  std::unordered_set<std::string> whilte_list_ = {
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
      "LogSoftmaxGrad",
      "LogSoftmaxV2",
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
      "SoftmaxV2",
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
      "Conv3DTranspose"};
};

AoeDumpGraphManager& aoe_manager();

} // namespace aoe
} // namespace npu
} // namespace native
} // namespace at

#endif // __NATIVE_NPU_TOOLS_AOEUTILS__