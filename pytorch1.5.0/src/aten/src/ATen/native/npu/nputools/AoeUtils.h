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
      "AbsGrad",
      "AcosGrad",
      "Add",
      "AsinGrad",
      "AsinhGrad",
      "AtanGrad",
      "AvgPool",
      "BatchMatMul",
      "BatchMatMulV2",
      "BiasAddGrad",
      "BNTrainingUpdate",
      "Cast",
      "Ceil",
      "ConcatD",
      "Conv2D",
      "Conv2DBackpropFilter",
      "Conv2DBackpropInput",
      "Conv2DCompress",
      "Conv2DTranspose",
      "Conv3D",
      "Conv3DBackpropFilter",
      "Conv3DBackpropInput",
      "Cos",
      "Cosh",
      "CosineEmbeddingLoss",
      "Deconvolution",
      "DepthwiseConv2D",
      "DepthwiseConv2DBackpropFilter",
      "DepthwiseConv2DBackpropInput",
      "Div",
      "DynamicRNN",
      "Elu",
      "EluGrad",
      "Equal",
      "Erf",
      "Erfc",
      "Exp",
      "Expm1",
      "Floor",
      "FullyConnection",
      "FullyConnectionCompress",
      "Gelu",
      "GeluGrad",
      "GEMM",
      "GNTrainingReduce",
      "GNTrainingUpdate",
      "INTrainingReduceV2",
      "INTrainingUpdateV2",
      "Inv",
      "InvGrad",
      "L2Loss",
      "L2Normalize",
      "L2NormalizeGrad",
      "Log",
      "Log1p",
      "LogSoftmaxGrad",
      "LogSoftmaxV2",
      "MatMul",
      "MatMulV2",
      "MatMulV2Compress",
      "Maximum",
      "Mod",
      "Mul",
      "Neg",
      "OnesLike",
      "Pooling",
      "Pow",
      "PReluGrad",
      "Reciprocal",
      "ReciprocalGrad",
      "ReduceAllD",
      "ReduceAnyD",
      "ReduceMaxD",
      "ReduceMeanD",
      "ReduceSumD",
      "Relu",
      "Relu6",
      "Relu6Grad",
      "ReluGrad",
      "Rint",
      "Round",
      "Rsqrt",
      "RsqrtGrad",
      "Selu",
      "Sigmoid",
      "SigmoidCrossEntropyWithLogits",
      "SigmoidGrad",
      "Sign",
      "Sinh",
      "SmoothL1Loss",
      "SoftmaxCrossEntropyWithLogits",
      "SoftmaxGrad",
      "SoftmaxV2",
      "Softplus",
      "Softsign",
      "SplitD",
      "Sqrt",
      "SqrtGrad",
      "Square",
      "StridedSliceD",
      "Sub",
      "Tanh",
      "TanhGrad",
      "TileD",
      "Unpack"};
};

AoeDumpGraphManager& aoe_manager();

} // namespace aoe
} // namespace npu
} // namespace native
} // namespace at

#endif // __NATIVE_NPU_TOOLS_AOEUTILS__