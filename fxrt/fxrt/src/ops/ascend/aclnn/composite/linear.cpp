#include "ops/ascend/aclnn/composite/linear.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"
#include "common/logger.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "ir/common/dtype.h"

namespace fxrt {
namespace ops {
void AclnnLinear::SetFlatternNdLinearTensorStorageInfo(
    ir::TensorPtr& tensor,
    int64_t newShapeFirst,
    const std::vector<int64_t>& shape) {
  auto newShapeSecond = shape[shape.size() - 1];
  std::vector<int64_t> newShape{newShapeFirst, newShapeSecond};

  // Set the new shape
  tensor->SetShape(newShape);

  // Calculate strides for the 2D view
  // For a 2D tensor [M, N], strides are [N, 1] (row-major)
  std::vector<int64_t> stridesNew{newShapeSecond, 1};
  tensor->SetStrides(stridesNew);

  // Set storage shape to match the new shape (for view operations)
  tensor->SetStorageShape(newShape);

  // Storage offset remains 0 for this view (no offset needed)
  tensor->SetStorageOffset(0);
}

std::vector<int64_t> AclnnLinear::CalculateStorageShapeForNZ(
    const std::vector<int64_t>& viewShape,
    ir::MemoryFormat format,
    ir::DataType dtype) {
  // Only process NZ format
  if (format != ir::MemoryFormat::FORMAT_FRACTAL_NZ) {
    // Non-NZ format, StorageShape equals ViewShape
    return viewShape;
  }

  std::vector<int64_t> storageShape;

  // Ensure at least 2 dimensions
  if (viewShape.size() == 0) {
    RT_GLOG(EXCEPTION) << "viewShape size is 0, but which should be >= 2";
    return {1, 1, 16, 16};
  }
  if (viewShape.size() == 1) {
    RT_GLOG(EXCEPTION) << "viewShape size is 1, but which should be >= 2";
    return {1, viewShape[0], 16, 16};
  }

  // Keep first N-2 dimensions
  size_t i = 0;
  for (; i < viewShape.size() - 2; i++) {
    storageShape.push_back(viewShape[i]);
  }

  // Get itemsize
  size_t itemsize = dtype.GetSize();
  // float32 will be cast to float16
  size_t itemsize_ = (itemsize > 2) ? 2 : itemsize;

  // Constants
  constexpr int64_t BLOCKSIZE = 16;
  constexpr int64_t BLOCKBYTES = 32;
  int64_t lastSize = BLOCKBYTES / itemsize_; // Usually 16

  // Convert last 2 dimensions to block format
  // For [..., H, W]:
  // storageShape = [..., ceil(W/16), ceil(H/16), 16, 16]
  int64_t W = viewShape[i + 1]; // Last dimension
  int64_t H = viewShape[i]; // Second to last dimension

  storageShape.push_back((W + lastSize - 1) / lastSize); // ceil(W/16)
  storageShape.push_back((H + BLOCKSIZE - 1) / BLOCKSIZE); // ceil(H/16)
  storageShape.push_back(BLOCKSIZE); // 16
  storageShape.push_back(lastSize); // 16

  return storageShape;
}

std::vector<int64_t> AclnnLinear::TransposeWeight(
    const ir::TensorPtr& wTensor,
    const std::vector<int64_t>& wShape,
    ir::TensorPtr& weightTransposeTensor) {
  std::vector<int64_t> weightTransposeShape = wShape;

  if (weightTransposeShape.size() >= kIndex2) {
    auto wRank = wShape.size();
    // Swap last two dimensions
    std::swap(weightTransposeShape[wRank - kIndex1], weightTransposeShape[wRank - kIndex2]);

    // Update the cloned tensor's shape  (in ViewShape)
    weightTransposeTensor->SetShape(weightTransposeShape);

    // Calculate transposed strides by swapping last two dimensions
    // Get original strides, or compute default contiguous strides if not set
    std::vector<int64_t> stridesTransposed = wTensor->Strides();

    // If strides are empty or default, compute contiguous strides first
    if (stridesTransposed.empty() || stridesTransposed.size() != wRank) {
      // Compute default contiguous strides (row-major)
      stridesTransposed.resize(wRank);
      int64_t stride = 1;
      for (int i = wRank - 1; i >= 0; --i) {
        stridesTransposed[i] = stride;
        stride *= wShape[i];
      }
    }

    // Swap last two strides to match transposed shape
    std::swap(stridesTransposed[wRank - 1], stridesTransposed[wRank - 2]);
    weightTransposeTensor->SetStrides(stridesTransposed);

    ir::MemoryFormat format = wTensor->Format();
    ir::DataType dtype = wTensor->Dtype();

    if (format == ir::MemoryFormat::FORMAT_FRACTAL_NZ) {
      // NZ format needs recalculate StorageShape based on new ViewShape
      std::vector<int64_t> storageShape = CalculateStorageShapeForNZ(weightTransposeShape, format, dtype);
      weightTransposeTensor_->SetStorageShape(storageShape);
    } else {
      // Update storage shape to match transposed shape
      weightTransposeTensor->SetStorageShape(weightTransposeShape);
    }

    // Storage offset remains 0 (no offset needed for transpose view)
    weightTransposeTensor->SetStorageOffset(0);
  }

  return weightTransposeShape;
}

OpsErrorCode AclnnLinear::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  CHECK_IF_FAIL(input.size() >= kIndex2);

  const auto& xTensor = input[kIndex0]->ToTensor();
  const auto& wTensor = input[kIndex1]->ToTensor();
  const auto& xShape = xTensor->Shape();
  const auto& wShape = wTensor->Shape();

  xRank_ = xShape.size();
  auto wRank = wShape.size();

  ir::MemoryFormat weightFormat = wTensor->Format();
  isWeightNz_ = (weightFormat == ir::MemoryFormat::FORMAT_FRACTAL_NZ);

  // 1. Generate transposed weight tensor
  // Create a clone of weight tensor for transpose view (deep copy, shares storage)
  weightTransposeTensor_ = wTensor->ShallowClone(); // deep copy for modify.
  auto weightTransposeShape = TransposeWeight(wTensor, wShape, weightTransposeTensor_);

  // 2. Check if bias is None
  isBiasNone_ = (input.size() <= kIndex2 || input[kIndex2] == nullptr || input[kIndex2]->IsNone());

  // Get cube math type
  cubeMathType_ = GetCubeMathType();

  // 3. Calculate workspace based on different scenarios
  if (isBiasNone_) {
    if (isWeightNz_) { // only support matmul(x1, x2), x1-2DTensor, x2-4DTensor
      if (xRank_ != kIndex2) {
        RT_GLOG(EXCEPTION) << "only support x1-2DTensor in NZ scene for MatMul";
      }
      executorMatmulNz_->GetWorkspaceSize(
          reinterpret_cast<uint64_t*>(workspaceSize),
          xTensor,
          weightTransposeTensor_,
          output->ToTensor(),
          cubeMathType_);
    } else {
      // Case 1: No bias - just matmul
      executorMatmul_->GetWorkspaceSize(
          reinterpret_cast<uint64_t*>(workspaceSize),
          xTensor,
          weightTransposeTensor_,
          output->ToTensor(),
          cubeMathType_);
    }
  } else {
    const auto& bias = input[kIndex2]->ToTensor();
    const auto& biasShape = bias->Shape();
    biasRank_ = biasShape.size();
    // Create alpha and beta scalars based on input tensor dtype
    alphaScalar_ = CreateScalarValueOne(xTensor->Dtype());
    betaScalar_ = CreateScalarValueOne(xTensor->Dtype());

    if (xRank_ == kIndex2) {
      if (isWeightNz_) {
        executorAddmmNz_->GetWorkspaceSize(
            reinterpret_cast<uint64_t*>(workspaceSize),
            bias,
            xTensor,
            weightTransposeTensor_,
            betaScalar_,
            alphaScalar_,
            output->ToTensor(),
            cubeMathType_);
      } else {
        executorAddmm_->GetWorkspaceSize(
            reinterpret_cast<uint64_t*>(workspaceSize),
            bias,
            xTensor,
            weightTransposeTensor_,
            betaScalar_,
            alphaScalar_,
            output->ToTensor(),
            cubeMathType_);
      }
    } else if (biasRank_ == 1 || xRank_ == 3) {
      // Case 3: Reshape input/output to 2D, then use addmm
      // Create clones (shallow copies) that share storage with originals
      inputKernelTensor_ = xTensor->ShallowClone();
      outputKernelTensor_ = output->ToTensor()->ShallowClone();

      int64_t inputReshapeSize = 1;
      if (xRank_ > kIndex1) {
        inputReshapeSize = std::accumulate(xShape.begin(), xShape.end() - 1, int64_t(1), std::multiplies<int64_t>());
      }

      // Flatten to 2D: [batch*..., features] x [features, out_features] -> [batch*..., out_features]
      SetFlatternNdLinearTensorStorageInfo(inputKernelTensor_, inputReshapeSize, xShape);

      SetFlatternNdLinearTensorStorageInfo(outputKernelTensor_, inputReshapeSize, weightTransposeShape);

      executorAddmm_->GetWorkspaceSize(
          reinterpret_cast<uint64_t*>(workspaceSize),
          bias,
          inputKernelTensor_,
          weightTransposeTensor_,
          betaScalar_,
          alphaScalar_,
          outputKernelTensor_,
          cubeMathType_);
    } else {
      // Case 4: Matmul first, then add bias separately
      RT_GLOG(EXCEPTION) << "Not support linear with bias.";
      std::vector<int64_t> matmulShape = xShape;
      matmulShape[xRank_ - 1] = weightTransposeShape[wRank - 1];

      // When x and w are 1D, output is scalar (0D)
      if (xRank_ == kIndex1 && wRank == kIndex1) {
        matmulShape = {};
      }

      matmulTensor_ = ir::MakeIntrusive<ir::Tensor>(matmulShape, xTensor->Dtype(), xTensor->GetDevice());
      // Calculate workspace for matmul
      size_t matmulWorkspace = 0;
      executorMatmul_->GetWorkspaceSize(
          reinterpret_cast<uint64_t*>(&matmulWorkspace), xTensor, weightTransposeTensor_, matmulTensor_, cubeMathType_);
      // Calculate workspace for add
      size_t addWorkspace = 0;
      executorAdd_->GetWorkspaceSize(
          reinterpret_cast<uint64_t*>(&addWorkspace), matmulTensor_, bias, alphaScalar_, output->ToTensor());

      // Total workspace is max of matmul and add, plus space for matmul output
      size_t matmulOutputSize = matmulTensor_->Numel() * matmulTensor_->Dtype().GetSize();
      *workspaceSize = std::max(matmulWorkspace, addWorkspace) + matmulOutputSize;
    }
  }

  return SUCCESS;
}

OpsErrorCode AclnnLinear::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  CHECK_IF_NULL(stream);
  CHECK_IF_FAIL(input.size() >= kIndex2);

  const auto& xTensor = input[kIndex0]->ToTensor();

  // 1. Transposed weight tensor already shares storage with original (via clone in CalcWorkspace)
  // Ensure the data pointer is correctly set (clone already shares storage, but verify)
  // The w_t_tensor_ was created in CalcWorkspace and shares storage, so no update needed

  // 2. Execute based on scenario
  if (isBiasNone_) {
    if (isWeightNz_) {
      executorMatmulNz_->Launch(
          workspace, workspaceSize, stream, xTensor, weightTransposeTensor_, output->ToTensor(), cubeMathType_);
    } else {
      // Case 1: No bias - just matmul
      executorMatmul_->Launch(
          workspace, workspaceSize, stream, xTensor, weightTransposeTensor_, output->ToTensor(), cubeMathType_);
    }
  } else {
    const auto& bias = input[kIndex2]->ToTensor();

    if (xRank_ == kIndex2) {
      if (isWeightNz_) {
        executorAddmmNz_->Launch(
            workspace,
            workspaceSize,
            stream,
            bias,
            xTensor,
            weightTransposeTensor_,
            betaScalar_,
            alphaScalar_,
            output->ToTensor(),
            cubeMathType_);
      } else {
        executorAddmm_->Launch(
            workspace,
            workspaceSize,
            stream,
            bias,
            xTensor,
            weightTransposeTensor_,
            betaScalar_,
            alphaScalar_,
            output->ToTensor(),
            cubeMathType_);
      }
    } else if (biasRank_ == 1 || xRank_ == 3) {
      // Case 3: Reshape and use addmm
      executorAddmm_->Launch(
          workspace,
          workspaceSize,
          stream,
          bias,
          inputKernelTensor_,
          weightTransposeTensor_,
          betaScalar_,
          alphaScalar_,
          outputKernelTensor_,
          cubeMathType_);
    } else {
      // Case 4: Matmul then add
      // Use workspace for matmul output
      size_t matmulOutputSize = matmulTensor_->Numel() * matmulTensor_->Dtype().GetSize();
      void* matmulOutputPtr = static_cast<char*>(workspace) + workspaceSize - matmulOutputSize;
      matmulTensor_->UpdateData(matmulOutputPtr);

      // Execute matmul
      size_t matmulWorkspaceSize = workspaceSize - matmulOutputSize;
      executorMatmul_->Launch(
          workspace, matmulWorkspaceSize, stream, xTensor, weightTransposeTensor_, matmulTensor_, cubeMathType_);
      // Execute add
      executorAdd_->Launch(workspace, workspaceSize, stream, matmulTensor_, bias, alphaScalar_, output->ToTensor());
    }
  }

  return SUCCESS;
}

ir::ValuePtr AclnnLinear::CreateScalarValueOne(ir::DataType dtype) {
  switch (dtype.value) {
    case ir::DataType::Float32:
    case ir::DataType::Float16:
    case ir::DataType::BFloat16:
      return ir::MakeIntrusive<ir::Value>(1.0f);
    case ir::DataType::UInt8:
    case ir::DataType::Int8:
    case ir::DataType::Int16:
    case ir::DataType::Int32:
    case ir::DataType::Int64:
      return ir::MakeIntrusive<ir::Value>(static_cast<int64_t>(1));
    case ir::DataType::Bool:
      return ir::MakeIntrusive<ir::Value>(true);
    default:
      RT_GLOG(EXCEPTION) << "Unsupported data type for scalar value: " << dtype.ToString();
      // Fallback to int64_t
      return ir::MakeIntrusive<ir::Value>(static_cast<int64_t>(1));
  }
}

} // namespace ops
} // namespace fxrt
