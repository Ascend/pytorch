#include "ops/ascend/aclnn/utils/view_utils.h"

#include <algorithm>

#include "common/common.h"
#include "common/logger.h"
#include "ops/utils/utils.h"
namespace fxrt {
namespace ops {

std::optional<std::vector<int64_t>> CalculateViewStrides(
    const std::vector<int64_t>& curShape,
    const std::vector<int64_t>& curStrides,
    const std::vector<int64_t>& newShape) {
  if (curShape.empty()) {
    return std::vector<int64_t>(newShape.size(), 1);
  }

  bool isOldEmpty = std::any_of(curShape.begin(), curShape.end(), [](int64_t dim) { return dim == 0; });
  if (isOldEmpty && curShape == newShape) {
    return curStrides;
  }

  const int64_t newRank = SizeToLong(newShape.size());
  std::vector<int64_t> newStrides(newRank, 0);
  if (isOldEmpty) {
    for (int64_t dim = newRank - 1; dim >= 0; --dim) {
      if (dim == (newRank - 1)) {
        newStrides[dim] = 1;
      } else {
        newStrides[dim] = std::max(newShape[dim + 1], static_cast<int64_t>(1)) * newStrides[dim + 1];
      }
    }
    return newStrides;
  }

  int64_t viewDim = newRank - 1;
  int64_t baseStride = curStrides.back();
  int64_t tensorElems = 1;
  int64_t viewElems = 1;
  for (int64_t dim = SizeToLong(curShape.size()) - 1; dim >= 0; --dim) {
    tensorElems *= curShape[dim];
    if (dim == 0 || (curShape[dim - 1] != 1 && curStrides[dim - 1] != tensorElems * baseStride)) {
      while (viewDim >= 0 && (viewElems < tensorElems || newShape[viewDim] == 1)) {
        newStrides[viewDim] = viewElems * baseStride;
        viewElems *= newShape[viewDim];
        --viewDim;
      }
      if (viewElems != tensorElems) {
        return std::nullopt;
      }
      if (dim > 0) {
        baseStride = curStrides[dim - 1];
        tensorElems = 1;
        viewElems = 1;
      }
    }
  }
  if (viewDim != -1) {
    return std::nullopt;
  }

  return newStrides;
}

void CheckViewMetaDataChangeForFormat(const ir::TensorPtr& inputTensorPtr, const std::vector<int64_t>& newShape) {
  if (!IsTensorBaseFormat(inputTensorPtr) && IsDefiniteTensorWhenMetaDataChanges(inputTensorPtr, newShape)) {
    RT_GLOG(EXCEPTION) << "Current tensor format " << ir::FormatEnumToStr(inputTensorPtr->Format())
                       << " does not support view metadata change when the view target shape has " << newShape.size()
                       << " dimensions"
                       << ". The view operation is rejected to avoid implicit format conversion.";
  }
}

std::vector<int64_t> CalculateStrides(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<int64_t> ret(shape.size(), 1);
  int64_t strides = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides *= shape[i];
    ret[i - 1] = strides;
  }
  return ret;
}

int64_t DynamicDimWrap(int64_t dim, int64_t dimPostExpr, bool wrapScalar) {
  if (dimPostExpr * -1 <= dim && dim < dimPostExpr) {
    if (dim < 0) {
      return dim + dimPostExpr;
    }
    return dim;
  }
  if (dimPostExpr == 0) {
    if (!wrapScalar) {
      RT_GLOG(EXCEPTION) << "dim value specified as " << dim << ", but tensor has no dimensions";
    }
    return DynamicDimWrap(dim, 1, false);
  }
  RT_GLOG(EXCEPTION) << "Dimension out of range (expected to be in range of [" << -dimPostExpr << ", " << dimPostExpr
                     << "), but got " << dim << ")";
  return -1;
}

std::vector<int64_t> GetTensorStrides(const ir::TensorPtr& tensorPtr) {
  const auto& strides = tensorPtr->Strides();
  if (strides.empty()) {
    return CalculateStrides(tensorPtr->Shape());
  }
  return strides;
}

void UpdateTensorViewInfo(
    const ir::TensorPtr& inputTensorPtr,
    const ir::TensorPtr& outputTensorPtr,
    const std::vector<int64_t>& newShape,
    const std::vector<int64_t>& newStrides,
    size_t newStorageOffset) {
  CheckViewMetaDataChangeForFormat(inputTensorPtr, newShape);
  outputTensorPtr->SetStrides(newStrides);
  outputTensorPtr->SetStorageOffset(newStorageOffset);
  outputTensorPtr->SetStorageShape(inputTensorPtr->StorageShape());
  outputTensorPtr->SetFormat(inputTensorPtr->Format());
}

std::vector<std::pair<uint32_t, uint32_t>> GenerateOutputInputRefPair(const ir::Value* output) {
  std::vector<std::pair<uint32_t, uint32_t>> result;

  if (output->IsTuple()) {
    const auto numOutput = output->ToTuple()->Size();
    result.reserve(numOutput);
    for (uint32_t i = 0; i < numOutput; ++i) {
      result.emplace_back(i, 0);
    }
  } else if (output->IsTensor()) {
    result.emplace_back(0, 0);
  } else {
    RT_GLOG(EXCEPTION) << "Output is not a tensor or tuple";
  }

  return result;
}
} // namespace ops
} // namespace fxrt
