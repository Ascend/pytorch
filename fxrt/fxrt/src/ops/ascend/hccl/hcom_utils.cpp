#include "ops/ascend/hccl/hcom_utils.h"

#include <set>
#include <algorithm>
#include <memory>
#include <utility>

namespace fxrt::ops {

inline int64_t LongMulWithOverflowCheck(int64_t a, int64_t b) {
  int64_t out = a * b;
  if (a != 0) {
    bool overflow = ((out / a) != b);
    if (overflow) {
      RT_GLOG(EXCEPTION) << "Mul: a(" << a << ") * b(" << b << ") result is overflow";
    }
  }
  return out;
}

inline size_t SizetMulWithOverflowCheck(size_t a, size_t b) {
  size_t out = a * b;
  if (a != 0) {
    if ((out / a) != b) {
      RT_GLOG(EXCEPTION) << "Mul: a(" << a << ") * b(" << b << ") result is overflow";
    }
  }
  return out;
}

inline size_t LongToSizeClipNeg(int64_t u) {
  return u < 0 ? 0 : static_cast<size_t>(u);
}

::HcclDataType HcomUtil::ConvertHcclType(DataType typeId) {
  auto iter = kConstOpHcomDataTypeMap.find(typeId);
  if (iter == kConstOpHcomDataTypeMap.end()) {
    RT_GLOG(EXCEPTION) << "HcomDataType can't support Current Ascend Data Type : " << typeId.ToString();
  }
  return iter->second;
}

bool HcomUtil::GetHcclOpSize(const HcclDataType& dataType, const std::vector<int64_t>& shape, size_t* size) {
  CHECK_IF_NULL(size);
  int64_t tmpSize = 1;
  uint32_t typeSize = 4;
  for (size_t i = 0; i < shape.size(); i++) {
    tmpSize = LongMulWithOverflowCheck(tmpSize, shape[i]);
  }

  if (!GetHcomTypeSize(dataType, &typeSize)) {
    return false;
  }

  *size = SizetMulWithOverflowCheck(LongToSizeClipNeg(tmpSize), typeSize);
  return true;
}

bool HcomUtil::GetHcomTypeSize(const HcclDataType& dataType, uint32_t* size) {
  CHECK_IF_NULL(size);
  auto iter = kConstOpHcomDataTypeSizeMap.find(dataType);
  if (iter == kConstOpHcomDataTypeSizeMap.end()) {
    RT_GLOG(ERROR) << "HcomUtil::HcomDataTypeSize, No DataTypeSize!";
    return false;
  }
  *size = iter->second;
  return true;
}

bool HcomUtil::GetHcomCount(
    const std::vector<HcclDataType>& dataTypeList,
    const std::vector<std::vector<int64_t>>& shapeList,
    const size_t inputTensorNum,
    const std::optional<int64_t> rankSizeOpt,
    uint64_t* totalCount) {
  CHECK_IF_NULL(totalCount);

  const uint32_t alignSize = 512;
  const uint32_t filledSize = 32;
  uint64_t totalSize = 0;
  size_t inputSize;
  uint32_t typeSize = 4;
  size_t rankSize = 1;
  bool isReduceScatter = false;
  if (rankSizeOpt.has_value()) {
    if (rankSizeOpt.value() <= 0) {
      RT_GLOG(ERROR) << "Get invalid rank size: " << rankSizeOpt.value();
      return false;
    }
    rankSize = static_cast<size_t>(rankSizeOpt.value());
    isReduceScatter = true;
  }
  CHECK_IF_FAIL(dataTypeList.size() == shapeList.size());

  for (size_t i = 0; i < dataTypeList.size(); ++i) {
    if (!GetHcomTypeSize(dataTypeList[i], &typeSize)) {
      return false;
    }

    if (!GetHcclOpSize(dataTypeList[i], shapeList[i], &inputSize)) {
      RT_GLOG(ERROR) << "Get GetHcclOpSize failed";
      return false;
    }

    if (inputTensorNum > 1) {
      // communication operator with dynamic input should have continuous memory.
      inputSize = (inputSize + alignSize - 1 + filledSize) / alignSize * alignSize;
    }

    if (isReduceScatter) {
      inputSize /= rankSize;
    }
    bool allDynamic = std::all_of(shapeList[i].begin(), shapeList[i].end(), [](int64_t x) { return x == -1; });
    if (!allDynamic && (typeSize == 0 || inputSize % typeSize != 0)) {
      return false;
    }
    totalSize += inputSize / typeSize;
  }
  *totalCount = totalSize;
  return true;
}

std::pair<uint64_t, ::HcclDataType> HcomUtil::GetHcclCountAndTypeFromTensor(
    const ir::TensorPtr& tensor,
    const std::optional<int64_t> rankSizeOpt) {
  auto typeId = tensor->Dtype();
  auto shape = tensor->Shape();

  auto hcclType = ConvertHcclType(typeId);

  uint64_t hcclCount = 0;
  constexpr size_t inputTensorSize = 1;
  if (!GetHcomCount({hcclType}, {shape}, inputTensorSize, rankSizeOpt, &hcclCount)) {
    RT_GLOG(EXCEPTION) << "GetHcomCount fail!";
  }
  return std::make_pair(hcclCount, hcclType);
}

void HcomUtil::CheckHcclInputContiguous(const ir::TensorPtr& tensor, const std::string& opName) {
  CHECK_IF_NULL(tensor);
  if (!tensor->IsContiguous()) {
    RT_GLOG(EXCEPTION) << opName << " does not support non-contiguous input tensor, shape: " << tensor->Shape()
                       << ", strides: " << tensor->Strides();
  }
}

CollectiveOpReduceType HcomUtil::GetCollectiveOpReduceType(const std::string& reduceOp) {
  auto iter = kConstOpCollectiveOpReduceTypeMap.find(reduceOp);
  if (iter == kConstOpCollectiveOpReduceTypeMap.end()) {
    RT_GLOG(EXCEPTION) << "HcomUtil::Get CollectiveOpReduceType fail, [" << reduceOp << "] not support!";
  }
  return iter->second;
}

HcclReduceOp HcomUtil::GetHcomReduceOpType(const std::string& reduceOp) {
  auto iter = kConstOpHcomReduceOpTypeMap.find(reduceOp);
  if (iter == kConstOpHcomReduceOpTypeMap.end()) {
    RT_GLOG(EXCEPTION) << "HcomUtil::Get HCOM_ATTR_REDUCE_TYPE fail, [" << reduceOp << "] not support!";
  }
  return iter->second;
}

} // namespace fxrt::ops
