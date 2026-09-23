#ifndef OPS_ASCEND_HCCL_HCOM_UTILS_H_
#define OPS_ASCEND_HCCL_HCOM_UTILS_H_

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <unordered_map>
#include <utility>
#include <optional>

#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "ir/common/dtype.h"
#include "ir/tensor/tensor.h"
#include "common/common.h"

#include "hccl/hccl_types.h"

namespace fxrt::ops {
using fxrt::ir::TensorPtr;
using ir::DataType;
using std::map;
using std::string;
using std::vector;
constexpr int64_t kComplex64ConvertFloat32Num = 2;

enum CollectiveOpReduceType : int64_t {
  Reduce_Mean = 0,
  Reduce_Max = 1,
  Reduce_Min = 2,
  Reduce_Prod = 3,
  Reduce_Sum = 4,
  Reduce_Sum_Square = 5,
  Reduce_ASum = 6,
  Reduce_All = 7
};

/* Correspondence between data_type and hcom data type in Ascend */
static const map<int64_t, HcclDataType> kConstOpHcomDataTypeMap = {
    {DataType::Int8, HCCL_DATA_TYPE_INT8},
    {DataType::Int16, HCCL_DATA_TYPE_INT16},
    {DataType::Int32, HCCL_DATA_TYPE_INT32},
    {DataType::Float32, HCCL_DATA_TYPE_FP32},
    {DataType::Int64, HCCL_DATA_TYPE_INT64},
    {DataType::UInt8, HCCL_DATA_TYPE_UINT8},
    {DataType::Float64, HCCL_DATA_TYPE_FP64},
    {DataType::Bool, HCCL_DATA_TYPE_INT8},
    {DataType::Float16, HCCL_DATA_TYPE_FP16},
    {DataType::BFloat16, HCCL_DATA_TYPE_BFP16},
#ifdef EXPERIMENT_A5
    {DataType::kNumberTypeHiFloat8, HCCL_DATA_TYPE_HIF8},
    {DataType::kNumberTypeFloat8E5M2, HCCL_DATA_TYPE_FP8E5M2},
    {DataType::kNumberTypeFloat8E4M3FN, HCCL_DATA_TYPE_FP8E4M3},
#endif
};

/* Correspondence between data_type and occupied byte size in hcom */
static const map<HcclDataType, uint32_t> kConstOpHcomDataTypeSizeMap = {
    {HCCL_DATA_TYPE_INT8, sizeof(int8_t)},
    {HCCL_DATA_TYPE_INT16, sizeof(int32_t) / 2},
    {HCCL_DATA_TYPE_INT32, sizeof(int32_t)},
    {HCCL_DATA_TYPE_FP16, sizeof(float) / 2},
    {HCCL_DATA_TYPE_FP32, sizeof(float)},
    {HCCL_DATA_TYPE_INT64, sizeof(int64_t)},
    {HCCL_DATA_TYPE_UINT64, sizeof(uint64_t)},
    {HCCL_DATA_TYPE_UINT8, sizeof(uint8_t)},
    {HCCL_DATA_TYPE_UINT16, sizeof(uint32_t) / 2},
    {HCCL_DATA_TYPE_UINT32, sizeof(uint32_t)},
    {HCCL_DATA_TYPE_FP64, sizeof(double)},
    {HCCL_DATA_TYPE_BFP16, sizeof(float) / 2},
#ifdef EXPERIMENT_A5
    {HCCL_DATA_TYPE_HIF8, sizeof(float) / 4},
    {HCCL_DATA_TYPE_FP8E5M2, sizeof(float) / 4},
    {HCCL_DATA_TYPE_FP8E4M3, sizeof(float) / 4},
#endif
};

static const std::map<CollectiveOpReduceType, HcclReduceOp> kHcomOpReduceTypeMap = {
    {CollectiveOpReduceType::Reduce_Max, HCCL_REDUCE_MAX},
    {CollectiveOpReduceType::Reduce_Min, HCCL_REDUCE_MIN},
    {CollectiveOpReduceType::Reduce_Prod, HCCL_REDUCE_PROD},
    {CollectiveOpReduceType::Reduce_Sum, HCCL_REDUCE_SUM}};

/* Correspondence between reduce str and enum in hcom  */
static const std::unordered_map<std::string, HcclReduceOp> kConstOpHcomReduceOpTypeMap = {
    {"min", HCCL_REDUCE_MIN},
    {"max", HCCL_REDUCE_MAX},
    {"prod", HCCL_REDUCE_PROD},
    {"sum", HCCL_REDUCE_SUM},
};

/* Correspondence between reduce str and enum in collective op  */
static const std::unordered_map<std::string, CollectiveOpReduceType> kConstOpCollectiveOpReduceTypeMap = {
    {"min", CollectiveOpReduceType::Reduce_Min},
    {"max", CollectiveOpReduceType::Reduce_Max},
    {"prod", CollectiveOpReduceType::Reduce_Prod},
    {"sum", CollectiveOpReduceType::Reduce_Sum},
};

class HcomUtil {
 public:
  static ::HcclDataType ConvertHcclType(DataType typeId);
  static HcclComm LoadHcclLibrary(const std::string& groupName) {
    int64_t hcclComm = collective::CollectiveManager::Instance().GetCommunicationGroup(groupName)->communicator();
    return reinterpret_cast<HcclComm>(static_cast<intptr_t>(hcclComm));
  }
  // static bool GetHcomDataType(const std::string &kernel_name, const std::vector<TensorPtr> &inputs,
  //                             const std::vector<TensorPtr> &outputs, std::vector<HcclDataType> *data_type_list);
  static bool GetHcclOpSize(const HcclDataType& dataType, const std::vector<int64_t>& shape, size_t* size);
  static bool GetHcomTypeSize(const HcclDataType& dataType, uint32_t* size);
  static bool GetHcomCount(
      const std::vector<HcclDataType>& dataTypeList,
      const std::vector<std::vector<int64_t>>& shapeList,
      const size_t inputTensorNum,
      const std::optional<int64_t> rankSizeOpt,
      uint64_t* totalCount);

  static std::pair<uint64_t, ::HcclDataType> GetHcclCountAndTypeFromTensor(
      const ir::TensorPtr& tensor,
      const std::optional<int64_t> rankSizeOpt = std::nullopt);
  static void CheckHcclInputContiguous(const ir::TensorPtr& tensor, const std::string& opName);
  static CollectiveOpReduceType GetCollectiveOpReduceType(const std::string& reduceOp);
  static HcclReduceOp GetHcomReduceOpType(const std::string& reduceOp);
};
} // namespace fxrt::ops

#endif // OPS_ASCEND_HCCL_HCOM_UTILS_H_
