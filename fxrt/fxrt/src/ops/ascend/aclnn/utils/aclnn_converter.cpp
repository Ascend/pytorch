#include <map>

#include "ops/ascend/aclnn/utils/aclnn_converter.h"

namespace fxrt {
namespace ops {
static const std::map<ir::DataType::Type, aclDataType> kDataTypeToAclDataTypeMap = {
    {ir::DataType::Type::Unknown, ACL_DT_UNDEFINED},
    {ir::DataType::Type::Float16, ACL_FLOAT16},
    {ir::DataType::Type::BFloat16, ACL_BF16},
    {ir::DataType::Type::Float32, ACL_FLOAT},
    {ir::DataType::Type::Float64, ACL_DOUBLE},
    {ir::DataType::Type::Complex64, ACL_COMPLEX64},
    {ir::DataType::Type::Int8, ACL_INT8},
    {ir::DataType::Type::Int16, ACL_INT16},
    {ir::DataType::Type::Int32, ACL_INT32},
    {ir::DataType::Type::Int64, ACL_INT64},
    {ir::DataType::Type::UInt8, ACL_UINT8},
    {ir::DataType::Type::Bool, ACL_BOOL},
    {ir::DataType::Type::QInt8, ACL_DT_UNDEFINED},
    {ir::DataType::Type::QUInt4x2, ACL_DT_UNDEFINED},
};

aclDataType Convert(ir::DataType::Type dtype) {
  auto iter = kDataTypeToAclDataTypeMap.find(dtype);
  if (iter == kDataTypeToAclDataTypeMap.end()) {
    RT_GLOG(EXCEPTION) << "Invalid dtype: " << dtype;
  }
  auto ret = iter->second;
  if (ret == ACL_DT_UNDEFINED) {
    RT_GLOG(EXCEPTION) << "Invalid dtype: " << dtype;
  }
  return ret;
}

template <typename T>
aclScalar* CreateAclScalar(T val, aclDataType dtype) {
  static const auto aclCreateScalar = GET_ACLNN_COMMON_META_FUNC(aclCreateScalar);
  CHECK_IF_NULL(aclCreateScalar);
  return aclCreateScalar(&val, dtype);
}

aclScalar* Convert(const ir::Value* value) {
  if (value == nullptr) {
    return nullptr;
  }
  if (value->IsInt() || value->IsSymbol()) {
    return CreateAclScalar(value->ToInt(), ACL_INT64);
  }
  if (value->IsDouble()) {
    return CreateAclScalar(value->ToDouble(), ACL_DOUBLE);
  }
  if (value->IsBool()) {
    return CreateAclScalar(value->ToBool(), ACL_BOOL);
  }
  RT_GLOG(EXCEPTION) << "Invalid value, value: " << value << ", type: " << TagToString(value->GetTag());
  return nullptr;
}
} // namespace ops
} // namespace fxrt
