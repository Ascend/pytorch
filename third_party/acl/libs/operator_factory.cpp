#include "graph/operator_factory.h"

namespace ge {
Operator OperatorFactory::CreateOperator(
    const char* operator_name,
    const char* operator_type) {
  return Operator();
}

OperatorCreatorRegister::OperatorCreatorRegister(
    const char* operator_type,
    const OpCreatorV2& op_creator) {}

bool OperatorFactory::IsExistOp(const char* operator_type) {
  return true;
}
} // namespace ge