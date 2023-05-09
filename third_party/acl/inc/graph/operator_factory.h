#ifndef INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_
#define INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "./operator.h"
#include "./ge_error_codes.h"

namespace ge {
using OpCreator = std::function<Operator(const std::string &)>;
using OpCreatorV2 = std::function<Operator(const AscendString &)>;
using InferShapeFunc = std::function<graphStatus(Operator &)>;
using InferFormatFunc = std::function<graphStatus(Operator &)>;
using VerifyFunc = std::function<graphStatus(Operator &)>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorFactory {
 public:
  ATTRIBUTED_DEPRECATED(static Operator CreateOperator(const char *, const char *))
  static Operator CreateOperator(const std::string &operator_name, const std::string &operator_type);

  static Operator CreateOperator(const char *operator_name, const char *operator_type);

  ATTRIBUTED_DEPRECATED(graphStatus GetOpsTypeList(std::vector<AscendString> &))
  static graphStatus GetOpsTypeList(std::vector<std::string> &all_ops);

  static graphStatus GetOpsTypeList(std::vector<AscendString> &all_ops);

  ATTRIBUTED_DEPRECATED(bool IsExistOp(const char *))
  static bool IsExistOp(const string &operator_type);

  static bool IsExistOp(const char *operator_type);
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorCreatorRegister {
 public:
  ATTRIBUTED_DEPRECATED(OperatorCreatorRegister(const char *, OpCreatorV2 const &))
  OperatorCreatorRegister(const string &operator_type, OpCreator const &op_creator);
  OperatorCreatorRegister(const char *operator_type, OpCreatorV2 const &op_creator);
  ~OperatorCreatorRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferShapeFuncRegister {
 public:
  ATTRIBUTED_DEPRECATED(InferShapeFuncRegister(const char *, const InferShapeFunc &))
  InferShapeFuncRegister(const std::string &operator_type, const InferShapeFunc &infer_shape_func);
  InferShapeFuncRegister(const char *operator_type, const InferShapeFunc &infer_shape_func);
  ~InferShapeFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferFormatFuncRegister {
 public:
  ATTRIBUTED_DEPRECATED(InferFormatFuncRegister(const char *, const InferFormatFunc &))
  InferFormatFuncRegister(const std::string &operator_type, const InferFormatFunc &infer_format_func);
  InferFormatFuncRegister(const char *operator_type, const InferFormatFunc &infer_format_func);
  ~InferFormatFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY VerifyFuncRegister {
 public:
  ATTRIBUTED_DEPRECATED(VerifyFuncRegister(const char *, const VerifyFunc &))
  VerifyFuncRegister(const std::string &operator_type, const VerifyFunc &verify_func);
  VerifyFuncRegister(const char *operator_type, const VerifyFunc &verify_func);
  ~VerifyFuncRegister() = default;
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_
