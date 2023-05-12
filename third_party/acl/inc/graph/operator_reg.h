#ifndef INC_EXTERNAL_GRAPH_OPERATOR_REG_H_
#define INC_EXTERNAL_GRAPH_OPERATOR_REG_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "graph/graph.h"

namespace ge {
using std::function;
using std::string;
using std::vector;

#define ATTR_String(x, ...)                                                 \
  graphStatus get_attr_##x(AscendString &ret) const {                       \
    string ret_str = __VA_ARGS__;                                           \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      ret = AscendString(ret_str.c_str());                                  \
    }                                                                       \
    return GRAPH_SUCCESS;                                                   \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const char *v) {                                 \
    Operator::SetAttr(#x, v);                                               \
    return *this;                                                           \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const function<AscendString()> &v) { return *this; }

#define ATTR_ListString(x, ...)                                             \
  graphStatus get_attr_##x(vector<AscendString> &ret) const {               \
    vector<string> ret_strs = __VA_ARGS__;                                  \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      for (auto &ret_str : ret_strs) {                                      \
        ret.emplace_back(ret_str.c_str());                                  \
      }                                                                     \
    }                                                                       \
    return GRAPH_SUCCESS;                                                   \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const vector<AscendString> &v) {                 \
    Operator::SetAttr(#x, v);                                               \
    return *this;                                                           \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const function<vector<AscendString>()> &v) {     \
    return *this; }

#define ATTR_AscendString(x, ...)                                           \
  graphStatus get_attr_##x(AscendString &ret) const {                       \
    AscendString ret_str = __VA_ARGS__;                                     \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      ret = AscendString(ret_str.c_str());                                  \
    }                                                                       \
    return GRAPH_SUCCESS;                                                   \
  }

#define ATTR_ListAscendString(x, ...)                                       \
  graphStatus get_attr_##x(vector<AscendString> &ret) const {               \
    vector<AscendString> ret_strs = __VA_ARGS__;                            \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      for (auto &ret_str : ret_strs) {                                      \
        if (ret_str.GetString() != nullptr) {                               \
          ret.emplace_back(ret_str.GetString());                            \
        }                                                                   \
      }                                                                     \
    }                                                                       \
    return GRAPH_SUCCESS;                                                   \
  }

#define ATTR_Int(x, ...)
#define ATTR_Float(x, ...)
#define ATTR_Bool(x, ...)
#define ATTR_Tensor(x, ...)
#define ATTR_Type(x, ...)
#define ATTR_NamedAttrs(x, ...)
#define ATTR_ListInt(x, ...)
#define ATTR_ListFloat(x, ...)
#define ATTR_ListBool(x, ...)
#define ATTR_ListTensor(x, ...)
#define ATTR_Bytes(x, ...)
#define ATTR_ListListInt(x, ...)
#define ATTR_ListType(x, ...)
#define ATTR_ListNamedAttrs(x, ...)

#define REGISTER_TYPE_AscendString(...) AscendString(__VA_ARGS__)
#define REGISTER_TYPE_String(...) REGISTER_TYPE_AscendString(__VA_ARGS__)

#define REGISTER_TYPE_Int(...) OpInt(__VA_ARGS__)
#define REGISTER_TYPE_Tensor(...) OpTensor(__VA_ARGS__)
#define REGISTER_TYPE_Bool(...) OpBool(__VA_ARGS__)

#define REQUIRED_ATTR_String(x)                                             \
  graphStatus get_attr_##x(AscendString &ret) const {                       \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      return GRAPH_FAILED;                                                  \
    }                                                                       \
    return GRAPH_SUCCESS;                                                   \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const char *v) {                                 \
    Operator::SetAttr(#x, v);                                               \
    return *this;                                                           \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const function<AscendString()> &v) { return *this; }

#define REQUIRED_ATTR_ListString(x)                                         \
  graphStatus get_attr_##x(vector<AscendString> &ret) const {               \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      return GRAPH_FAILED;                                                  \
    }                                                                       \
    return GRAPH_SUCCESS;                                                   \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const vector<AscendString> &v) {                 \
    Operator::SetAttr(#x, v);                                               \
    return *this;                                                           \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const function<vector<AscendString>()> &v) {     \
    return *this; }

#define REQUIRED_ATTR_AscendString(x)                                       \
  graphStatus get_attr_##x(AscendString &ret) const {                       \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      return GRAPH_FAILED                                                   \
    }                                                                       \
    return GRAPH_SUCCESS;                                                   \
  }

#define REQUIRED_ATTR_ListAscendString(x)                                   \
  graphStatus get_attr_##x(vector<AscendString> &ret) const {               \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      return GRAPH_FAILED;                                                  \
    }                                                                       \
    return GRAPH_SUCCESS;                                                   \
  }

#define REQUIRED_ATTR_Int(x)
#define REQUIRED_ATTR_Float(x)
#define REQUIRED_ATTR_Bool(x)
#define REQUIRED_ATTR_Tensor(x)
#define REQUIRED_ATTR_Type(x)
#define REQUIRED_ATTR_NamedAttrs(x)
#define REQUIRED_ATTR_ListInt(x)
#define REQUIRED_ATTR_ListFloat(x)
#define REQUIRED_ATTR_ListBool(x)
#define REQUIRED_ATTR_ListTensor(x)
#define REQUIRED_ATTR_Bytes(x)
#define REQUIRED_ATTR_ListListInt(x)
#define REQUIRED_ATTR_ListType(x)
#define REQUIRED_ATTR_ListNamedAttrs(x)

class OpReg {
 public:
  OpReg &N() { return *this; }

  OpReg &ATTR() { return *this; }

  OpReg &REQUIRED_ATTR() { return *this; }

  OpReg &INPUT() { return *this; }

  OpReg &OPTIONAL_INPUT() { return *this; }

  OpReg &OUTPUT() { return *this; }

  OpReg &GRAPH() { return *this; }

  OpReg &DYNAMIC_GRAPH() { return *this; }

  OpReg &INFER_SHAPE_AND_TYPE() { return *this; }
};

#define REG_OP(x)                                                    \
  namespace op {                                                     \
  class x : public Operator {                                        \
    typedef x _THIS_TYPE;                                            \
                                                                     \
   public:                                                           \
   ATTRIBUTED_DEPRECATED(x(const char *))                            \
    explicit x(const string &name) : Operator(name.c_str(), #x) { __##x(); } \
    explicit x(const char *name) : Operator(name, #x) { __##x(); }   \
    explicit x(const AscendString &name) : Operator(name, #x) {      \
      __##x(); }                                                     \
    x() : Operator(#x) { __##x(); }                                  \
                                                                     \
   private:                                                          \
    void __##x() {                                                   \
    OpReg()

#define ATTR(x, Type, ...)                                                  \
  N();                                                                      \
  __attr_##x();                                                             \
  }                                                                         \
                                                                            \
 public:                                                                    \
  ATTRIBUTED_DEPRECATED(static const void name_attr_##x(AscendString &))    \
  static const string name_attr_##x() { return #x; }                        \
  static const void name_attr_##x(AscendString &attr) {                     \
    attr = AscendString(#x);                                                \
  }                                                                         \
  ATTR_##Type(x, __VA_ARGS__)                                               \
  Op##Type get_attr_##x() const {                                           \
    Op##Type ret = __VA_ARGS__;                                             \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      return ret;                                                           \
    }                                                                       \
    return ret;                                                             \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const Op##Type &v) {                             \
    Operator::SetAttr(#x, v);                                               \
    return *this;                                                           \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const function<Op##Type()> &v) { return *this; } \
                                                                            \
 private:                                                                   \
  void __attr_##x() {                                                       \
    Operator::AttrRegister(#x, REGISTER_TYPE_##Type(__VA_ARGS__));          \
    string attr_name(#x);                                                   \
    (void)OpReg()

#define REQUIRED_ATTR(x, Type)                                              \
  N();                                                                      \
  __required_attr_##x();                                                    \
  }                                                                         \
                                                                            \
 public:                                                                    \
  ATTRIBUTED_DEPRECATED(static const void name_attr_##x(AscendString &))    \
  static const string name_attr_##x() { return #x; }                        \
  static const void name_attr_##x(AscendString &attr_name) {                \
    attr_name = AscendString(#x);                                           \
  }                                                                         \
  REQUIRED_ATTR_##Type(x)                                                   \
  Op##Type get_attr_##x() const {                                           \
    Op##Type ret;                                                           \
    if (Operator::GetAttr(#x, ret) == GRAPH_FAILED) {                       \
      return ret;                                                           \
    }                                                                       \
    return ret;                                                             \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const Op##Type &v) {                             \
    Operator::SetAttr(#x, v);                                               \
    return *this;                                                           \
  }                                                                         \
  _THIS_TYPE &set_attr_##x(const function<Op##Type()> &v) { return *this; } \
                                                                            \
 private:                                                                   \
  void __required_attr_##x() {                                              \
    Operator::RequiredAttrRegister(#x);                                     \
    string attr_name(#x);                                                   \
    (void)OpReg()

#define INPUT(x, t)                                                            \
  N();                                                                         \
  __input_##x();                                                               \
  }                                                                            \
                                                                               \
 public:                                                                       \
  ATTRIBUTED_DEPRECATED(static const void name_in_##x(AscendString &))         \
  static const string name_in_##x() { return #x; }                             \
  static const void name_in_##x(AscendString &name) {                          \
    name = AscendString(#x);                                                   \
  }                                                                            \
  ATTRIBUTED_DEPRECATED(_THIS_TYPE &set_input_##x##_by_name(Operator &, const char *)) \
  _THIS_TYPE &set_input_##x(Operator &v, const string &srcName) {              \
    Operator::SetInput(#x, v, srcName.c_str());                                \
    return *this;                                                              \
  }                                                                            \
  _THIS_TYPE &set_input_##x##_by_name(Operator &v, const char *srcName) {      \
    Operator::SetInput(#x, v, srcName);                                        \
    return *this;                                                              \
  }                                                                            \
  _THIS_TYPE &set_input_##x(Operator &v, uint32_t index) {                     \
    Operator::SetInput(#x, v, index);                                          \
    return *this;                                                              \
  }                                                                            \
  _THIS_TYPE &set_input_##x(Operator &v) {                                     \
    Operator::SetInput(#x, v);                                                 \
    return *this;                                                              \
  }                                                                            \
  TensorDesc get_input_desc_##x() const { return Operator::GetInputDescByName(#x); } \
  graphStatus update_input_desc_##x(const TensorDesc &tensorDesc) {            \
    return Operator::UpdateInputDesc(#x, tensorDesc);                          \
  }                                                                            \
                                                                               \
 private:                                                                      \
  void __input_##x() {                                                         \
    Operator::InputRegister(#x);                                               \
    (void)OpReg()

#define OPTIONAL_INPUT(x, t)                                                   \
  N();                                                                         \
  __optional_input_##x();                                                      \
  }                                                                            \
                                                                               \
 public:                                                                       \
 ATTRIBUTED_DEPRECATED(static const void name_in_##x(AscendString &))          \
  static const string name_in_##x() { return #x; }                             \
  static const void name_in_##x(AscendString &name) {                          \
    name = AscendString(#x);                                                   \
  }                                                                            \
  _THIS_TYPE &set_input_##x(Operator &v) {                                     \
    Operator::SetInput(#x, v);                                                 \
    return *this;                                                              \
  }                                                                            \
  ATTRIBUTED_DEPRECATED(_THIS_TYPE &set_input_##x##_by_name(Operator &, const char *))  \
  _THIS_TYPE &set_input_##x(Operator &v, const string &srcName) {              \
    Operator::SetInput(#x, v, srcName.c_str());                                \
    return *this;                                                              \
  }                                                                            \
  _THIS_TYPE &set_input_##x##_by_name(Operator &v, const char *srcName) {      \
    Operator::SetInput(#x, v, srcName);                                        \
    return *this;                                                              \
  }                                                                            \
  _THIS_TYPE &set_input_##x(Operator &v, uint32_t index) {                     \
    Operator::SetInput(#x, v, index);                                          \
    return *this;                                                              \
  }                                                                            \
  TensorDesc get_input_desc_##x() const { return Operator::GetInputDescByName(#x); } \
  graphStatus update_input_desc_##x(const TensorDesc &tensorDesc) {            \
    return Operator::UpdateInputDesc(#x, tensorDesc);                          \
  }                                                                            \
                                                                               \
 private:                                                                      \
  void __optional_input_##x() {                                                \
    Operator::OptionalInputRegister(#x);                                       \
    (void)OpReg()

#define OUTPUT(x, t)                                                             \
  N();                                                                           \
  __out_##x();                                                                   \
  }                                                                              \
                                                                                 \
 public:                                                                         \
  ATTRIBUTED_DEPRECATED(static const void name_out_##x(AscendString &))          \
  static const string name_out_##x() { return #x; }                              \
  static const void name_out_##x(AscendString &name) {                           \
    name = AscendString(#x);                                                     \
  }                                                                              \
  TensorDesc get_output_desc_##x() const { return Operator::GetOutputDescByName(#x); } \
  graphStatus update_output_desc_##x(const TensorDesc &tensorDesc) {             \
    return Operator::UpdateOutputDesc(#x, tensorDesc);                           \
  }                                                                              \
                                                                                 \
 private:                                                                        \
  void __out_##x() {                                                             \
    Operator::OutputRegister(#x);                                                \
    (void)OpReg()

#define DYNAMIC_INPUT(x, t)                                                                   \
  N();                                                                                        \
  __dy_input_##x();                                                                           \
  }                                                                                           \
                                                                                              \
 public:                                                                                      \
  _THIS_TYPE &create_dynamic_input_##x(uint32_t num, bool isPushBack = true) {                \
    Operator::DynamicInputRegister(#x, num, isPushBack);                                      \
    return *this;                                                                             \
  }                                                                                           \
  _THIS_TYPE &create_dynamic_input_byindex_##x(uint32_t num, size_t index) {                  \
    Operator::DynamicInputRegisterByIndex(#x, num, index);                                    \
    return *this;                                                                             \
  }                                                                                           \
  TensorDesc get_dynamic_input_desc_##x(uint32_t index) const {                               \
    return Operator::GetDynamicInputDesc(#x, index);                                          \
  }                                                                                           \
  graphStatus update_dynamic_input_desc_##x(uint32_t index, const TensorDesc &tensorDesc) {   \
    return Operator::UpdateDynamicInputDesc(#x, index, tensorDesc);                           \
  }                                                                                           \
  _THIS_TYPE &set_dynamic_input_##x(uint32_t dstIndex, Operator &v) {                         \
    Operator::SetInput(#x, dstIndex, v);                                                      \
    return *this;                                                                             \
  }                                                                                           \
  ATTRIBUTED_DEPRECATED(_THIS_TYPE &set_dynamic_input_##x(uint32_t, Operator &, const char *))\
  _THIS_TYPE &set_dynamic_input_##x(uint32_t dstIndex, Operator &v, const string &srcName) {  \
    Operator::SetInput(#x, dstIndex, v, srcName.c_str());                                     \
    return *this;                                                                             \
  }                                                                                           \
  _THIS_TYPE &set_dynamic_input_##x(uint32_t dstIndex, Operator &v, const char *srcName) {    \
    Operator::SetInput(#x, dstIndex, v, srcName);                                             \
    return *this;                                                                             \
  }                                                                                           \
                                                                                              \
 private:                                                                                     \
  void __dy_input_##x() {                                                                     \
  Operator::DynamicInputRegister(#x, 0, true);                                                \
  (void)OpReg()

#define DYNAMIC_OUTPUT(x, t)                                                                  \
  N();                                                                                        \
  __dy_output_##x();                                                                          \
  }                                                                                           \
                                                                                              \
 public:                                                                                      \
  _THIS_TYPE &create_dynamic_output_##x(uint32_t num, bool isPushBack = true) {               \
    Operator::DynamicOutputRegister(#x, num, isPushBack);                                     \
    return *this;                                                                             \
  }                                                                                           \
  TensorDesc get_dynamic_output_desc_##x(uint32_t index) const {                              \
    return Operator::GetDynamicOutputDesc(#x, index);                                         \
  }                                                                                           \
  graphStatus update_dynamic_output_desc_##x(uint32_t index, const TensorDesc &tensorDesc) {  \
    return Operator::UpdateDynamicOutputDesc(#x, index, tensorDesc);                          \
  }                                                                                           \
                                                                                              \
 private:                                                                                     \
  void __dy_output_##x() {                                                                    \
  Operator::DynamicOutputRegister(#x, 0, true);                                               \
  (void)OpReg()

#define GRAPH(x)                                                                              \
  N();                                                                                        \
  __graph_##x();                                                                              \
  }                                                                                           \
                                                                                              \
 public:                                                                                      \
  ATTRIBUTED_DEPRECATED(static const void name_graph_##x(AscendString &))                     \
  static const string name_graph_##x() { return #x; }                                         \
  static const void name_graph_##x(AscendString &name) {                                      \
    name = AscendString(#x);                                                                  \
  }                                                                                           \
  SubgraphBuilder get_subgraph_builder_##x() const {                                          \
    return Operator::GetSubgraphBuilder(#x);                                                  \
  }                                                                                           \
  _THIS_TYPE &set_subgraph_builder_##x(const SubgraphBuilder &v) {                            \
    Operator::SetSubgraphBuilder(#x, 0, v);                                                   \
    return *this;                                                                             \
  }                                                                                           \
  Graph get_subgraph_##x() const {                                                            \
    return Operator::GetSubgraph(#x);                                                         \
  }                                                                                           \
                                                                                              \
 private:                                                                                     \
  void __graph_##x() {                                                                        \
    Operator::SubgraphRegister(#x, false);                                                    \
    Operator::SubgraphCountRegister(#x, 1);                                                   \
    (void)OpReg()

#define DYNAMIC_GRAPH(x)                                                                      \
  N();                                                                                        \
  __graph_##x();                                                                              \
  }                                                                                           \
                                                                                              \
 public:                                                                                      \
  ATTRIBUTED_DEPRECATED(static const void name_graph_##x(AscendString &))                     \
  static const string name_graph_##x() { return #x; }                                         \
  static const void name_graph_##x(AscendString &name) {                                      \
    name = AscendString(#x);                                                                  \
  }                                                                                           \
  _THIS_TYPE &create_dynamic_subgraph_##x(uint32_t num) {                                     \
    Operator::SubgraphCountRegister(#x, num);                                                 \
    return *this;                                                                             \
  }                                                                                           \
  SubgraphBuilder get_dynamic_subgraph_builder_##x(uint32_t index) const {                    \
    return Operator::GetDynamicSubgraphBuilder(#x, index);                                    \
  }                                                                                           \
  Graph get_dynamic_subgraph_##x(uint32_t index) const {                                      \
    return Operator::GetDynamicSubgraph(#x, index);                                           \
  }                                                                                           \
  _THIS_TYPE &set_dynamic_subgraph_builder_##x(uint32_t index,const SubgraphBuilder &v) {     \
    Operator::SetSubgraphBuilder(#x, index, v);                                               \
    return *this;                                                                             \
  }                                                                                           \
                                                                                              \
 private:                                                                                     \
  void __graph_##x() {                                                                        \
    Operator::SubgraphRegister(#x, true);                                                     \
    (void)OpReg()


#define PASTE(g_register, y) g_register##y
#define __OP_END_IMPL__(x, y)                                                                                      \
  N();                                                                                                             \
  }                                                                                                                \
  static_assert(                                                                                                   \
      std::is_same<x, _THIS_TYPE>::value,                                                                          \
      "The class name entered into the OP_END_FACTORY_REG needs to be the same as the operator name you define."); \
  }                                                                                                                \
  ;                                                                                                                \
  static const OperatorCreatorRegister PASTE(g_register, y)(#x, [](const AscendString &name) { return x(name); }); \
  }
#define OP_END_FACTORY_REG(x) __OP_END_IMPL__(x, __COUNTER__)

// Specialized shape inferencer macro

#define IMPLEMT_INFERFUNC(op_name, func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(op::op_name &op)

#define IMPLEMT_COMMON_INFERFUNC(func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(Operator &op)

#define IMPLEMT_INFERFORMAT_FUNC(op_name, func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(op::op_name &op)

// Specialized verifier macro

#define IMPLEMT_VERIFIER(op_name, func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(op::op_name op)

#define INFER_VERIFY_FUNC(op_name, x) [](Operator &v) { return x((op::op_name &)v); }

#define COMMON_INFER_VERIFY_FUNC(x) [](Operator &v) { return x(v); }

#define INFER_FORMAT_FUNC(op_name, x) [](Operator &v) { return x((op::op_name &)v); }

#define __INFER_FUNC_REG_IMPL__(op_name, x, n) static const InferShapeFuncRegister PASTE(if_register, n)(#op_name, x)

#define __VERIFY_FUNC_REG_IMPL__(op_name, x, n) static const VerifyFuncRegister PASTE(vf_register, n)(#op_name, x)
// Infer format func register
#define __INFER_FORMAT_FUNC_REG_IMPL__(op_name, x, n) \
  static const InferFormatFuncRegister PASTE(ff_register, n)(#op_name, x)

// Shape inferencer & verifier register macro

#define INFER_FUNC_REG(op_name, x) __INFER_FUNC_REG_IMPL__(op_name, INFER_VERIFY_FUNC(op_name, x), __COUNTER__)

#define COMMON_INFER_FUNC_REG(op_name, x) __INFER_FUNC_REG_IMPL__(op_name, COMMON_INFER_VERIFY_FUNC(x), __COUNTER__)

#define VERIFY_FUNC_REG(op_name, x) __VERIFY_FUNC_REG_IMPL__(op_name, INFER_VERIFY_FUNC(op_name, x), __COUNTER__)

// Infer format func reg
#define INFER_FORMAT_FUNC_REG(op_name, x) \
  __INFER_FORMAT_FUNC_REG_IMPL__(op_name, INFER_FORMAT_FUNC(op_name, x), __COUNTER__)

// Common shape inferencer

#define ELMTWISE_INFER_SHAPEANDTYPE(in_name, out_name)            \
  [](Operator op)->graphStatus {                                  \
    auto x_shape = op.GetInputDescByName(in_name).GetShape().GetDims(); \
    auto x_type = op.GetInputDescByName(in_name).GetDataType();         \
    TensorDesc op_output_desc = op.GetOutputDescByName(out_name); \
    op_output_desc.SetShape(ge::Shape(x_shape));                  \
    op_output_desc.SetOriginShape(ge::Shape(x_shape));            \
    op_output_desc.SetDataType(x_type);                           \
    return op.UpdateOutputDesc(out_name, op_output_desc);         \
  }

graphStatus BroadCastInfer(const function<vector<int64_t>()> &get_in1_shape,
                           const function<vector<int64_t>()> &get_in2_shape,
                           const function<void(const vector<int64_t> &y_shape)> &set_out_shape);

#define BROADCAST_INFER(in1_name, in2_name, out_name)                                       \
  [](Operator op) -> graphStatus {                                                          \
    return BroadCastInfer([&]() { return op.GetInputDescByName(in1_name).GetShape().GetDims(); }, \
                          [&]() { return op.GetInputDescByName(in2_name).GetShape().GetDims(); }, \
                          [&](const vector<int64_t> &y_shape) {                             \
                            TensorDesc op_output_desc = op.GetOutputDescByName(out_name);   \
                            op_output_desc.SetShape(ge::Shape(y_shape));                    \
                            (void)op.UpdateOutputDesc(out_name, op_output_desc);});         \
  }
}  // namespace ge
#endif  // INC_EXTERNAL_GRAPH_OPERATOR_REG_H_
