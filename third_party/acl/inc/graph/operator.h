#ifndef INC_EXTERNAL_GRAPH_OPERATOR_H_
#define INC_EXTERNAL_GRAPH_OPERATOR_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "./ge_error_codes.h"
#include "./inference_context.h"
#include "./tensor.h"

#ifndef USER_GE_LOGI
#define USER_GE_LOGI(...)
#endif  // USER_GE_LOGI

#ifndef USER_GE_LOGW
#define USER_GE_LOGW(...)
#endif  // USER_GE_LOGW

#ifndef USER_GE_LOGE
#define USER_GE_LOGE(...)
#endif  // USER_GE_LOGE

#define DYNAMIC_OUTPUT_TD_NUM(name) ("__dynamic_output_" + name + "_cnt")
#define DYNAMIC_INPUT_TD_NUM(name) ("__dynamic_input_" + name + "_cnt")

namespace ge {
class Operator;
class OperatorImpl;
class NodeUtils;
class NamedAttrs;
class Graph;
class AttrValue;
class Node;

using SubgraphBuilder = std::function<Graph()>;
using OperatorImplPtr = std::shared_ptr<OperatorImpl>;
using OperatorPtr = std::shared_ptr<Operator>;

class OpIO;
using OutHandler = std::shared_ptr<OpIO>;
using InHandler = std::shared_ptr<OpIO>;

using std::function;
using std::shared_ptr;
using std::string;

/*lint -e148*/
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator {
 public:
  friend class OperatorImpl;
  friend class GraphBuilderImpl;
  friend class NodeUtils;
  friend class OpDescUtils;
  friend class GraphUtils;

  using OpInt = int64_t;
  using OpFloat = float;
  using OpString = string;
  using OpAscendString = AscendString;
  using OpBool = bool;
  using OpTensor = Tensor;
  using OpType = ge::DataType;
  using OpNamedAttrs = ge::NamedAttrs;
  using OpListInt = std::vector<int64_t>;
  using OpListFloat = std::vector<float>;
  using OpListString = std::vector<std::string>;
  using OpListAcendString = std::vector<AscendString>;
  using OpListBool = std::vector<bool>;
  using OpListTensor = std::vector<Tensor>;
  using OpBytes = std::vector<uint8_t>;
  using OpListListInt = std::vector<std::vector<int64_t>>;
  using OpListType = std::vector<ge::DataType>;
  using OpListNamedAttrs = std::vector<ge::NamedAttrs>;

  Operator() {}
  ATTRIBUTED_DEPRECATED(Operator(const char *))
  explicit Operator(const std::string &type);

  explicit Operator(const char *type);

  ATTRIBUTED_DEPRECATED(Operator(const char *, const char *))
  Operator(const std::string &name, const std::string &type);

  Operator(const AscendString &name, const AscendString &type);

  Operator(const char *name, const char *type);

  virtual ~Operator() = default;

  bool IsEmpty() const;

  ATTRIBUTED_DEPRECATED(graphStatus GetName(AscendString &) const)
  std::string GetName() const;

  graphStatus GetName(AscendString &name) const;

  ATTRIBUTED_DEPRECATED(graphStatus GetOpType(AscendString &) const)
  std::string GetOpType() const;

  graphStatus GetOpType(AscendString &type) const;

  // Only has one output index = 0
  ATTRIBUTED_DEPRECATED(Operator &SetInput(const char *, const Operator &))
  Operator &SetInput(const std::string &dst_name, const Operator &src_oprt);

  Operator &SetInput(const char *dst_name, const Operator &src_oprt);

  ATTRIBUTED_DEPRECATED(Operator &SetInput(const char *, const Operator &, const char *))
  Operator &SetInput(const std::string &dst_name, const Operator &src_oprt, const std::string &name);

  Operator &SetInput(const char *dst_name, const Operator &src_oprt, const char *name);

  ATTRIBUTED_DEPRECATED(Operator &SetInput(const char *, const Operator &, uint32_t))
  Operator &SetInput(const std::string &dst_name, const Operator &src_oprt, uint32_t index);

  Operator &SetInput(const char *dst_name, const Operator &src_oprt, uint32_t index);

  Operator &SetInput(uint32_t dst_index, const Operator &src_oprt, uint32_t src_index);

  Operator &AddControlInput(const Operator &src_oprt);

  ATTRIBUTED_DEPRECATED(graphStatus GetInputConstData(const char *, Tensor &) const)
  graphStatus GetInputConstData(const std::string &dst_name, Tensor &data) const;

  graphStatus GetInputConstData(const char *dst_name, Tensor &data) const;

  ATTRIBUTED_DEPRECATED(TensorDesc GetInputDescByName(const char *) const)
  TensorDesc GetInputDesc(const std::string &name) const;

  TensorDesc GetInputDescByName(const char *name) const;

  TensorDesc GetInputDesc(uint32_t index) const;

  ATTRIBUTED_DEPRECATED(int GetDynamicOutputNum(const char *) const)
  int GetDynamicOutputNum(const std::string &name) const;

  int GetDynamicOutputNum(const char *name) const;

  ATTRIBUTED_DEPRECATED(int GetDynamicInputNum(const char *))
  int GetDynamicInputNum(const std::string &name) const;

  int GetDynamicInputNum(const char *name) const;

  ATTRIBUTED_DEPRECATED(graphStatus TryGetInputDesc(const char *, TensorDesc &) const)
  graphStatus TryGetInputDesc(const std::string &name, TensorDesc &tensor_desc) const;

  graphStatus TryGetInputDesc(const char *name, TensorDesc &tensor_desc) const;

  ATTRIBUTED_DEPRECATED(graphStatus UpdateInputDesc(const char *, const TensorDesc &))
  graphStatus UpdateInputDesc(const std::string &name, const TensorDesc &tensor_desc);

  graphStatus UpdateInputDesc(const char *name, const TensorDesc &tensor_desc);

  ATTRIBUTED_DEPRECATED(TensorDesc GetOutputDescByName(const char *) const)
  TensorDesc GetOutputDesc(const std::string &name) const;

  TensorDesc GetOutputDescByName(const char *name) const;

  TensorDesc GetOutputDesc(uint32_t index) const;

  ATTRIBUTED_DEPRECATED(graphStatus UpdateOutputDesc(const char *, const TensorDesc &tensor_desc))
  graphStatus UpdateOutputDesc(const std::string &name, const TensorDesc &tensor_desc);

  graphStatus UpdateOutputDesc(const char *name, const TensorDesc &tensor_desc);

  ATTRIBUTED_DEPRECATED(TensorDesc GetDynamicInputDesc(const char *, uint32_t) const)
  TensorDesc GetDynamicInputDesc(const std::string &name, uint32_t index) const;

  TensorDesc GetDynamicInputDesc(const char *name, uint32_t index) const;

  ATTRIBUTED_DEPRECATED(graphStatus UpdateDynamicInputDesc(const char *, uint32_t, const TensorDesc &))
  graphStatus UpdateDynamicInputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc);

  graphStatus UpdateDynamicInputDesc(const char *name, uint32_t index, const TensorDesc &tensor_desc);

  ATTRIBUTED_DEPRECATED(TensorDesc GetDynamicOutputDesc(const char *, uint32_t) const)
  TensorDesc GetDynamicOutputDesc(const std::string &name, uint32_t index) const;

  TensorDesc GetDynamicOutputDesc(const char *name, uint32_t index) const;

  ATTRIBUTED_DEPRECATED(graphStatus UpdateDynamicOutputDesc(const char *, uint32_t, const TensorDesc &))
  graphStatus UpdateDynamicOutputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc);

  graphStatus UpdateDynamicOutputDesc(const char *name, uint32_t index, const TensorDesc &tensor_desc);

  graphStatus InferShapeAndType();  // lint !e148

  void SetInferenceContext(const InferenceContextPtr &inference_context);
  InferenceContextPtr GetInferenceContext() const;

  graphStatus VerifyAllAttr(bool disable_common_verifier = false);  // lint !e148

  size_t GetInputsSize() const;

  size_t GetOutputsSize() const;

  ATTRIBUTED_DEPRECATED(graphStatus GetAllAttrNamesAndTypes(std::map<AscendString, AscendString> &) const)
  const std::map<std::string, std::string> GetAllAttrNamesAndTypes() const;

  graphStatus GetAllAttrNamesAndTypes(std::map<AscendString, AscendString> &attr_name_types) const;

  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, int64_t))
  Operator &SetAttr(const std::string &name, int64_t attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, int32_t))
  Operator &SetAttr(const std::string &name, int32_t attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, uint32_t))
  Operator &SetAttr(const std::string &name, uint32_t attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, int64_t &) const)
  graphStatus GetAttr(const std::string &name, int64_t &attr_value) const;
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, int32_t &) const)
  graphStatus GetAttr(const std::string &name, int32_t &attr_value) const;
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, uint32_t &) const)
  graphStatus GetAttr(const std::string &name, uint32_t &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<int64_t> &))
  Operator &SetAttr(const std::string &name, const std::vector<int64_t> &attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<int32_t> &))
  Operator &SetAttr(const std::string &name, const std::vector<int32_t> &attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<uint32_t> &))
  Operator &SetAttr(const std::string &name, const std::vector<uint32_t> &attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, std::initializer_list<int64_t> &&))
  Operator &SetAttr(const std::string &name, std::initializer_list<int64_t> &&attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *name, std::vector<int64_t> &) const)
  graphStatus GetAttr(const std::string &name, std::vector<int64_t> &attr_value) const;
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *name, std::vector<int32_t> &) const)
  graphStatus GetAttr(const std::string &name, std::vector<int32_t> &attr_value) const;
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const std::string &, std::vector<uint32_t> &) const)
  graphStatus GetAttr(const std::string &name, std::vector<uint32_t> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, float attr_value))
  Operator &SetAttr(const std::string &name, float attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, float &) const)
  graphStatus GetAttr(const std::string &name, float &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<float> &))
  Operator &SetAttr(const std::string &name, const std::vector<float> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<float> &) const)
  graphStatus GetAttr(const std::string &name, std::vector<float> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, AttrValue &&))
  Operator &SetAttr(const std::string &name, AttrValue &&attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, AttrValue &) const)
  graphStatus GetAttr(const std::string &name, AttrValue &attr_value) const;
  Operator &SetAttr(const std::string &name, const std::string &attr_value);
  graphStatus GetAttr(const std::string &name, std::string &attr_value) const;
  Operator &SetAttr(const std::string &name, const std::vector<std::string> &attr_value);
  graphStatus GetAttr(const std::string &name, std::vector<std::string> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, bool))
  Operator &SetAttr(const std::string &name, bool attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, bool &) const)
  graphStatus GetAttr(const std::string &name, bool &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<bool> &))
  Operator &SetAttr(const std::string &name, const std::vector<bool> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<bool> &) const)
  graphStatus GetAttr(const std::string &name, std::vector<bool> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const Tensor &))
  Operator &SetAttr(const std::string &name, const Tensor &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, Tensor &) const)
  graphStatus GetAttr(const std::string &name, Tensor &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<Tensor> &))
  Operator &SetAttr(const std::string &name, const std::vector<Tensor> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<Tensor> &) const)
  graphStatus GetAttr(const std::string &name, std::vector<Tensor> &attr_value) const;

  // Bytes type
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const OpBytes &))
  Operator &SetAttr(const std::string &name, const OpBytes &attr_value);
  // Bytes type
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, OpBytes &) const)
  graphStatus GetAttr(const std::string &name, OpBytes &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<std::vector<int64_t>> &))
  Operator &SetAttr(const std::string &name, const std::vector<std::vector<int64_t>> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<std::vector<int64_t>> &) const)
  graphStatus GetAttr(const std::string &name, std::vector<std::vector<int64_t>> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<ge::DataType> &))
  Operator &SetAttr(const std::string &name, const std::vector<ge::DataType> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<ge::DataType> &) const)
  graphStatus GetAttr(const std::string &name, std::vector<ge::DataType> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const ge::DataType &))
  Operator &SetAttr(const std::string &name, const ge::DataType &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, ge::DataType &) const)
  graphStatus GetAttr(const std::string &name, ge::DataType &attr_value) const;

  // func type
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const ge::NamedAttrs &))
  Operator &SetAttr(const std::string &name, const ge::NamedAttrs &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, ge::NamedAttrs &) const)
  graphStatus GetAttr(const std::string &name, ge::NamedAttrs &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<ge::NamedAttrs> &))
  Operator &SetAttr(const std::string &name, const std::vector<ge::NamedAttrs> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<ge::NamedAttrs> &) const)
  graphStatus GetAttr(const std::string &name, std::vector<ge::NamedAttrs> &attr_value) const;

  Operator &SetAttr(const char *name, int64_t attr_value);
  Operator &SetAttr(const char *name, int32_t attr_value);
  Operator &SetAttr(const char *name, uint32_t attr_value);
  graphStatus GetAttr(const char *name, int64_t &attr_value) const;
  graphStatus GetAttr(const char *name, int32_t &attr_value) const;
  graphStatus GetAttr(const char *name, uint32_t &attr_value) const;
  Operator &SetAttr(const char *name, const std::vector<int64_t> &attr_value);
  Operator &SetAttr(const char *name, const std::vector<int32_t> &attr_value);
  Operator &SetAttr(const char *name, const std::vector<uint32_t> &attr_value);
  Operator &SetAttr(const char *name, std::initializer_list<int64_t> &&attr_value);
  graphStatus GetAttr(const char *name, std::vector<int64_t> &attr_value) const;
  graphStatus GetAttr(const char *name, std::vector<int32_t> &attr_value) const;
  graphStatus GetAttr(const char *name, std::vector<uint32_t> &attr_value) const;

  Operator &SetAttr(const char *name, float attr_value);
  graphStatus GetAttr(const char *name, float &attr_value) const;
  Operator &SetAttr(const char *name, const std::vector<float> &attr_value);
  graphStatus GetAttr(const char *name, std::vector<float> &attr_value) const;
  Operator &SetAttr(const char *name, AttrValue &&attr_value);
  graphStatus GetAttr(const char *name, AttrValue &attr_value) const;

  Operator &SetAttr(const char *name, const char *attr_value);
  Operator &SetAttr(const char *name, const AscendString &attr_value);
  graphStatus GetAttr(const char *name, AscendString &attr_value) const;
  Operator &SetAttr(const char *name, const std::vector<AscendString> &attr_values);
  graphStatus GetAttr(const char *name, std::vector<AscendString> &attr_values) const;

  Operator &SetAttr(const char *name, bool attr_value);
  graphStatus GetAttr(const char *name, bool &attr_value) const;
  Operator &SetAttr(const char *name, const std::vector<bool> &attr_value);
  graphStatus GetAttr(const char *name, std::vector<bool> &attr_value) const;

  Operator &SetAttr(const char *name, const Tensor &attr_value);
  graphStatus GetAttr(const char *name, Tensor &attr_value) const;
  Operator &SetAttr(const char *name, const std::vector<Tensor> &attr_value);
  graphStatus GetAttr(const char *name, std::vector<Tensor> &attr_value) const;

  // Bytes type
  Operator &SetAttr(const char *name, const OpBytes &attr_value);
  // Bytes type
  graphStatus GetAttr(const char *name, OpBytes &attr_value) const;

  Operator &SetAttr(const char *name, const std::vector<std::vector<int64_t>> &attr_value);
  graphStatus GetAttr(const char *name, std::vector<std::vector<int64_t>> &attr_value) const;

  Operator &SetAttr(const char *name, const std::vector<ge::DataType> &attr_value);
  graphStatus GetAttr(const char *name, std::vector<ge::DataType> &attr_value) const;

  Operator &SetAttr(const char *name, const ge::DataType &attr_value);
  graphStatus GetAttr(const char *name, ge::DataType &attr_value) const;

  // func type
  Operator &SetAttr(const char *name, const ge::NamedAttrs &attr_value);
  graphStatus GetAttr(const char *name, ge::NamedAttrs &attr_value) const;
  Operator &SetAttr(const char *name, const std::vector<ge::NamedAttrs> &attr_value);
  graphStatus GetAttr(const char *name, std::vector<ge::NamedAttrs> &attr_value) const;

  void BreakConnect() const;

  size_t GetSubgraphNamesCount() const;
  ATTRIBUTED_DEPRECATED(graphStatus GetSubgraphNames(std::vector<AscendString> &) const)
  std::vector<std::string> GetSubgraphNames() const;
  graphStatus GetSubgraphNames(std::vector<AscendString> &names) const;
  ATTRIBUTED_DEPRECATED(SubgraphBuilder GetSubgraphBuilder(const char *) const)
  SubgraphBuilder GetSubgraphBuilder(const std::string &name) const;
  SubgraphBuilder GetSubgraphBuilder(const char *name) const;
  ATTRIBUTED_DEPRECATED(Graph GetSubgraph(const char *) const)
  Graph GetSubgraph(const std::string &name) const;
  Graph GetSubgraph(const char *name) const;
  ATTRIBUTED_DEPRECATED(SubgraphBuilder GetDynamicSubgraphBuilder(const char *, uint32_t) const)
  SubgraphBuilder GetDynamicSubgraphBuilder(const std::string &name, uint32_t index) const;
  SubgraphBuilder GetDynamicSubgraphBuilder(const char *name, uint32_t index) const;
  ATTRIBUTED_DEPRECATED(Graph GetDynamicSubgraph(const char *, uint32_t) const)
  Graph GetDynamicSubgraph(const std::string &name, uint32_t index) const;
  Graph GetDynamicSubgraph(const char *name, uint32_t index) const;

 protected:
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, float))
  void AttrRegister(const std::string &name, float attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const std::vector<float> &))
  void AttrRegister(const std::string &name, const std::vector<float> &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, int64_t))
  void AttrRegister(const std::string &name, int64_t attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const std::vector<int64_t> &))
  void AttrRegister(const std::string &name, const std::vector<int64_t> &attr_value);
  void AttrRegister(const std::string &name, const std::string &attr_value);
  void AttrRegister(const std::string &name, const std::vector<std::string> &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, bool))
  void AttrRegister(const std::string &name, bool attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const std::vector<bool> &))
  void AttrRegister(const std::string &name, const std::vector<bool> &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const Tensor &))
  void AttrRegister(const std::string &name, const Tensor &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const std::vector<Tensor> &))
  void AttrRegister(const std::string &name, const std::vector<Tensor> &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const OpBytes &))
  void AttrRegister(const std::string &name, const OpBytes &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const std::vector<std::vector<int64_t>> &))
  void AttrRegister(const std::string &name, const std::vector<std::vector<int64_t>> &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const std::vector<ge::DataType> &))
  void AttrRegister(const std::string &name, const std::vector<ge::DataType> &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const ge::DataType &))
  void AttrRegister(const std::string &name, const ge::DataType &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const ge::NamedAttrs &))
  void AttrRegister(const std::string &name, const ge::NamedAttrs &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const std::vector<ge::NamedAttrs> &))
  void AttrRegister(const std::string &name, const std::vector<ge::NamedAttrs> &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const AscendString &))
  void AttrRegister(const std::string &name, const AscendString &attr_value);
  ATTRIBUTED_DEPRECATED(void AttrRegister(const char *, const std::vector<AscendString> &))
  void AttrRegister(const std::string &name, const std::vector<AscendString> &attr_value);

  void AttrRegister(const char *name, float attr_value);
  void AttrRegister(const char *name, const std::vector<float> &attr_value);
  void AttrRegister(const char *name, int64_t attr_value);
  void AttrRegister(const char *name, const std::vector<int64_t> &attr_value);
  void AttrRegister(const char *name, const char *attr_value);
  void AttrRegister(const char *name, bool attr_value);
  void AttrRegister(const char *name, const std::vector<bool> &attr_value);
  void AttrRegister(const char *name, const Tensor &attr_value);
  void AttrRegister(const char *name, const std::vector<Tensor> &attr_value);
  void AttrRegister(const char *name, const OpBytes &attr_value);
  void AttrRegister(const char *name, const std::vector<std::vector<int64_t>> &attr_value);
  void AttrRegister(const char *name, const std::vector<ge::DataType> &attr_value);
  void AttrRegister(const char *name, const ge::DataType &attr_value);
  void AttrRegister(const char *name, const ge::NamedAttrs &attr_value);
  void AttrRegister(const char *name, const std::vector<ge::NamedAttrs> &attr_value);
  void AttrRegister(const char *name, const AscendString &attr_value);
  void AttrRegister(const char *name, const std::vector<AscendString> &attr_value);

  explicit Operator(OperatorImplPtr &&op_impl);

  ATTRIBUTED_DEPRECATED(void InputRegister(const char *))
  void InputRegister(const std::string &name);
  void InputRegister(const char *name);

  ATTRIBUTED_DEPRECATED(void OptionalInputRegister(const char *))
  void OptionalInputRegister(const std::string &name);
  void OptionalInputRegister(const char *name);

  void InferFuncRegister(const std::function<graphStatus(Operator &)> &func);

  void VerifierFuncRegister(const std::function<graphStatus(Operator &)> &func);

  void InferFormatFuncRegister(const std::function<graphStatus(Operator &)> &func);

  ATTRIBUTED_DEPRECATED(void OutputRegister(const char *))
  void OutputRegister(const std::string &name);
  void OutputRegister(const char *name);

  ATTRIBUTED_DEPRECATED(void DynamicInputRegister(const char *, const unsigned int, bool))
  void DynamicInputRegister(const std::string &name, const unsigned int num, bool is_push_back = true);
  void DynamicInputRegister(const char *name, const unsigned int num, bool is_push_back = true);

  ATTRIBUTED_DEPRECATED(void DynamicInputRegisterByIndex(const char *, const unsigned int, size_t))
  void DynamicInputRegisterByIndex(const std::string &name, const unsigned int num, size_t index);
  void DynamicInputRegisterByIndex(const char *name, const unsigned int num, size_t index);

  ATTRIBUTED_DEPRECATED(void DynamicOutputRegister(const char *, const unsigned int, bool))
  void DynamicOutputRegister(const std::string &name, const unsigned int num, bool is_push_back = true);
  void DynamicOutputRegister(const char *name, const unsigned int num, bool is_push_back = true);

  ATTRIBUTED_DEPRECATED(void RequiredAttrRegister(const char *))
  void RequiredAttrRegister(const std::string &name);
  void RequiredAttrRegister(const char *name);

  graphStatus VerifyAll();

  // Only has one output index = 0
  ATTRIBUTED_DEPRECATED(Operator &SetInput(const char *, uint32_t, const Operator &))
  Operator &SetInput(const std::string &dst_name, uint32_t dst_index,
                     const Operator &src_oprt);
  Operator &SetInput(const char *dst_name, uint32_t dst_index,
                     const Operator &src_oprt);

  ATTRIBUTED_DEPRECATED(Operator &SetInput(const char *, uint32_t, const Operator &, const char *))
  Operator &SetInput(const std::string &dst_name, uint32_t dst_index, const Operator &src_oprt,
                     const std::string &name);
  Operator &SetInput(const char *dst_name, uint32_t dst_index, const Operator &src_oprt,
                     const char *name);

  ATTRIBUTED_DEPRECATED(void SubgraphRegister(const char *, bool))
  void SubgraphRegister(const std::string &ir_name, bool dynamic);
  void SubgraphRegister(const char *ir_name, bool dynamic);
  ATTRIBUTED_DEPRECATED(void SubgraphCountRegister(const char *, uint32_t))
  void SubgraphCountRegister(const std::string &ir_name, uint32_t count);
  void SubgraphCountRegister(const char *ir_name, uint32_t count);
  ATTRIBUTED_DEPRECATED(void SetSubgraphBuilder(const char *, uint32_t, const SubgraphBuilder &))
  void SetSubgraphBuilder(const std::string &ir_name, uint32_t index, const SubgraphBuilder &builder);
  void SetSubgraphBuilder(const char *ir_name, uint32_t index, const SubgraphBuilder &builder);
  ATTRIBUTED_DEPRECATED(Graph GetSubgraphImpl(const char *) const)
  Graph GetSubgraphImpl(const std::string &name) const;
  Graph GetSubgraphImpl(const char *name) const;

 private:
  ATTRIBUTED_DEPRECATED(Operator &SetInput(const char *, const OutHandler &))
  Operator &SetInput(const std::string &dst_name, const OutHandler &out_handler);
  Operator &SetInput(const char *dst_name, const OutHandler &out_handler);

  ATTRIBUTED_DEPRECATED(OutHandler GetOutput(const char *) const)
  OutHandler GetOutput(const std::string &name) const;
  OutHandler GetOutput(const char *name) const;

  OutHandler GetOutput(uint32_t index) const;

  OperatorImplPtr GetOperatorImplPtr() const;

  OperatorImplPtr operator_impl_{nullptr};

  ATTRIBUTED_DEPRECATED(graphStatus GetInputConstDataOut(const char *, Tensor &) const)
  graphStatus GetInputConstDataOut(const std::string &dst_name, Tensor &data) const;
  graphStatus GetInputConstDataOut(const char *dst_name, Tensor &data) const;

  std::shared_ptr<const Node> GetNode() const;
};
/*lint +e148*/
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_OPERATOR_H_
