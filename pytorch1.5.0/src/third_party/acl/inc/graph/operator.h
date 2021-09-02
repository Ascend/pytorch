/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
  using OpListString = std::vector<string>;
  using OpListAcendString = std::vector<AscendString>;
  using OpListBool = std::vector<bool>;
  using OpListTensor = std::vector<Tensor>;
  using OpBytes = std::vector<uint8_t>;
  using OpListListInt = std::vector<std::vector<int64_t>>;
  using OpListType = std::vector<ge::DataType>;
  using OpListNamedAttrs = std::vector<ge::NamedAttrs>;

  Operator() {}
  ATTRIBUTED_DEPRECATED(Operator(const char *))
  explicit Operator(const string &type);

  explicit Operator(const char *type);

  ATTRIBUTED_DEPRECATED(Operator(const char *, const char *))
  Operator(const string &name, const string &type);  // lint !e148

  Operator(const AscendString &name, const AscendString &type);

  Operator(const char *name, const char *type);

  virtual ~Operator() = default;

  bool IsEmpty() const;

  ATTRIBUTED_DEPRECATED(graphStatus GetName(AscendString &) const)
  string GetName() const;

  graphStatus GetName(AscendString &name) const;

  ATTRIBUTED_DEPRECATED(graphStatus GetOpType(AscendString &) const)
  string GetOpType() const;

  graphStatus GetOpType(AscendString &type) const;

  // Only has one output index = 0
  ATTRIBUTED_DEPRECATED(Operator &SetInput(const char *, const Operator &))
  Operator &SetInput(const string &dst_name, const Operator &src_oprt);

  Operator &SetInput(const char *dst_name, const Operator &src_oprt);

  ATTRIBUTED_DEPRECATED(Operator &SetInput(const char *, const Operator &, const char *))
  Operator &SetInput(const string &dst_name, const Operator &src_oprt, const string &name);  // lint !e148

  Operator &SetInput(const char *dst_name, const Operator &src_oprt, const char *name);

  ATTRIBUTED_DEPRECATED(Operator &SetInput(const char *, const Operator &, uint32_t))
  Operator &SetInput(const string &dst_name, const Operator &src_oprt, uint32_t index);

  Operator &SetInput(const char *dst_name, const Operator &src_oprt, uint32_t index);

  Operator &AddControlInput(const Operator &src_oprt);

  ATTRIBUTED_DEPRECATED(graphStatus GetInputConstData(const char *, Tensor &) const)
  graphStatus GetInputConstData(const string &dst_name, Tensor &data) const;

  graphStatus GetInputConstData(const char *dst_name, Tensor &data) const;

  ATTRIBUTED_DEPRECATED(TensorDesc GetInputDescByName(const char *) const)
  TensorDesc GetInputDesc(const string &name) const;

  TensorDesc GetInputDescByName(const char *name) const;

  TensorDesc GetInputDesc(uint32_t index) const;

  ATTRIBUTED_DEPRECATED(int GetDynamicOutputNum(const char *) const)
  int GetDynamicOutputNum(const string &name) const;

  int GetDynamicOutputNum(const char *name) const;

  ATTRIBUTED_DEPRECATED(int GetDynamicInputNum(const char *))
  int GetDynamicInputNum(const string &name) const;

  int GetDynamicInputNum(const char *name) const;

  ATTRIBUTED_DEPRECATED(graphStatus TryGetInputDesc(const char *, TensorDesc &) const)
  graphStatus TryGetInputDesc(const string &name, TensorDesc &tensor_desc) const;

  graphStatus TryGetInputDesc(const char *name, TensorDesc &tensor_desc) const;

  ATTRIBUTED_DEPRECATED(graphStatus UpdateInputDesc(const char *, const TensorDesc &))
  graphStatus UpdateInputDesc(const string &name, const TensorDesc &tensor_desc);

  graphStatus UpdateInputDesc(const char *name, const TensorDesc &tensor_desc);

  ATTRIBUTED_DEPRECATED(TensorDesc GetOutputDescByName(const char *) const)
  TensorDesc GetOutputDesc(const string &name) const;

  TensorDesc GetOutputDescByName(const char *name) const;

  TensorDesc GetOutputDesc(uint32_t index) const;

  ATTRIBUTED_DEPRECATED(graphStatus UpdateOutputDesc(const char *, const TensorDesc &tensor_desc))
  graphStatus UpdateOutputDesc(const string &name, const TensorDesc &tensor_desc);  // lint !e148

  graphStatus UpdateOutputDesc(const char *name, const TensorDesc &tensor_desc);

  ATTRIBUTED_DEPRECATED(TensorDesc GetDynamicInputDesc(const char *, uint32_t) const)
  TensorDesc GetDynamicInputDesc(const string &name, uint32_t index) const;

  TensorDesc GetDynamicInputDesc(const char *name, uint32_t index) const;

  ATTRIBUTED_DEPRECATED(graphStatus UpdateDynamicInputDesc(const char *, uint32_t, const TensorDesc &))
  graphStatus UpdateDynamicInputDesc(const string &name, uint32_t index, const TensorDesc &tensor_desc);  // lint !e148

  graphStatus UpdateDynamicInputDesc(const char *name, uint32_t index, const TensorDesc &tensor_desc);

  ATTRIBUTED_DEPRECATED(TensorDesc GetDynamicOutputDesc(const char *, uint32_t) const)
  TensorDesc GetDynamicOutputDesc(const string &name, uint32_t index) const;

  TensorDesc GetDynamicOutputDesc(const char *name, uint32_t index) const;

  ATTRIBUTED_DEPRECATED(graphStatus UpdateDynamicOutputDesc(const char *, uint32_t, const TensorDesc &))
  graphStatus UpdateDynamicOutputDesc(const string &name, uint32_t index, const TensorDesc &tensor_desc);  // lint !e148

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
  Operator &SetAttr(const string &name, int64_t attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, int32_t))
  Operator &SetAttr(const string &name, int32_t attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, uint32_t))
  Operator &SetAttr(const string &name, uint32_t attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, int64_t &) const)
  graphStatus GetAttr(const string &name, int64_t &attr_value) const;
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, int32_t &) const)
  graphStatus GetAttr(const string &name, int32_t &attr_value) const;
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, uint32_t &) const)
  graphStatus GetAttr(const string &name, uint32_t &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<int64_t> &))
  Operator &SetAttr(const string &name, const std::vector<int64_t> &attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<int32_t> &))
  Operator &SetAttr(const string &name, const std::vector<int32_t> &attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<uint32_t> &))
  Operator &SetAttr(const string &name, const std::vector<uint32_t> &attr_value);
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, std::initializer_list<int64_t> &&))
  Operator &SetAttr(const string &name, std::initializer_list<int64_t> &&attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *name, std::vector<int64_t> &) const)
  graphStatus GetAttr(const string &name, std::vector<int64_t> &attr_value) const;
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *name, std::vector<int32_t> &) const)
  graphStatus GetAttr(const string &name, std::vector<int32_t> &attr_value) const;
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const string &, std::vector<uint32_t> &) const)
  graphStatus GetAttr(const string &name, std::vector<uint32_t> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, float attr_value))
  Operator &SetAttr(const string &name, float attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, float &) const)
  graphStatus GetAttr(const string &name, float &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<float> &))
  Operator &SetAttr(const string &name, const std::vector<float> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<float> &) const)
  graphStatus GetAttr(const string &name, std::vector<float> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, AttrValue &&))
  Operator &SetAttr(const string &name, AttrValue &&attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, AttrValue &) const)
  graphStatus GetAttr(const string &name, AttrValue &attr_value) const;
  Operator &SetAttr(const string &name, const string &attr_value);
  graphStatus GetAttr(const string &name, string &attr_value) const;
  Operator &SetAttr(const string &name, const std::vector<string> &attr_value);
  graphStatus GetAttr(const string &name, std::vector<string> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, bool))
  Operator &SetAttr(const string &name, bool attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, bool &) const)
  graphStatus GetAttr(const string &name, bool &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<bool> &))
  Operator &SetAttr(const string &name, const std::vector<bool> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<bool> &) const)
  graphStatus GetAttr(const string &name, std::vector<bool> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const Tensor &))
  Operator &SetAttr(const string &name, const Tensor &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, Tensor &) const)
  graphStatus GetAttr(const string &name, Tensor &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<Tensor> &))
  Operator &SetAttr(const string &name, const std::vector<Tensor> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<Tensor> &) const)
  graphStatus GetAttr(const string &name, std::vector<Tensor> &attr_value) const;

  // Bytes type
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const OpBytes &))
  Operator &SetAttr(const string &name, const OpBytes &attr_value);
  // Bytes type
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, OpBytes &) const)
  graphStatus GetAttr(const string &name, OpBytes &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<std::vector<int64_t>> &))
  Operator &SetAttr(const string &name, const std::vector<std::vector<int64_t>> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<std::vector<int64_t>> &) const)
  graphStatus GetAttr(const string &name, std::vector<std::vector<int64_t>> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<ge::DataType> &))
  Operator &SetAttr(const string &name, const std::vector<ge::DataType> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<ge::DataType> &) const)
  graphStatus GetAttr(const string &name, std::vector<ge::DataType> &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const ge::DataType &))
  Operator &SetAttr(const string &name, const ge::DataType &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, ge::DataType &) const)
  graphStatus GetAttr(const string &name, ge::DataType &attr_value) const;

  // func type
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const ge::NamedAttrs &))
  Operator &SetAttr(const string &name, const ge::NamedAttrs &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, ge::NamedAttrs &) const)
  graphStatus GetAttr(const string &name, ge::NamedAttrs &attr_value) const;
  ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char *, const std::vector<ge::NamedAttrs> &))
  Operator &SetAttr(const string &name, const std::vector<ge::NamedAttrs> &attr_value);
  ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char *, std::vector<ge::NamedAttrs> &) const)
  graphStatus GetAttr(const string &name, std::vector<ge::NamedAttrs> &attr_value) const;

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
  SubgraphBuilder GetSubgraphBuilder(const string &name) const;
  SubgraphBuilder GetSubgraphBuilder(const char *name) const;
  ATTRIBUTED_DEPRECATED(Graph GetSubgraph(const char *) const)
  Graph GetSubgraph(const string &name) const;
  Graph GetSubgraph(const char *name) const;
  ATTRIBUTED_DEPRECATED(SubgraphBuilder GetDynamicSubgraphBuilder(const char *, uint32_t) const)
  SubgraphBuilder GetDynamicSubgraphBuilder(const string &name, uint32_t index) const;
  SubgraphBuilder GetDynamicSubgraphBuilder(const char *name, uint32_t index) const;
  ATTRIBUTED_DEPRECATED(Graph GetDynamicSubgraph(const char *, uint32_t) const)
  Graph GetDynamicSubgraph(const string &name, uint32_t index) const;
  Graph GetDynamicSubgraph(const char *name, uint32_t index) const;

 protected:
  void AttrRegister(const string &name, float attr_value);
  void AttrRegister(const string &name, const std::vector<float> &attr_value);
  void AttrRegister(const string &name, int64_t attr_value);
  void AttrRegister(const string &name, const std::vector<int64_t> &attr_value);
  void AttrRegister(const string &name, const string &attr_value);
  void AttrRegister(const string &name, const std::vector<string> &attr_value);
  void AttrRegister(const string &name, bool attr_value);
  void AttrRegister(const string &name, const std::vector<bool> &attr_value);
  void AttrRegister(const string &name, const Tensor &attr_value);
  void AttrRegister(const string &name, const std::vector<Tensor> &attr_value);
  void AttrRegister(const string &name, const OpBytes &attr_value);
  void AttrRegister(const string &name, const std::vector<std::vector<int64_t>> &attr_value);
  void AttrRegister(const string &name, const std::vector<ge::DataType> &attr_value);
  void AttrRegister(const string &name, const ge::DataType &attr_value);
  void AttrRegister(const string &name, const ge::NamedAttrs &attr_value);
  void AttrRegister(const string &name, const std::vector<ge::NamedAttrs> &attr_value);
  void AttrRegister(const string &name, const AscendString &attr_value);
  void AttrRegister(const string &name, const std::vector<AscendString> &attr_value);

  explicit Operator(OperatorImplPtr &&op_impl);

  void InputRegister(const string &name);

  void OptionalInputRegister(const string &name);

  void InferFuncRegister(const std::function<graphStatus(Operator &)> &func);

  void VerifierFuncRegister(const std::function<graphStatus(Operator &)> &func);

  void InferFormatFuncRegister(const std::function<graphStatus(Operator &)> &func);

  void OutputRegister(const string &name);

  void DynamicInputRegister(const string &name, const unsigned int num, bool is_push_back = true);

  void DynamicInputRegisterByIndex(const string &name, const unsigned int num, size_t index);

  void DynamicOutputRegister(const string &name, const unsigned int num, bool is_push_back = true);

  void RequiredAttrRegister(const string &name);

  graphStatus VerifyAll();  // lint !e148

  // Only has one output index = 0
  Operator &SetInput(const string &dst_name, uint32_t dst_index,
                     const Operator &src_oprt);

  Operator &SetInput(const string &dst_name, uint32_t dst_index, const Operator &src_oprt,
                     const string &name);  // lint !e148

  void SubgraphRegister(const string &ir_name, bool dynamic);
  void SubgraphCountRegister(const string &ir_name, uint32_t count);
  void SetSubgraphBuilder(const string &ir_name, uint32_t index, const SubgraphBuilder &builder);
  Graph GetSubgraphImpl(const string &name) const;

 private:
  Operator &SetInput(const string &dst_name, const OutHandler &out_handler);  // lint !e148

  OutHandler GetOutput(const string &name) const;

  OutHandler GetOutput(uint32_t index) const;

  OperatorImplPtr GetOperatorImplPtr() const;

  OperatorImplPtr operator_impl_{nullptr};

  graphStatus GetInputConstDataOut(const string &dst_name, Tensor &data) const;

  std::shared_ptr<const Node> GetNode() const;
};
/*lint +e148*/
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_OPERATOR_H_
