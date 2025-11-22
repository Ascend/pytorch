/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

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
#include "./types.h"

#ifndef USER_GE_LOGI
#define USER_GE_LOGI(...)
#endif  // USER_GE_LOGI

#ifndef USER_GE_LOGW
#define USER_GE_LOGW(...)
#endif  // USER_GE_LOGW

#ifndef USER_GE_LOGE
#define USER_GE_LOGE(...)
#endif  // USER_GE_LOGE

#define DYNAMIC_OUTPUT_TD_NUM(name) ("__dynamic_output_" + (name) + "_cnt")
#define DYNAMIC_INPUT_TD_NUM(name) ("__dynamic_input_" + (name) + "_cnt")

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
    friend class MultiThreadGraphBuilder;
    friend class NodeUtils;
    friend class OpDescUtils;
    friend class GraphUtils;
    friend class NodeUtilsEx;
    friend class GraphUtilsEx;

    using OpInt = int64_t;
    using OpFloat = float32_t;
    using OpString = std::string;
    using OpAscendString = AscendString;
    using OpBool = bool;
    using OpTensor = Tensor;
    using OpType = ge::DataType;
    using OpNamedAttrs = ge::NamedAttrs;
    using OpListInt = std::vector<int64_t>;
    using OpListFloat = std::vector<float32_t>;
    using OpListString = std::vector<std::string>;
    using OpListAcendString = std::vector<AscendString>;
    using OpListAscendString = OpListAcendString;
    using OpListBool = std::vector<bool>;
    using OpListTensor = std::vector<Tensor>;
    using OpBytes = std::vector<uint8_t>;
    using OpListListInt = std::vector<std::vector<int64_t>>;
    using OpListType = std::vector<ge::DataType>;
    using OpListNamedAttrs = std::vector<ge::NamedAttrs>;

    Operator() {}
    ATTRIBUTED_DEPRECATED(Operator(const char_t *))
    explicit Operator(const std::string &type);

    explicit Operator(const char_t *type);

    ATTRIBUTED_DEPRECATED(Operator(const char_t *, const char_t *))
    Operator(const std::string &name, const std::string &type);

    Operator(const AscendString &name, const AscendString &type);

    Operator(const char_t *name, const char_t *type);

    virtual ~Operator() = default;

    bool IsEmpty() const;

    ATTRIBUTED_DEPRECATED(graphStatus GetName(AscendString &) const)
    std::string GetName() const;

    graphStatus GetName(AscendString &name) const;

    ATTRIBUTED_DEPRECATED(graphStatus GetOpType(AscendString &) const)
    std::string GetOpType() const;

    graphStatus GetOpType(AscendString &type) const;

    // Only has one output index = 0
    ATTRIBUTED_DEPRECATED(Operator &SetInput(const char_t *, const Operator &))
    Operator &SetInput(const std::string &dst_name, const Operator &src_oprt);

    Operator &SetInput(const char_t *dst_name, const Operator &src_oprt);

    ATTRIBUTED_DEPRECATED(Operator &SetInput(const char_t *, const Operator &, const char_t *))
    Operator &SetInput(const std::string &dst_name, const Operator &src_oprt, const std::string &name);

    Operator &SetInput(const char_t *dst_name, const Operator &src_oprt, const char_t *name);

    ATTRIBUTED_DEPRECATED(Operator &SetInput(const char_t *, const Operator &, uint32_t))
    Operator &SetInput(const std::string &dst_name, const Operator &src_oprt, uint32_t index);

    Operator &SetInput(const char_t *dst_name, const Operator &src_oprt, uint32_t index);

    Operator &SetInput(uint32_t dst_index, const Operator &src_oprt, uint32_t src_index);

    Operator &AddControlInput(const Operator &src_oprt);

    ATTRIBUTED_DEPRECATED(graphStatus GetInputConstData(const char_t *, Tensor &) const)
    graphStatus GetInputConstData(const std::string &dst_name, Tensor &data) const;

    graphStatus GetInputConstData(const char_t *dst_name, Tensor &data) const;

    ATTRIBUTED_DEPRECATED(TensorDesc GetInputDescByName(const char_t *) const)
    TensorDesc GetInputDesc(const std::string &name) const;

    TensorDesc GetInputDescByName(const char_t *name) const;

    TensorDesc GetInputDesc(uint32_t index) const;

    ATTRIBUTED_DEPRECATED(int GetDynamicOutputNum(const char_t *) const)
    int32_t GetDynamicOutputNum(const std::string &name) const;

    int32_t GetDynamicOutputNum(const char_t *name) const;

    ATTRIBUTED_DEPRECATED(int GetDynamicInputNum(const char_t *))
    int32_t GetDynamicInputNum(const std::string &name) const;

    int32_t GetDynamicInputNum(const char_t *name) const;

    ATTRIBUTED_DEPRECATED(graphStatus TryGetInputDesc(const char_t *, TensorDesc &) const)
    graphStatus TryGetInputDesc(const std::string &name, TensorDesc &tensor_desc) const;

    graphStatus TryGetInputDesc(const char_t *name, TensorDesc &tensor_desc) const;

    ATTRIBUTED_DEPRECATED(graphStatus UpdateInputDesc(const char_t *, const TensorDesc &))
    graphStatus UpdateInputDesc(const std::string &name, const TensorDesc &tensor_desc);

    graphStatus UpdateInputDesc(const char_t *name, const TensorDesc &tensor_desc);

    ATTRIBUTED_DEPRECATED(TensorDesc GetOutputDescByName(const char_t *) const)
    TensorDesc GetOutputDesc(const std::string &name) const;

    TensorDesc GetOutputDescByName(const char_t *name) const;

    TensorDesc GetOutputDesc(uint32_t index) const;

    ATTRIBUTED_DEPRECATED(graphStatus UpdateOutputDesc(const char_t *, const TensorDesc &tensor_desc))
    graphStatus UpdateOutputDesc(const std::string &name, const TensorDesc &tensor_desc);

    graphStatus UpdateOutputDesc(const char_t *name, const TensorDesc &tensor_desc);

    ATTRIBUTED_DEPRECATED(TensorDesc GetDynamicInputDesc(const char_t *, uint32_t) const)
    TensorDesc GetDynamicInputDesc(const std::string &name, uint32_t index) const;

    TensorDesc GetDynamicInputDesc(const char_t *name, uint32_t index) const;

    ATTRIBUTED_DEPRECATED(graphStatus UpdateDynamicInputDesc(const char_t *, uint32_t, const TensorDesc &))
    graphStatus UpdateDynamicInputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc);

    graphStatus UpdateDynamicInputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc);

    ATTRIBUTED_DEPRECATED(TensorDesc GetDynamicOutputDesc(const char_t *, uint32_t) const)
    TensorDesc GetDynamicOutputDesc(const std::string &name, uint32_t index) const;

    TensorDesc GetDynamicOutputDesc(const char_t *name, uint32_t index) const;

    ATTRIBUTED_DEPRECATED(graphStatus UpdateDynamicOutputDesc(const char_t *, uint32_t, const TensorDesc &))
    graphStatus UpdateDynamicOutputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc);

    graphStatus UpdateDynamicOutputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc);

    graphStatus InferShapeAndType();

    void SetInferenceContext(const InferenceContextPtr &inference_context);
    InferenceContextPtr GetInferenceContext() const;

    graphStatus VerifyAllAttr(bool disable_common_verifier = false);

    size_t GetInputsSize() const;

    size_t GetOutputsSize() const;

    ATTRIBUTED_DEPRECATED(graphStatus GetAllAttrNamesAndTypes(std::map<AscendString, AscendString> &) const)
    const std::map<std::string, std::string> GetAllAttrNamesAndTypes() const;
    /**
     * 获取调用对象上设置好的`自定义属性`和`ir定义属性`的属性名称和类型
     * @param attr_name_types 出参
     * @return
     */
    graphStatus GetAllAttrNamesAndTypes(std::map<AscendString, AscendString> &attr_name_types) const;
    /**
     * 获取ir定义的属性名称和类型（包含普通attr和required_attr）
     * @param attr_name_types 出参
     * @return
     */
    graphStatus GetAllIrAttrNamesAndTypes(std::map<AscendString, AscendString> &attr_name_types) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, int64_t))
    Operator &SetAttr(const std::string &name, int64_t attr_value);
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, int32_t))
    Operator &SetAttr(const std::string &name, int32_t attr_value);
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, uint32_t))
    Operator &SetAttr(const std::string &name, uint32_t attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, int64_t &) const)
    graphStatus GetAttr(const std::string &name, int64_t &attr_value) const;
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, int32_t &) const)
    graphStatus GetAttr(const std::string &name, int32_t &attr_value) const;
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, uint32_t &) const)
    graphStatus GetAttr(const std::string &name, uint32_t &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const std::vector<int64_t> &))
    Operator &SetAttr(const std::string &name, const std::vector<int64_t> &attr_value);
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const std::vector<int32_t> &))
    Operator &SetAttr(const std::string &name, const std::vector<int32_t> &attr_value);
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const std::vector<uint32_t> &))
    Operator &SetAttr(const std::string &name, const std::vector<uint32_t> &attr_value);
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, std::initializer_list<int64_t> &&))
    Operator &SetAttr(const std::string &name, std::initializer_list<int64_t> &&attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *name, std::vector<int64_t> &) const)
    graphStatus GetAttr(const std::string &name, std::vector<int64_t> &attr_value) const;
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *name, std::vector<int32_t> &) const)
    graphStatus GetAttr(const std::string &name, std::vector<int32_t> &attr_value) const;
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const std::string &, std::vector<uint32_t> &) const)
    graphStatus GetAttr(const std::string &name, std::vector<uint32_t> &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, float32_t attr_value))
    Operator &SetAttr(const std::string &name, float32_t attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, float32_t &) const)
    graphStatus GetAttr(const std::string &name, float32_t &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const std::vector<float32_t> &))
    Operator &SetAttr(const std::string &name, const std::vector<float32_t> &attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, std::vector<float32_t> &) const)
    graphStatus GetAttr(const std::string &name, std::vector<float32_t> &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, AttrValue &&))
    Operator &SetAttr(const std::string &name, AttrValue &&attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, AttrValue &) const)
    graphStatus GetAttr(const std::string &name, AttrValue &attr_value) const;
    Operator &SetAttr(const std::string &name, const std::string &attr_value);
    graphStatus GetAttr(const std::string &name, std::string &attr_value) const;
    Operator &SetAttr(const std::string &name, const std::vector<std::string> &attr_value);
    graphStatus GetAttr(const std::string &name, std::vector<std::string> &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, bool))
    Operator &SetAttr(const std::string &name, bool attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, bool &) const)
    graphStatus GetAttr(const std::string &name, bool &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const std::vector<bool> &))
    Operator &SetAttr(const std::string &name, const std::vector<bool> &attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, std::vector<bool> &) const)
    graphStatus GetAttr(const std::string &name, std::vector<bool> &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const Tensor &))
    Operator &SetAttr(const std::string &name, const Tensor &attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, Tensor &) const)
    graphStatus GetAttr(const std::string &name, Tensor &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const std::vector<Tensor> &))
    Operator &SetAttr(const std::string &name, const std::vector<Tensor> &attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, std::vector<Tensor> &) const)
    graphStatus GetAttr(const std::string &name, std::vector<Tensor> &attr_value) const;

    // Bytes type
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const OpBytes &))
    Operator &SetAttr(const std::string &name, const OpBytes &attr_value);
    // Bytes type
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, OpBytes &) const)
    graphStatus GetAttr(const std::string &name, OpBytes &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const std::vector<std::vector<int64_t>> &))
    Operator &SetAttr(const std::string &name, const std::vector<std::vector<int64_t>> &attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, std::vector<std::vector<int64_t>> &) const)
    graphStatus GetAttr(const std::string &name, std::vector<std::vector<int64_t>> &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const std::vector<ge::DataType> &))
    Operator &SetAttr(const std::string &name, const std::vector<ge::DataType> &attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, std::vector<ge::DataType> &) const)
    graphStatus GetAttr(const std::string &name, std::vector<ge::DataType> &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const ge::DataType &))
    Operator &SetAttr(const std::string &name, const ge::DataType &attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, ge::DataType &) const)
    graphStatus GetAttr(const std::string &name, ge::DataType &attr_value) const;

    // func type
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const ge::NamedAttrs &))
    Operator &SetAttr(const std::string &name, const ge::NamedAttrs &attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, ge::NamedAttrs &) const)
    graphStatus GetAttr(const std::string &name, ge::NamedAttrs &attr_value) const;
    ATTRIBUTED_DEPRECATED(Operator &SetAttr(const char_t *, const std::vector<ge::NamedAttrs> &))
    Operator &SetAttr(const std::string &name, const std::vector<ge::NamedAttrs> &attr_value);
    ATTRIBUTED_DEPRECATED(graphStatus GetAttr(const char_t *, std::vector<ge::NamedAttrs> &) const)
    graphStatus GetAttr(const std::string &name, std::vector<ge::NamedAttrs> &attr_value) const;

    Operator &SetAttr(const char_t *name, int64_t attr_value);
    Operator &SetAttr(const char_t *name, int32_t attr_value);
    Operator &SetAttr(const char_t *name, uint32_t attr_value);
    graphStatus GetAttr(const char_t *name, int64_t &attr_value) const;
    graphStatus GetAttr(const char_t *name, int32_t &attr_value) const;
    graphStatus GetAttr(const char_t *name, uint32_t &attr_value) const;
    Operator &SetAttr(const char_t *name, const std::vector<int64_t> &attr_value);
    Operator &SetAttr(const char_t *name, const std::vector<int32_t> &attr_value);
    Operator &SetAttr(const char_t *name, const std::vector<uint32_t> &attr_value);
    Operator &SetAttr(const char_t *name, std::initializer_list<int64_t> &&attr_value);
    graphStatus GetAttr(const char_t *name, std::vector<int64_t> &attr_value) const;
    graphStatus GetAttr(const char_t *name, std::vector<int32_t> &attr_value) const;
    graphStatus GetAttr(const char_t *name, std::vector<uint32_t> &attr_value) const;

    Operator &SetAttr(const char_t *name, float32_t attr_value);
    graphStatus GetAttr(const char_t *name, float32_t &attr_value) const;
    Operator &SetAttr(const char_t *name, const std::vector<float32_t> &attr_value);
    graphStatus GetAttr(const char_t *name, std::vector<float32_t> &attr_value) const;
    Operator &SetAttr(const char_t *name, AttrValue &&attr_value);
    Operator &SetAttr(const char_t *name, const AttrValue &attr_value);
    graphStatus GetAttr(const char_t *name, AttrValue &attr_value) const;

    Operator &SetAttr(const char_t *name, const char_t *attr_value);
    Operator &SetAttr(const char_t *name, const AscendString &attr_value);
    graphStatus GetAttr(const char_t *name, AscendString &attr_value) const;
    Operator &SetAttr(const char_t *name, const std::vector<AscendString> &attr_values);
    graphStatus GetAttr(const char_t *name, std::vector<AscendString> &attr_values) const;

    Operator &SetAttr(const char_t *name, bool attr_value);
    graphStatus GetAttr(const char_t *name, bool &attr_value) const;
    Operator &SetAttr(const char_t *name, const std::vector<bool> &attr_value);
    graphStatus GetAttr(const char_t *name, std::vector<bool> &attr_value) const;

    Operator &SetAttr(const char_t *name, const Tensor &attr_value);
    graphStatus GetAttr(const char_t *name, Tensor &attr_value) const;
    Operator &SetAttr(const char_t *name, const std::vector<Tensor> &attr_value);
    graphStatus GetAttr(const char_t *name, std::vector<Tensor> &attr_value) const;

    // Bytes type
    Operator &SetAttr(const char_t *name, const OpBytes &attr_value);
    // Bytes type
    graphStatus GetAttr(const char_t *name, OpBytes &attr_value) const;

    Operator &SetAttr(const char_t *name, const std::vector<std::vector<int64_t>> &attr_value);
    graphStatus GetAttr(const char_t *name, std::vector<std::vector<int64_t>> &attr_value) const;

    Operator &SetAttr(const char_t *name, const std::vector<ge::DataType> &attr_value);
    graphStatus GetAttr(const char_t *name, std::vector<ge::DataType> &attr_value) const;

    Operator &SetAttr(const char_t *name, const ge::DataType &attr_value);
    graphStatus GetAttr(const char_t *name, ge::DataType &attr_value) const;

    // func type
    Operator &SetAttr(const char_t *name, const ge::NamedAttrs &attr_value);
    graphStatus GetAttr(const char_t *name, ge::NamedAttrs &attr_value) const;
    Operator &SetAttr(const char_t *name, const std::vector<ge::NamedAttrs> &attr_value);
    graphStatus GetAttr(const char_t *name, std::vector<ge::NamedAttrs> &attr_value) const;

    void BreakConnect() const;

    size_t GetSubgraphNamesCount() const;
    ATTRIBUTED_DEPRECATED(graphStatus GetSubgraphNames(std::vector<AscendString> &) const)
    std::vector<std::string> GetSubgraphNames() const;
    graphStatus GetSubgraphNames(std::vector<AscendString> &names) const;
    ATTRIBUTED_DEPRECATED(SubgraphBuilder GetSubgraphBuilder(const char_t *) const)
    SubgraphBuilder GetSubgraphBuilder(const std::string &name) const;
    SubgraphBuilder GetSubgraphBuilder(const char_t *name) const;
    ATTRIBUTED_DEPRECATED(Graph GetSubgraph(const char_t *) const)
    Graph GetSubgraph(const std::string &name) const;
    Graph GetSubgraph(const char_t *name) const;
    ATTRIBUTED_DEPRECATED(SubgraphBuilder GetDynamicSubgraphBuilder(const char_t *, uint32_t) const)
    SubgraphBuilder GetDynamicSubgraphBuilder(const std::string &name, uint32_t index) const;
    SubgraphBuilder GetDynamicSubgraphBuilder(const char_t *name, uint32_t index) const;
    ATTRIBUTED_DEPRECATED(Graph GetDynamicSubgraph(const char_t *, uint32_t) const)
    Graph GetDynamicSubgraph(const std::string &name, uint32_t index) const;
    Graph GetDynamicSubgraph(const char_t *name, uint32_t index) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, const char_t *attr_value);
    Operator &SetOutputAttr(const int32_t index, const char_t *name, const char_t *attr_value);
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, const char_t *attr_value);
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const char_t *attr_value);

    Operator &SetInputAttr(const int32_t index, const char_t *name, const AscendString &attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, AscendString &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, const AscendString &attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, AscendString &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, const AscendString &attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, AscendString &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const AscendString &attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, AscendString &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, int64_t attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, int64_t &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, int64_t attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, int64_t &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, int64_t attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, int64_t &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, int64_t attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, int64_t &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, int32_t attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, int32_t &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, int32_t attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, int32_t &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, int32_t attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, int32_t &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, int32_t attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, int32_t &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, uint32_t attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, uint32_t &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, uint32_t attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, uint32_t &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, uint32_t attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, uint32_t &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, uint32_t attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, uint32_t &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, bool attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, bool &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, bool attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, bool &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, bool attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, bool &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, bool attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, bool &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, float32_t attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, float32_t &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, float32_t attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, float32_t &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, float32_t attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, float32_t &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, float32_t attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, float32_t &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, const std::vector<AscendString> &attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, std::vector<AscendString> &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<AscendString> &attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<AscendString> &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, const std::vector<AscendString> &attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, std::vector<AscendString> &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<AscendString> &attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<AscendString> &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, const std::vector<int64_t> &attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, std::vector<int64_t> &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<int64_t> &attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<int64_t> &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, const std::vector<int64_t> &attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, std::vector<int64_t> &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<int64_t> &attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<int64_t> &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, const std::vector<int32_t> &attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, std::vector<int32_t> &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<int32_t> &attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<int32_t> &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, const std::vector<int32_t> &attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, std::vector<int32_t> &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<int32_t> &attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<int32_t> &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, const std::vector<uint32_t> &attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, std::vector<uint32_t> &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<uint32_t> &attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<uint32_t> &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, const std::vector<uint32_t> &attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, std::vector<uint32_t> &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<uint32_t> &attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<uint32_t> &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, const std::vector<bool> &attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, std::vector<bool> &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<bool> &attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<bool> &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, const std::vector<bool> &attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, std::vector<bool> &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<bool> &attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<bool> &attr_value) const;

    Operator &SetInputAttr(const int32_t index, const char_t *name, const std::vector<float32_t> &attr_value);
    graphStatus GetInputAttr(const int32_t index, const char_t *name, std::vector<float32_t> &attr_value) const;
    Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<float32_t> &attr_value);
    graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<float32_t> &attr_value) const;
    Operator &SetInputAttr(const char_t *dst_name, const char_t *name, const std::vector<float32_t> &attr_value);
    graphStatus GetInputAttr(const char_t *dst_name, const char_t *name, std::vector<float32_t> &attr_value) const;
    Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<float32_t> &attr_value);
    graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<float32_t> &attr_value) const;

    Operator &SetInput(const char_t *dst_name, uint32_t dst_index, const Operator &src_oprt, const char_t *name);
    Operator &SetInput(const char_t *dst_name, uint32_t dst_index, const Operator &src_oprt);

    void DynamicInputRegister(const char_t *name, const uint32_t num, bool is_push_back = true);
    void DynamicInputRegister(const char_t *name, const uint32_t num, const char_t *datatype_symbol,
                              bool is_push_back = true);

    void DynamicInputRegisterByIndex(const char_t *name, const uint32_t num, size_t index);

    void DynamicOutputRegister(const char_t *name, const uint32_t num, bool is_push_back = true);
    void DynamicOutputRegister(const char_t *name, const uint32_t num, const char_t *datatype_symbol,
                              bool is_push_back = true);

    void SubgraphCountRegister(const char_t *ir_name, uint32_t count);

    void SetSubgraphBuilder(const char_t *ir_name, uint32_t index, const SubgraphBuilder &builder);

    graphStatus UpdateInputDesc(const uint32_t index, const TensorDesc &tensor_desc);

    graphStatus UpdateOutputDesc(const uint32_t index, const TensorDesc &tensor_desc);

    void AttrRegister(const char_t *name, const AttrValue &attr_value);
    graphStatus SetSubgraphInstanceName(const uint32_t index, const char_t *name);
protected:
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, float32_t))
    void AttrRegister(const std::string &name, float32_t attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const std::vector<float32_t> &))
    void AttrRegister(const std::string &name, const std::vector<float32_t> &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, int64_t))
    void AttrRegister(const std::string &name, int64_t attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const std::vector<int64_t> &))
    void AttrRegister(const std::string &name, const std::vector<int64_t> &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const AscendString &))
    void AttrRegister(const std::string &name, const std::string &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const std::vector<AscendString> &))
    void AttrRegister(const std::string &name, const std::vector<std::string> &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, bool))
    void AttrRegister(const std::string &name, bool attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const std::vector<bool> &))
    void AttrRegister(const std::string &name, const std::vector<bool> &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const Tensor &))
    void AttrRegister(const std::string &name, const Tensor &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const std::vector<Tensor> &))
    void AttrRegister(const std::string &name, const std::vector<Tensor> &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const OpBytes &))
    void AttrRegister(const std::string &name, const OpBytes &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const std::vector<std::vector<int64_t>> &))
    void AttrRegister(const std::string &name, const std::vector<std::vector<int64_t>> &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const std::vector<ge::DataType> &))
    void AttrRegister(const std::string &name, const std::vector<ge::DataType> &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const ge::DataType &))
    void AttrRegister(const std::string &name, const ge::DataType &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const ge::NamedAttrs &))
    void AttrRegister(const std::string &name, const ge::NamedAttrs &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const std::vector<ge::NamedAttrs> &))
    void AttrRegister(const std::string &name, const std::vector<ge::NamedAttrs> &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const AscendString &))
    void AttrRegister(const std::string &name, const AscendString &attr_value);
    ATTRIBUTED_DEPRECATED(void AttrRegister(const char_t *, const std::vector<AscendString> &))
    void AttrRegister(const std::string &name, const std::vector<AscendString> &attr_value);

    void AttrRegister(const char_t *name, float32_t attr_value);
    void AttrRegister(const char_t *name, const std::vector<float32_t> &attr_value);
    void AttrRegister(const char_t *name, int64_t attr_value);
    void AttrRegister(const char_t *name, const std::vector<int64_t> &attr_value);
    void AttrRegister(const char_t *name, const char_t *attr_value);
    void AttrRegister(const char_t *name, bool attr_value);
    void AttrRegister(const char_t *name, const std::vector<bool> &attr_value);
    void AttrRegister(const char_t *name, const Tensor &attr_value);
    void AttrRegister(const char_t *name, const std::vector<Tensor> &attr_value);
    void AttrRegister(const char_t *name, const OpBytes &attr_value);
    void AttrRegister(const char_t *name, const std::vector<std::vector<int64_t>> &attr_value);
    void AttrRegister(const char_t *name, const std::vector<ge::DataType> &attr_value);
    void AttrRegister(const char_t *name, const ge::DataType &attr_value);
    void AttrRegister(const char_t *name, const ge::NamedAttrs &attr_value);
    void AttrRegister(const char_t *name, const std::vector<ge::NamedAttrs> &attr_value);
    void AttrRegister(const char_t *name, const AscendString &attr_value);
    void AttrRegister(const char_t *name, const std::vector<AscendString> &attr_value);
    explicit Operator(OperatorImplPtr &&op_impl);

    ATTRIBUTED_DEPRECATED(void InputRegister(const char_t *))
    void InputRegister(const std::string &name);
    void InputRegister(const char_t *name);
    void InputRegister(const char_t *name, const char_t *datatype_symbol);

    ATTRIBUTED_DEPRECATED(void OptionalInputRegister(const char_t *))
    void OptionalInputRegister(const std::string &name);
    void OptionalInputRegister(const char_t *name);
    void OptionalInputRegister(const char_t *name, const char_t *datatype_symbol);

    void InferFuncRegister(const std::function<graphStatus(Operator &)> &func);

    void VerifierFuncRegister(const std::function<graphStatus(Operator &)> &func);

    void InferFormatFuncRegister(const std::function<graphStatus(Operator &)> &func);

    ATTRIBUTED_DEPRECATED(void OutputRegister(const char_t *))
    void OutputRegister(const std::string &name);
    void OutputRegister(const char_t *name);

    void OutputRegister(const char_t *name, const char_t *datatype_symbol);

    ATTRIBUTED_DEPRECATED(void DynamicInputRegister(const char_t *, const uint32_t, bool))
    void DynamicInputRegister(const std::string &name, const uint32_t num, bool is_push_back = true);

    ATTRIBUTED_DEPRECATED(void DynamicInputRegisterByIndex(const char_t *, const uint32_t, size_t))
    void DynamicInputRegisterByIndex(const std::string &name, const uint32_t num, size_t index);

    ATTRIBUTED_DEPRECATED(void DynamicOutputRegister(const char_t *, const uint32_t, bool))
    void DynamicOutputRegister(const std::string &name, const uint32_t num, bool is_push_back = true);

    ATTRIBUTED_DEPRECATED(void RequiredAttrRegister(const char_t *))
    void RequiredAttrRegister(const std::string &name);
    void RequiredAttrRegister(const char_t *name);
    void RequiredAttrWithTypeRegister(const char_t *name, const char_t *type);

    void DataTypeRegister(const char_t *datatype_symbol, const TensorType &type_range);
    void DataTypeRegister(const char_t *datatype_symbol, const ListTensorType &list_type_range);
    void DataTypeRegister(const char_t *datatype_symbol, const Promote &promote_rule);

    graphStatus VerifyAll();

    // Only has one output index = 0
    ATTRIBUTED_DEPRECATED(Operator &SetInput(const char_t *, uint32_t, const Operator &))
    Operator &SetInput(const std::string &dst_name, uint32_t dst_index,
                      const Operator &src_oprt);

    ATTRIBUTED_DEPRECATED(Operator &SetInput(const char_t *, uint32_t, const Operator &, const char_t *))
    Operator &SetInput(const std::string &dst_name, uint32_t dst_index, const Operator &src_oprt,
                      const std::string &name);

    ATTRIBUTED_DEPRECATED(void SubgraphRegister(const char_t *, bool))
    void SubgraphRegister(const std::string &ir_name, bool dynamic);
    void SubgraphRegister(const char_t *ir_name, bool dynamic);
    ATTRIBUTED_DEPRECATED(void SubgraphCountRegister(const char_t *, uint32_t))
    void SubgraphCountRegister(const std::string &ir_name, uint32_t count);
    ATTRIBUTED_DEPRECATED(void SetSubgraphBuilder(const char_t *, uint32_t, const SubgraphBuilder &))
    void SetSubgraphBuilder(const std::string &ir_name, uint32_t index, const SubgraphBuilder &builder);
    ATTRIBUTED_DEPRECATED(Graph GetSubgraphImpl(const char_t *) const)
    Graph GetSubgraphImpl(const std::string &name) const;
    Graph GetSubgraphImpl(const char_t *name) const;

private:
    ATTRIBUTED_DEPRECATED(Operator &SetInput(const char_t *, const OutHandler &))
    Operator &SetInput(const std::string &dst_name, const OutHandler &out_handler);
    Operator &SetInput(const char_t *dst_name, const OutHandler &out_handler);

    ATTRIBUTED_DEPRECATED(OutHandler GetOutput(const char_t *) const)
    OutHandler GetOutput(const std::string &name) const;
    OutHandler GetOutput(const char_t *name) const;

    OutHandler GetOutput(uint32_t index) const;

    OperatorImplPtr GetOperatorImplPtr() const;

    OperatorImplPtr operator_impl_{nullptr};

    ATTRIBUTED_DEPRECATED(graphStatus GetInputConstDataOut(const char_t *, Tensor &) const)
    graphStatus GetInputConstDataOut(const std::string &dst_name, Tensor &data) const;
    graphStatus GetInputConstDataOut(const char_t *dst_name, Tensor &data) const;

    std::shared_ptr<const Node> GetNode() const;
};
/*lint +e148*/
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_OPERATOR_H_
