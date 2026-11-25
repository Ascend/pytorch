#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "graph/operator.h"

namespace fx_converter {

// Stage is a plain string attribute. GE already recognizes these values.
constexpr const char* kStageAttr = "gert.stage";
constexpr const char* kStageInit = "init";
constexpr const char* kStageMain = "main";
constexpr const char* kStageDeinit = "deinit";

// FXRT-owned node construction request. It holds public tensor descs and attributes.
// FxGraphBuilder turns this request into a public ES Operator.
class FxNodeSpec {
 public:
  FxNodeSpec(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)) {}

  const std::string& GetName() const {
    return name_;
  }
  const std::string& GetType() const {
    return type_;
  }

  ge::graphStatus AddInputDesc(const ge::TensorDesc& desc);
  ge::graphStatus AddInputDesc(const std::string& name, const ge::TensorDesc& desc);
  ge::graphStatus AddOutputDesc(const ge::TensorDesc& desc);
  ge::graphStatus AddOutputDesc(const std::string& name, const ge::TensorDesc& desc);

  const ge::TensorDesc& GetInputDesc(size_t index) const;
  const ge::TensorDesc& GetOutputDesc(size_t index) const;
  size_t GetInputsSize() const {
    return inputs_.size();
  }
  size_t GetOutputsSize() const {
    return outputs_.size();
  }

  void AppendIrInput(const char* name, int32_t kind) {
    if (name != nullptr) {
      ir_inputs_.emplace_back(name);
      ir_input_kinds_.push_back(kind);
    }
  }
  void AppendIrOutput(const char* name, int32_t kind) {
    if (name != nullptr) {
      ir_outputs_.emplace_back(name);
      ir_output_kinds_.push_back(kind);
    }
  }
  void AppendIrAttrName(const char* name) {
    if (name != nullptr) {
      ir_attrs_.emplace_back(name);
    }
  }
  void SetOpInferDepends(const std::vector<std::string>& depends) {
    infer_depends_ = depends;
  }
  void AddSubgraphName(const std::string& name) {
    subgraph_names_.push_back(name);
  }

  template <typename T>
  ge::graphStatus SetAttr(const std::string& name, const T& value) {
    ge::AttrValue attr;
    const auto ret = attr.SetAttrValue(value);
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    attrs_[name] = attr;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus SetAttr(const std::string& name, const ge::AttrValue& value) {
    attrs_[name] = value;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus SetBytesAttr(const std::string& name, const void* data, size_t size);

  struct NamedDesc {
    std::string name;
    ge::TensorDesc desc;
  };
  struct AttrEntry {
    std::string name;
    ge::AttrValue value;
  };

  // These accessors are consumed only by the ES adapter implementation.
  const std::vector<NamedDesc>& Inputs() const {
    return inputs_;
  }
  const std::vector<NamedDesc>& Outputs() const {
    return outputs_;
  }
  const std::map<std::string, ge::AttrValue>& Attrs() const {
    return attrs_;
  }
  const std::map<std::string, ge::Operator::OpBytes>& BytesAttrs() const {
    return bytes_attrs_;
  }
  const std::vector<std::string>& IrInputs() const {
    return ir_inputs_;
  }
  const std::vector<std::string>& IrOutputs() const {
    return ir_outputs_;
  }
  const std::vector<std::string>& SubgraphNames() const {
    return subgraph_names_;
  }

 private:
  std::string name_;
  std::string type_;
  std::vector<NamedDesc> inputs_;
  std::vector<NamedDesc> outputs_;
  std::map<std::string, ge::AttrValue> attrs_;
  std::map<std::string, ge::Operator::OpBytes> bytes_attrs_;
  std::vector<std::string> ir_inputs_;
  std::vector<int32_t> ir_input_kinds_;
  std::vector<std::string> ir_outputs_;
  std::vector<int32_t> ir_output_kinds_;
  std::vector<std::string> ir_attrs_;
  std::vector<std::string> infer_depends_;
  std::vector<std::string> subgraph_names_;
};

using FxNodeSpecPtr = std::shared_ptr<FxNodeSpec>;

// Attribute helpers keep lowering code independent from GE's OpDesc/AttrUtils.
struct FxAttrUtils {
  static bool SetInt(const FxNodeSpecPtr& spec, const char* name, int64_t value);
  static bool SetBool(const FxNodeSpecPtr& spec, const char* name, bool value);
  static bool SetFloat(const FxNodeSpecPtr& spec, const char* name, float value);
  static bool SetStr(const FxNodeSpecPtr& spec, const char* name, const char* value);
  static bool SetListInt(const FxNodeSpecPtr& spec, const char* name, const std::vector<int64_t>& value);
  static bool SetZeroCopyBytes(const FxNodeSpecPtr& spec, const char* name, const void* data, size_t size);
};

} // namespace fx_converter
