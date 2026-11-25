#include "exe_graph/fx_node_spec.h"

#include <cstring>

namespace fx_converter {

namespace {

ge::TensorDesc EmptyDesc() {
  return ge::TensorDesc();
}

}  // namespace

ge::graphStatus FxNodeSpec::AddInputDesc(const ge::TensorDesc &desc) {
  inputs_.push_back({"input" + std::to_string(inputs_.size()), desc});
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FxNodeSpec::AddInputDesc(const std::string &name, const ge::TensorDesc &desc) {
  inputs_.push_back({name, desc});
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FxNodeSpec::AddOutputDesc(const ge::TensorDesc &desc) {
  outputs_.push_back({"output" + std::to_string(outputs_.size()), desc});
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FxNodeSpec::AddOutputDesc(const std::string &name, const ge::TensorDesc &desc) {
  outputs_.push_back({name, desc});
  return ge::GRAPH_SUCCESS;
}

const ge::TensorDesc &FxNodeSpec::GetInputDesc(size_t index) const {
  static const ge::TensorDesc empty = EmptyDesc();
  return index < inputs_.size() ? inputs_[index].desc : empty;
}

const ge::TensorDesc &FxNodeSpec::GetOutputDesc(size_t index) const {
  static const ge::TensorDesc empty = EmptyDesc();
  return index < outputs_.size() ? outputs_[index].desc : empty;
}

ge::graphStatus FxNodeSpec::SetBytesAttr(const std::string &name, const void *data, size_t size) {
  if (data == nullptr && size != 0U) {
    return ge::GRAPH_PARAM_INVALID;
  }
  ge::Operator::OpBytes bytes;
  const auto *begin = static_cast<const uint8_t *>(data);
  if (begin != nullptr) {
    bytes.assign(begin, begin + size);
  }
  bytes_attrs_[name] = std::move(bytes);
  return ge::GRAPH_SUCCESS;
}

bool FxAttrUtils::SetInt(const FxNodeSpecPtr &spec, const char *name, int64_t value) {
  return spec != nullptr && name != nullptr && spec->SetAttr(name, value) == ge::GRAPH_SUCCESS;
}

bool FxAttrUtils::SetBool(const FxNodeSpecPtr &spec, const char *name, bool value) {
  return spec != nullptr && name != nullptr && spec->SetAttr(name, value) == ge::GRAPH_SUCCESS;
}

bool FxAttrUtils::SetFloat(const FxNodeSpecPtr &spec, const char *name, float value) {
  return spec != nullptr && name != nullptr && spec->SetAttr(name, value) == ge::GRAPH_SUCCESS;
}

bool FxAttrUtils::SetStr(const FxNodeSpecPtr &spec, const char *name, const char *value) {
  return spec != nullptr && name != nullptr && value != nullptr &&
         spec->SetAttr(name, ge::AscendString(value)) == ge::GRAPH_SUCCESS;
}

bool FxAttrUtils::SetListInt(const FxNodeSpecPtr &spec, const char *name, const std::vector<int64_t> &value) {
  return spec != nullptr && name != nullptr && spec->SetAttr(name, value) == ge::GRAPH_SUCCESS;
}

bool FxAttrUtils::SetZeroCopyBytes(const FxNodeSpecPtr &spec, const char *name, const void *data, size_t size) {
  return spec != nullptr && name != nullptr && spec->SetBytesAttr(name, data, size) == ge::GRAPH_SUCCESS;
}

}  // namespace fx_converter
