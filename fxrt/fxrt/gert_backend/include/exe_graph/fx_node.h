#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "graph/gnode.h"

namespace fx_converter {

class FxGraphBuilder;

constexpr uint64_t kInvalidSymbol = std::numeric_limits<uint64_t>::max();

// Converter-owned node. The public GNode is the graph object; symbol ids stay
// here so lowering can keep its existing data-flow without GE ExtendInfo.
class Node {
 public:
  class Extend {
   public:
    explicit Extend(Node* owner) : owner_(owner) {}

    void UpdateInputSymbols(size_t count) {
      owner_->in_syms_.assign(count, kInvalidSymbol);
    }
    void UpdateOutputSymbols(size_t count) {
      owner_->out_syms_.assign(count, kInvalidSymbol);
    }
    void SetInputSymbol(size_t index, uint64_t symbol) {
      SetAt(&owner_->in_syms_, index, symbol);
    }
    void SetOutputSymbol(size_t index, uint64_t symbol) {
      SetAt(&owner_->out_syms_, index, symbol);
    }
    uint64_t GetInputSymbol(size_t index) const {
      return At(owner_->in_syms_, index);
    }
    uint64_t GetOutputSymbol(size_t index) const {
      return At(owner_->out_syms_, index);
    }

   private:
    static void SetAt(std::vector<uint64_t>* values, size_t index, uint64_t symbol) {
      if (values->size() <= index) {
        values->resize(index + 1U, kInvalidSymbol);
      }
      (*values)[index] = symbol;
    }
    static uint64_t At(const std::vector<uint64_t>& values, size_t index) {
      return index < values.size() ? values[index] : kInvalidSymbol;
    }

    Node* owner_ = nullptr;
  };

  Node(FxGraphBuilder* builder, const ge::GNode& gnode, std::string name, std::string type)
      : builder_(builder), gnode_(gnode), name_(std::move(name)), type_(std::move(type)), extend_(this) {}

  Extend* GetExtendInfo() {
    return &extend_;
  }
  const Extend* GetExtendInfo() const {
    return &extend_;
  }

  const std::string& GetName() const {
    return name_;
  }
  const char* GetNamePtr() const {
    return name_.c_str();
  }
  const std::string& GetType() const {
    return type_;
  }
  const char* GetTypePtr() const {
    return type_.c_str();
  }

  ge::GNode& gnode() {
    return gnode_;
  }
  const ge::GNode& gnode() const {
    return gnode_;
  }
  FxGraphBuilder* builder() const {
    return builder_;
  }

  size_t InputCount() const {
    return gnode_.GetInputsSize();
  }
  size_t OutputCount() const {
    return gnode_.GetOutputsSize();
  }

  bool CopyOutputDesc(uint32_t index, ge::TensorDesc* out) const {
    return out != nullptr && gnode_.GetOutputDesc(static_cast<int32_t>(index), *out) == ge::GRAPH_SUCCESS;
  }
  bool IntAttr(const char* name, int64_t* out) const {
    if (name == nullptr || out == nullptr) {
      return false;
    }
    ge::AscendString key(name);
    return gnode_.GetAttr(key, *out) == ge::GRAPH_SUCCESS;
  }
  bool BoolAttr(const char* name, bool* out) const {
    if (name == nullptr || out == nullptr) {
      return false;
    }
    ge::AscendString key(name);
    return gnode_.GetAttr(key, *out) == ge::GRAPH_SUCCESS;
  }
  bool BytesAttr(const char* name, std::vector<uint8_t>* out) const {
    if (name == nullptr || out == nullptr) {
      return false;
    }
    ge::OpBytes bytes;
    ge::AscendString key(name);
    if (gnode_.GetAttr(key, bytes) != ge::GRAPH_SUCCESS) {
      return false;
    }
    *out = std::move(bytes);
    return true;
  }

  Node* InputProducer(int32_t index) const;

 private:
  FxGraphBuilder* builder_ = nullptr;
  ge::GNode gnode_;
  std::string name_;
  std::string type_;
  std::vector<uint64_t> in_syms_;
  std::vector<uint64_t> out_syms_;
  Extend extend_;
};

} // namespace fx_converter
