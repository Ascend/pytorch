/* Copyright 2026 Huawei Technologies Co., Ltd */

#include "shape/fx_infer_func.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>

#include "common/logger.h"
#include "ir/symbolic/symbolic.h"

namespace fx_converter {
namespace {

std::atomic<uint64_t> g_artifact_id{0U};

std::string ShellQuote(const std::string& value) {
  std::string result("'");
  for (const char c : value) {
    if (c == '\'') {
      result += "'\\''";
    } else {
      result.push_back(c);
    }
  }
  result.push_back('\'');
  return result;
}

std::string InferIncludeDir() {
  const char* root = std::getenv("ASCEND_HOME_PATH");
  if (root == nullptr || root[0] == '\0') {
    return {};
  }
  const std::filesystem::path base(root);
  for (const char* arch : {"aarch64-linux", "arm64-linux", "x86_64-linux"}) {
    const auto candidate = base / arch / "include";
    if (std::filesystem::exists(candidate / "exe_graph/runtime/tensor.h")) {
      return candidate.string();
    }
  }
  const auto candidate = base / "include";
  return std::filesystem::exists(candidate / "exe_graph/runtime/tensor.h") ? candidate.string() : std::string();
}

std::string EmitExpression(
    const fxrt::ir::SymbolicExprPtr& expr,
    const std::unordered_map<std::string, FxSymbolEnv::Seed>& seeds) {
  if (expr == nullptr) {
    return {};
  }
  using Kind = fxrt::ir::SymbolicExpr::Kind;
  if (expr->GetKind() == Kind::Constant) {
    return std::to_string(static_cast<const fxrt::ir::SymbolicConst*>(expr.get())->GetValue());
  }
  if (expr->GetKind() == Kind::Variable) {
    const auto& name = static_cast<const fxrt::ir::SymbolicVar*>(expr.get())->GetName();
    const auto iter = seeds.find(name);
    if (iter == seeds.cend() || iter->second.feed_index < 0 || iter->second.dim_index < 0) {
      return {};
    }
    std::ostringstream stream;
    stream << "inputs[" << iter->second.feed_index << "]->GetOriginShape().GetDim(" << iter->second.dim_index << "U)";
    return stream.str();
  }
  const auto* binary = dynamic_cast<const fxrt::ir::SymbolicBinaryOp*>(expr.get());
  if (binary == nullptr) {
    return {};
  }
  const auto lhs = EmitExpression(binary->getLHS(), seeds);
  const auto rhs = EmitExpression(binary->getRHS(), seeds);
  if (lhs.empty() || rhs.empty()) {
    return {};
  }
  std::ostringstream stream;
  switch (expr->GetKind()) {
    case Kind::Add:
      stream << "(" << lhs << " + " << rhs << ")";
      break;
    case Kind::Mul:
      stream << "(" << lhs << " * " << rhs << ")";
      break;
    case Kind::TrueDiv:
      stream << "(" << lhs << " / " << rhs << ")";
      break;
    case Kind::FloorDiv:
      stream << "fx_floor(" << lhs << ", " << rhs << ")";
      break;
    case Kind::CeilDiv:
      stream << "fx_ceil(" << lhs << ", " << rhs << ")";
      break;
    case Kind::Mod:
      stream << "(" << lhs << " % " << rhs << ")";
      break;
    case Kind::Min:
      stream << "std::min<int64_t>(" << lhs << ", " << rhs << ")";
      break;
    default:
      return {};
  }
  return stream.str();
}

bool BuildSource(const FxFusedShapeRules& rules, const std::string& symbol, std::string* source) {
  if (source == nullptr || (rules.slot_rules.empty() && rules.feed_shape_indices.empty())) {
    return false;
  }
  std::unordered_map<std::string, FxSymbolEnv::Seed> seeds;
  size_t required_input_num = 0U;
  for (const auto* rule : rules.slot_rules) {
    if (rule == nullptr || rule->env == nullptr) {
      return false;
    }
    for (const auto& seed : rule->env->seeds) {
      if (seed.feed_index < 0 || seed.dim_index < 0) {
        return false;
      }
      required_input_num = std::max(required_input_num, static_cast<size_t>(seed.feed_index) + 1U);
      const auto inserted = seeds.emplace(seed.name, seed);
      if (!inserted.second &&
          (inserted.first->second.feed_index != seed.feed_index ||
           inserted.first->second.dim_index != seed.dim_index)) {
        return false;
      }
    }
  }
  for (const size_t feed_index : rules.feed_shape_indices) {
    required_input_num = std::max(required_input_num, feed_index + 1U);
  }

  std::ostringstream stream;
  stream << "#include <algorithm>\n"
         << "#include <cstdint>\n"
         << "#include <limits>\n"
         << "#include <type_traits>\n"
         << "#include \"exe_graph/runtime/tensor.h\"\n"
         << "extern \"C\" int32_t " << symbol
         << "(const gert::Tensor *const *inputs, size_t input_num, gert::Tensor *const *outputs, "
            "size_t output_num) {\n"
         << "  if (output_num != " << rules.slot_rules.size() + rules.feed_shape_indices.size() << "U || "
         << "(output_num != 0U && outputs == nullptr) || input_num < " << required_input_num << "U || "
         << "(" << required_input_num << "U != 0U && inputs == nullptr)) return 1;\n"
         << "  auto fx_floor = [](int64_t a, int64_t b) -> int64_t { if (b == 0) return 0; int64_t q = a / b; "
            "int64_t r = a % b; return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q; };\n"
         << "  auto fx_ceil = [](int64_t a, int64_t b) -> int64_t { if (b == 0) return 0; int64_t q = a / b; "
            "int64_t r = a % b; return (r != 0 && ((r < 0) == (b < 0))) ? q + 1 : q; };\n";

  for (size_t slot = 0U; slot < rules.slot_rules.size(); ++slot) {
    const auto* rule = rules.slot_rules[slot];
    if (rule->out_dims_ir.empty() && !rule->is_rank0) {
      return false;
    }
    stream << "  if (outputs[" << slot << "U] == nullptr) return 1;\n"
           << "  auto &origin_" << slot << " = outputs[" << slot << "U]->MutableOriginShape();\n"
           << "  auto &storage_" << slot << " = outputs[" << slot << "U]->MutableStorageShape();\n"
           << "  origin_" << slot << ".SetDimNum(" << rule->out_dims_ir.size() << "U);\n"
           << "  storage_" << slot << ".SetDimNum(" << rule->out_dims_ir.size() << "U);\n";
    for (size_t dim = 0U; dim < rule->out_dims_ir.size(); ++dim) {
      const auto expression = EmitExpression(rule->out_dims_ir[dim], seeds);
      if (expression.empty()) {
        return false;
      }
      stream << "  { const int64_t value = " << expression << "; if (value < 0) return 1; origin_" << slot << ".SetDim("
             << dim << "U, value); storage_" << slot << ".SetDim(" << dim << "U, value); }\n";
    }
  }
  for (size_t i = 0U; i < rules.feed_shape_indices.size(); ++i) {
    const size_t slot = rules.slot_rules.size() + i;
    const size_t feed = rules.feed_shape_indices[i];
    stream << "  if (inputs[" << feed << "U] == nullptr || outputs[" << slot << "U] == nullptr) return 1;\n"
           << "  outputs[" << slot << "U]->MutableOriginShape() = inputs[" << feed << "U]->GetOriginShape();\n"
           << "  outputs[" << slot << "U]->MutableStorageShape() = inputs[" << feed << "U]->GetOriginShape();\n";
  }
  stream << "  return 0;\n}\n";
  *source = stream.str();
  return true;
}

void AppendU32(std::vector<uint8_t>* payload, const uint32_t value) {
  for (size_t i = 0U; i < sizeof(value); ++i) {
    payload->push_back(static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU));
  }
}

void AppendU64(std::vector<uint8_t>* payload, const uint64_t value) {
  for (size_t i = 0U; i < sizeof(value); ++i) {
    payload->push_back(static_cast<uint8_t>((value >> (i * 8U)) & 0xFFU));
  }
}

bool ReadU32(const uint8_t* data, const size_t size, size_t* offset, uint32_t* value) {
  if (data == nullptr || offset == nullptr || value == nullptr || *offset > size || size - *offset < sizeof(uint32_t)) {
    return false;
  }
  *value = 0U;
  for (size_t i = 0U; i < sizeof(uint32_t); ++i) {
    *value |= static_cast<uint32_t>(data[*offset + i]) << (i * 8U);
  }
  *offset += sizeof(uint32_t);
  return true;
}

bool ReadU64(const uint8_t* data, const size_t size, size_t* offset, uint64_t* value) {
  if (data == nullptr || offset == nullptr || value == nullptr || *offset > size || size - *offset < sizeof(uint64_t)) {
    return false;
  }
  *value = 0U;
  for (size_t i = 0U; i < sizeof(uint64_t); ++i) {
    *value |= static_cast<uint64_t>(data[*offset + i]) << (i * 8U);
  }
  *offset += sizeof(uint64_t);
  return true;
}

} // namespace

bool BuildInferShapeSharedObject(const FxFusedShapeRules& rules, FxInferFuncArtifact* artifact) {
  if (artifact == nullptr) {
    return false;
  }
  const auto include_dir = InferIncludeDir();
  if (include_dir.empty()) {
    RT_GLOG(ERROR) << "InferShape codegen: CANN tensor header is unavailable";
    return false;
  }
  std::filesystem::path output_dir = std::filesystem::temp_directory_path();
  std::error_code error;
  std::filesystem::create_directories(output_dir, error);
  if (error) {
    RT_GLOG(ERROR) << "InferShape codegen: cannot create " << output_dir << ": " << error.message();
    return false;
  }
  const auto id = g_artifact_id.fetch_add(1U);
  const std::string base = "fx_infer_shape_" + std::to_string(static_cast<unsigned long long>(getpid())) + "_" +
      std::to_string(static_cast<unsigned long long>(id));
  artifact->symbol = "FxInferShapeEntry_" + std::to_string(static_cast<unsigned long long>(id));
  const auto source_path = output_dir / (base + ".cc");
  const auto so_path = output_dir / (base + ".so");
  std::string source;
  if (!BuildSource(rules, artifact->symbol, &source)) {
    RT_GLOG(WARNING) << "InferShape codegen: failed to emit shape function source";
    return false;
  }
  std::ofstream output(source_path);
  if (!output.is_open()) {
    RT_GLOG(ERROR) << "InferShape codegen: cannot write " << source_path;
    return false;
  }
  output << source;
  output.close();

  const char* compiler = std::getenv("CXX");
  const std::string cxx = (compiler == nullptr || compiler[0] == '\0') ? "c++" : compiler;
  const std::string command = ShellQuote(cxx) + " -std=c++17 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 -I" +
      ShellQuote(include_dir) + " -o " + ShellQuote(so_path.string()) + " " + ShellQuote(source_path.string());
  const int result = std::system(command.c_str());
  if (result != 0 || !std::filesystem::exists(so_path)) {
    RT_GLOG(ERROR) << "InferShape codegen: compiler failed, status=" << result << " command=" << command;
    return false;
  }
  artifact->so_path = std::filesystem::absolute(so_path).string();
  RT_GLOG(INFO) << "InferShape codegen: generated " << artifact->so_path << " symbol=" << artifact->symbol;
  return true;
}

bool SerializeInferFuncPayload(const FxInferFuncArtifact& artifact, std::vector<uint8_t>* payload) {
  if (payload == nullptr || artifact.so_path.empty() || artifact.symbol.empty() ||
      artifact.so_path.size() > std::numeric_limits<uint32_t>::max() ||
      artifact.symbol.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  std::vector<uint8_t> body;
  AppendU32(&body, kFxInferFuncAbiVersion);
  AppendU32(&body, static_cast<uint32_t>(artifact.so_path.size()));
  body.insert(body.end(), artifact.so_path.begin(), artifact.so_path.end());
  AppendU32(&body, static_cast<uint32_t>(artifact.symbol.size()));
  body.insert(body.end(), artifact.symbol.begin(), artifact.symbol.end());
  payload->clear();
  AppendU32(payload, kFxConstPayloadMagic);
  AppendU32(payload, static_cast<uint32_t>(FxConstPayloadType::kInferFunc));
  AppendU64(payload, body.size());
  payload->insert(payload->end(), body.begin(), body.end());
  return true;
}

bool ParseInferFuncPayload(const FxSerializedBuffer& buffer, FxInferFuncArtifact* artifact) {
  if (artifact == nullptr || buffer.data == nullptr || buffer.size < 16U) {
    return false;
  }
  size_t offset = 0U;
  uint32_t magic = 0U;
  uint32_t type = 0U;
  uint64_t body_size = 0U;
  if (!ReadU32(buffer.data, buffer.size, &offset, &magic) || !ReadU32(buffer.data, buffer.size, &offset, &type) ||
      !ReadU64(buffer.data, buffer.size, &offset, &body_size) || magic != kFxConstPayloadMagic ||
      type != static_cast<uint32_t>(FxConstPayloadType::kInferFunc) || body_size != buffer.size - offset) {
    return false;
  }
  uint32_t version = 0U;
  uint32_t path_size = 0U;
  uint32_t symbol_size = 0U;
  if (!ReadU32(buffer.data, buffer.size, &offset, &version) || version != kFxInferFuncAbiVersion ||
      !ReadU32(buffer.data, buffer.size, &offset, &path_size) || offset > buffer.size ||
      path_size > buffer.size - offset) {
    return false;
  }
  artifact->so_path.assign(reinterpret_cast<const char*>(buffer.data + offset), path_size);
  offset += path_size;
  if (!ReadU32(buffer.data, buffer.size, &offset, &symbol_size) || offset > buffer.size ||
      symbol_size > buffer.size - offset || symbol_size == 0U) {
    return false;
  }
  artifact->symbol.assign(reinterpret_cast<const char*>(buffer.data + offset), symbol_size);
  return !artifact->so_path.empty();
}

ge::graphStatus DeserializeInferFunc(const FxSerializedBuffer& buffer, FxInferFuncHandle** handle) {
  if (handle == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  *handle = nullptr;
  FxInferFuncArtifact artifact;
  if (!ParseInferFuncPayload(buffer, &artifact)) {
    RT_GLOG(ERROR) << "DeserializeInferFunc: malformed payload";
    return ge::GRAPH_FAILED;
  }
  void* so = dlopen(artifact.so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (so == nullptr) {
    RT_GLOG(ERROR) << "DeserializeInferFunc: dlopen failed for " << artifact.so_path << ": " << dlerror();
    return ge::GRAPH_FAILED;
  }
  dlerror();
  auto func = reinterpret_cast<FxInferShapeFunc>(dlsym(so, artifact.symbol.c_str()));
  const char* error = dlerror();
  if (func == nullptr || error != nullptr) {
    RT_GLOG(ERROR) << "DeserializeInferFunc: dlsym failed for " << artifact.symbol << ": "
                   << (error == nullptr ? "unknown error" : error);
    (void)dlclose(so);
    return ge::GRAPH_FAILED;
  }
  auto* result = new (std::nothrow) FxInferFuncHandle{so, func};
  if (result == nullptr) {
    (void)dlclose(so);
    return ge::GRAPH_FAILED;
  }
  *handle = result;
  return ge::GRAPH_SUCCESS;
}

} // namespace fx_converter
