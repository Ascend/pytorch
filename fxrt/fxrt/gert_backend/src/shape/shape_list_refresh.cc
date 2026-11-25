#include "shape/shape_list_refresh.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#include "common/logger.h"
#include "shape/fx_rule_eval.h"
#include "shape/shape_list_utils.h"

namespace fx_converter {
namespace {

bool CopyPayload(void *host, void *dev, const void *src, size_t nbytes) {
  if (host == nullptr || (src == nullptr && nbytes != 0)) return false;
  std::memcpy(host, src, nbytes);
  return dev == nullptr || aclrtMemcpy(dev, nbytes, host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool BindSeed(FxSymbolEnv *env, const FxSymbolEnv::Seed &seed, gert::Tensor **inputs, size_t input_num) {
  if (seed.feed_index < 0 || static_cast<size_t>(seed.feed_index) >= input_num || inputs[seed.feed_index] == nullptr) {
    RT_GLOG(ERROR) << "ES shape refresh: feed index out of range for symbol " << seed.name;
    return false;
  }
  const auto &shape = inputs[seed.feed_index]->GetOriginShape();
  if (seed.dim_index < 0 || static_cast<size_t>(seed.dim_index) >= shape.GetDimNum()) {
    RT_GLOG(ERROR) << "ES shape refresh: dimension index out of range for symbol " << seed.name;
    return false;
  }
  const int64_t value = shape.GetDim(seed.dim_index);
  if (value < 0) {
    RT_GLOG(ERROR) << "ES shape refresh: negative dimension for symbol " << seed.name;
    return false;
  }
  env->SetValue(seed.name, value);
  return true;
}

bool BindSymbols(FxSymbolEnv *env, gert::Tensor **inputs, size_t input_num,
                 const std::vector<std::pair<std::string, int64_t>> &scalar_syms) {
  env->ClearValues();
  if (std::any_of(env->seeds.begin(), env->seeds.end(),
                  [&](const auto &seed) { return !BindSeed(env, seed, inputs, input_num); })) return false;
  env->bound = true;
  for (const auto &kv : scalar_syms) env->SetValue(kv.first, kv.second);
  return true;
}

bool ResolveTemplate(const BuiltExecuteGraph::AiCoreShapeTensorRefresh &refresh, std::vector<int64_t> *values) {
  if (refresh.host == nullptr || refresh.size_template.size() != refresh.axis_exprs.size()) {
    RT_GLOG(ERROR) << "ES shape refresh: invalid shape-list entry";
    return false;
  }
  *values = refresh.size_template;
  try {
    for (size_t i = 0; i < values->size(); ++i) {
      if (refresh.axis_exprs[i] == nullptr) continue;
      (*values)[i] = refresh.axis_exprs[i]->Evaluate();
      if (!refresh.allow_nonpositive && (*values)[i] <= 0) {
        RT_GLOG(ERROR) << "ES shape refresh: invalid resolved dimension " << (*values)[i];
        return false;
      }
    }
  } catch (const std::exception &ex) {
    RT_GLOG(ERROR) << "ES shape refresh: expression evaluation failed: " << ex.what();
    return false;
  }
  return true;
}

bool WriteValues(const BuiltExecuteGraph::AiCoreShapeTensorRefresh &refresh, const std::vector<int64_t> &values) {
  if (refresh.payload_type == ge::DT_INT64) {
    return CopyPayload(refresh.host, refresh.dev, values.data(), values.size() * sizeof(int64_t));
  }
  if (refresh.payload_type == ge::DT_INT32) {
    if (std::any_of(values.begin(), values.end(), [](int64_t value) {
          return value > std::numeric_limits<int32_t>::max() || value < std::numeric_limits<int32_t>::min();
        })) return false;
    std::vector<int32_t> narrowed;
    narrowed.reserve(values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(narrowed),
                   [](int64_t value) { return static_cast<int32_t>(value); });
    return CopyPayload(refresh.host, refresh.dev, narrowed.data(), narrowed.size() * sizeof(int32_t));
  }
  std::vector<double> widened(values.size());
  std::transform(values.begin(), values.end(), widened.begin(), [](int64_t value) { return static_cast<double>(value); });
  if (refresh.payload_type == ge::DT_FLOAT) {
    std::vector<float> narrowed(values.size());
    std::transform(values.begin(), values.end(), narrowed.begin(), [](int64_t value) { return static_cast<float>(value); });
    return CopyPayload(refresh.host, refresh.dev, narrowed.data(), narrowed.size() * sizeof(float));
  }
  if (refresh.payload_type == ge::DT_DOUBLE) {
    return CopyPayload(refresh.host, refresh.dev, widened.data(), widened.size() * sizeof(double));
  }
  std::vector<uint16_t> narrowed(values.size());
  const bool fp16 = refresh.payload_type == ge::DT_FLOAT16;
  std::transform(values.begin(), values.end(), narrowed.begin(), [fp16](int64_t value) {
    const float f = static_cast<float>(value);
    return fp16 ? Float32ToFp16Bits(f) : Float32ToBf16Bits(f);
  });
  return CopyPayload(refresh.host, refresh.dev, narrowed.data(), narrowed.size() * sizeof(uint16_t));
}

}  // namespace

bool RefreshShapeListInputs(const BuiltExecuteGraph &built, gert::Tensor **inputs, size_t input_num,
                            const std::vector<std::pair<std::string, int64_t>> &scalar_syms) {
  auto *env = built.symbol_env.get();
  if (env == nullptr) return built.aicore_shape_refreshes.empty() && scalar_syms.empty();
  if (!BindSymbols(env, inputs, input_num, scalar_syms)) return false;
  for (const auto &refresh : built.aicore_shape_refreshes) {
    std::vector<int64_t> values;
    if (!ResolveTemplate(refresh, &values) || !WriteValues(refresh, values)) return false;
  }
  return true;
}

}  // namespace fx_converter
