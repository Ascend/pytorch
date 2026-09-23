#ifndef __FXRT_OPS_ATB_ADAPTER_H__
#define __FXRT_OPS_ATB_ADAPTER_H__

#include <string>
#include <vector>
#include <utility>
#include <optional>
#include <unordered_map>
#include "atb/types.h"
#include "atb/operation.h"
#include "atb/utils.h"
#include "ir/value/value.h"
#include "ops/ascend/aclnn/utils/aclnn_hash.h"

namespace fxrt {
namespace ops {

atb::Context* GetAtbContext(const aclrtStream& stream);

template <typename... Args>
uint64_t AtbHash(const Args&... args) {
  gHashOffset = 0;
  GatherHash(args...);
  return CalcHashId();
}

class ParamSetter {
 public:
  ParamSetter& Input(const ir::Value* value);
  ParamSetter& Input(std::optional<const ir::Value*> value);
  ParamSetter& Output(const ir::Value* value);
  ParamSetter& Output(std::optional<const ir::Value*> value);

  void Clear() {
    variant_pack.inTensors.clear();
    variant_pack.outTensors.clear();
  }

  ParamSetter& SetIndex(const std::vector<size_t> inputs, const std::vector<size_t> outputs) {
    input_ids = std::move(inputs);
    output_ids = std::move(outputs);
    variant_pack.inTensors.clear();
    variant_pack.outTensors.clear();
    return *this;
  }

  void Update(const std::vector<const ir::Value*>& inputs, const ir::Value* output);

  std::vector<size_t> input_ids;
  std::vector<size_t> output_ids;
  atb::VariantPack variant_pack;
};

class AtbContextManager {
 public:
  static AtbContextManager& GetInstance();
  atb::Context* GetContext(const aclrtStream& stream);
  ~AtbContextManager();

 private:
  AtbContextManager() = default;
  std::unordered_map<aclrtStream, atb::Context*> context_map_{};
};

} // namespace ops
} // namespace fxrt

#endif
