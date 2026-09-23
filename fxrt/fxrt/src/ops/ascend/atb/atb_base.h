#ifndef __OPS_ASCEND_ATB_ATB_KERNEL_MOD_H__
#define __OPS_ASCEND_ATB_ATB_KERNEL_MOD_H__

#include <deque>
#include <vector>
#include <string>
#include <unordered_map>
#include "common/common.h"
#include "common/visible.h"
#include "ops/operator.h"
#include "ops/utils/utils.h"
#include "atb/atb_infer.h"
#include "ops/ascend/atb/atb_adapter.h"

namespace fxrt {
namespace ops {

struct AtbCacheEntry {
  atb::Operation* op = nullptr;
  size_t workspace_size = 0;
  bool workspace_cached = false;
};

class FXRT_EXPORT AtbBase : public Operator {
 public:
  explicit AtbBase(const std::string& op_name)
      : op_name_(op_name), current_hash_id_(0), op_(nullptr), cache_capacity_(GetAtbCacheCapacity()) {}
  ~AtbBase() override;

  OpsErrorCode CalcWorkspace(
      const std::vector<const ir::Value*>& inputs,
      const ir::Value* output,
      size_t* workspace_size) override = 0;

  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& inputs,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override = 0;

 protected:
  std::unordered_map<uint64_t, AtbCacheEntry>::iterator AppendCacheEntry(uint64_t hash);

  template <typename ParamType>
  AtbCacheEntry& GetOrCreateEntry(
      const ParamType& param,
      const std::vector<const ir::Value*>& inputs,
      const ir::Value* output) {
    uint64_t hash = AtbHash(inputs, output, op_name_);
    current_hash_id_ = hash;

    auto it = cache_.find(hash);
    if (it == cache_.end()) {
      it = AppendCacheEntry(hash);
    }

    AtbCacheEntry& entry = it->second;
    if (entry.op == nullptr) {
      auto ret = atb::CreateOperation(param, &entry.op);
      if (ret != 0) {
        RT_GLOG(EXCEPTION) << "Failed to create ATB operation " << op_name_ << ", ret: " << ret;
      }
    }
    op_ = entry.op;
    return entry;
  }

  OpsErrorCode GetWorkspaceSize(AtbCacheEntry& entry, atb::VariantPack& variant_pack, size_t* workspace_size);

  OpsErrorCode LaunchAtb(atb::VariantPack variant_pack, void* workspace_ptr, size_t workspace_size, aclrtStream stream);

  std::string op_name_;
  uint64_t current_hash_id_;
  atb::Operation* op_;
  std::unordered_map<uint64_t, AtbCacheEntry> cache_;
  // FIFO queue for eviction by insertion order to prevent unbounded growth.
  std::deque<uint64_t> cache_fifo_;
  size_t cache_capacity_ = 64;
  ParamSetter param_setter_;
};

} // namespace ops
} // namespace fxrt

#endif
