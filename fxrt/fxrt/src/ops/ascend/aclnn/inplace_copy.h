#ifndef __OPS_ASCEND_ACLNN_INPLACE_COPY_H__
#define __OPS_ASCEND_ACLNN_INPLACE_COPY_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"

namespace fxrt {
namespace ops {
class AclnnInplaceCopy : public Operator {
 public:
  AclnnInplaceCopy() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnInplaceCopy");
    res_manager_ = std::make_unique<device::ascend::AscendResManager>();
    stream_mng_ = std::make_unique<device::ascend::AscendStreamMng>();
  }
  ~AclnnInplaceCopy() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const {
    return {std::pair<uint32_t, uint32_t>(0, 0)};
  }

 private:
  device::CopyType copyMode_{fxrt::device::CopyType::D2D};
  std::unique_ptr<AclnnExecutor> executor_{nullptr};
  std::unique_ptr<device::ascend::AscendResManager> res_manager_{nullptr};
  std::unique_ptr<device::ascend::AscendStreamMng> stream_mng_{nullptr};
  bool non_blocking_{false};
  bool srcContiguous_{false};
  bool dstContiguous_{false};
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_INPLACE_COPY_H__
