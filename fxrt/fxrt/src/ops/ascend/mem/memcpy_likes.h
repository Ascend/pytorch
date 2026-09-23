#ifndef __OPS_ASCEND_MEM_MEMCPY_LIKES_H__
#define __OPS_ASCEND_MEM_MEMCPY_LIKES_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class MemcpyOpBase : public Operator {
 public:
  MemcpyOpBase() = default;
  ~MemcpyOpBase() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override {
    return {std::pair<uint32_t, uint32_t>(0, 0)};
  }
  bool NeedLaunch() override;
};

#define DefineMemcpyOp(op_name)         \
  class op_name : public MemcpyOpBase { \
   public:                              \
    op_name() {}                        \
    ~op_name() override = default;      \
  }

// view memcpy ops
DefineMemcpyOp(Unsqueeze);
DefineMemcpyOp(Squeeze);
} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_MEM_MEMCPY_LIKES_H__
