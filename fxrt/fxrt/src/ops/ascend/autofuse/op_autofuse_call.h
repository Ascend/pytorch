#ifndef __OPS_ASCEND_AUTOFUSE_OP_AUTOFUSE_CALL_H__
#define __OPS_ASCEND_AUTOFUSE_OP_AUTOFUSE_CALL_H__

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ops/op_register.h"

namespace fxrt {
namespace ops {

/**
 * Native launcher for an Inductor AutoFuse wrapper.so.
 *
 * IR inputs are:
 *   [0] generated typed wrapper shim.so path
 *   [1] wrapper.so path
 *   [2] kernel.so path
 *   [3] static kernel key (empty string means the default kernel)
 *   [4] tuple of mutated argument positions
 *   [5..] original wrapper call arguments
 *
 * AutoFuse wrappers differ in their C signatures.  FXRT's graph-time
 * codegen emits a typed shim for each signature.  This operator only loads
 * that shim and uses its fixed ABI; it never calls an AutoFuse adapter or a
 * kernel symbol directly.
 */
class OpAutofuseCall : public Operator {
 public:
  OpAutofuseCall() = default;
  ~OpAutofuseCall() override;

  void Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output) override;
  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
  bool NeedLaunch() override {
    return false;
  }

  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override {
    return refPairs_;
  }

 private:
  using AbiVersionFunc = uint32_t (*)();
  using ArgNumFunc = uint32_t (*)();
  using InitFunc = int64_t (*)(void**, const char*, const char*, const char*);
  using LaunchFunc = int64_t (*)(void*, const uint64_t*, uint32_t, void*, const char*);
  using FinalizeFunc = int64_t (*)(void*, const char*);

  static constexpr size_t kStubPathInputIndex = 0;
  static constexpr size_t kWrapperPathInputIndex = 1;
  static constexpr size_t kKernelPathInputIndex = 2;
  static constexpr size_t kKernelKeyInputIndex = 3;
  static constexpr size_t kMutatedArgIndicesInputIndex = 4;
  static constexpr size_t kRealInputStartIndex = 5;

  static std::vector<size_t> ParseMutatedArgIndices(const ir::Value* value);
  static std::vector<const ir::Value*> FlattenOutputs(const ir::Value* output);
  static uint64_t EncodeArg(const ir::Value* value, size_t index);
  OpsErrorCode LaunchWrapper(void* stream);
  void SyncOutputMetadata(const ir::Value* output) const;
  void CloseWrapper();

  std::string stubPath_;
  std::string wrapperPath_;
  std::string kernelPath_;
  std::string kernelKey_;
  void* stubHandle_{nullptr};
  void* context_{nullptr};
  AbiVersionFunc abiVersionFunc_{nullptr};
  ArgNumFunc argNumFunc_{nullptr};
  InitFunc initFunc_{nullptr};
  LaunchFunc launchFunc_{nullptr};
  FinalizeFunc finalizeFunc_{nullptr};
  uint32_t expectedArgNum_{0};
  std::vector<const ir::Value*> realInputs_;
  std::vector<std::pair<uint32_t, uint32_t>> refPairs_;
};

} // namespace ops
} // namespace fxrt

#endif // __OPS_ASCEND_AUTOFUSE_OP_AUTOFUSE_CALL_H__
