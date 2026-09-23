#ifndef __OPS_OP_BASE_OP_COMPILED_KERNEL_MUTATION_H__
#define __OPS_OP_BASE_OP_COMPILED_KERNEL_MUTATION_H__

#include <cstdint>
#include <utility>
#include <vector>

#include "ops/op_base/op_python_call.h"

namespace fxrt {
namespace ops {
/**
 * @brief Graph IR counterpart of the `compiled_kernel_wrapper_mutation` HOP.
 *
 * The fx_wrapper keeps a fusion backend's compiled kernels alive inside the inductor host FX graph as HOP nodes
 * (see fxrt/compiled_kernel_hop.py). A fused kernel does not return values, it writes into the buffers
 * passed at its mutated arg positions. Those buffers stay ordinary inputs and each becomes an output that refs its
 * input, so the kernel writes straight into the memory the graph already holds -- no copy, no second allocation, and
 * whatever the buffer held on entry is still there for a kernel that only updates part of it. Because the ref makes
 * the output share the input's storage, a kernel is free to write a graph input in place: the caller sees it.
 * The outputs exist so that reads of a mutated buffer depend on this node rather than on the value it had before.
 *
 * IR layout:
 *   input[0]   : int index of the kernel in the compiled-kernel side table.
 *   input[1]   : tuple[int] of the kernel arg positions the kernel writes.
 *   input[2..] : the kernel args, in kernel arg order.
 *   output     : Tensor, or Tuple[Tensor], one per input[1] entry and in that order, each refing input[2 + pos].
 *
 * The compiled kernel is a python callable, and it is launched on the stream torch considers current -- the same
 * assumption `python_call` already makes -- not on the stream FXRT hands to Launch.
 *
 * Deriving from OpPythonCall reuses its ir::Value -> python jump table, its zero-copy at::Tensor cache, and the way it
 * runs: the callable is invoked from CalcWorkspace and NeedLaunch() is false, so no launch task is ever queued. This
 * matters beyond bookkeeping. Calling the kernel means calling back into the framework, which appends the real launch
 * to the very queue an executor would have queued this op on; running the call from a queued task would put that
 * append behind the ops queued after it. CalcWorkspace runs inline in the executor loop, so the append keeps its
 * place. Only the resolution of the callable differs -- the side table, not a module path.
 */
class OpCompiledKernelMutation : public OpPythonCall {
 public:
  OpCompiledKernelMutation() {
    SetOpType(OpType::PythonCallOp);
  }
  ~OpCompiledKernelMutation() override;

  void Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output) override;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;

  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override {
    return refPairs_;
  }

 private:
  int64_t kernelIdx_{-1};
  // One (output index, input index) pair per mutated arg: the buffer the kernel writes.
  std::vector<std::pair<uint32_t, uint32_t>> refPairs_;
};

} // namespace ops
} // namespace fxrt

#endif // __OPS_OP_BASE_OP_COMPILED_KERNEL_MUTATION_H__
