#ifndef __OPS_CPU_ATEN_OP_COMPILED_KERNEL_MUTATION_H__
#define __OPS_CPU_ATEN_OP_COMPILED_KERNEL_MUTATION_H__

#include <ops/op_base/op_compiled_kernel_mutation.h>

namespace fxrt {
namespace ops {
class CPUOpCompiledKernelMutation : public OpCompiledKernelMutation {
 public:
  CPUOpCompiledKernelMutation() = default;
  ~CPUOpCompiledKernelMutation() override = default;
};
} // namespace ops
} // namespace fxrt

#endif // __OPS_CPU_ATEN_OP_COMPILED_KERNEL_MUTATION_H__
