#ifndef __OPS_ASCEND_CUSTOM_OP_COMPILED_KERNEL_MUTATION_H__
#define __OPS_ASCEND_CUSTOM_OP_COMPILED_KERNEL_MUTATION_H__

#include <ops/op_base/op_compiled_kernel_mutation.h>

namespace fxrt {
namespace ops {
class AscendOpCompiledKernelMutation : public OpCompiledKernelMutation {
 public:
  AscendOpCompiledKernelMutation() = default;
  ~AscendOpCompiledKernelMutation() override = default;
};
} // namespace ops
} // namespace fxrt

#endif // __OPS_ASCEND_CUSTOM_OP_COMPILED_KERNEL_MUTATION_H__
