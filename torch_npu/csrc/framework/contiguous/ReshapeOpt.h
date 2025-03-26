#ifndef __PLUGIN_NATIVE_NPU_CONTIGUOUS_RESHAPE__
#define __PLUGIN_NATIVE_NPU_CONTIGUOUS_RESHAPE__

#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"

namespace at_npu {
namespace native {

bool can_use_memecpy_for_NZ_format(const ContiguousTensorDesc &tensor_desc);
bool can_use_memcpy_for_other_format(const ContiguousTensorDesc &tensor_desc);
bool check_reshape_match_flex(const ContiguousTensorDesc &,
                              const ContiguousTensorDesc &);
bool check_reshape_match(const ContiguousTensorDesc &self_desc,
                         const ContiguousTensorDesc &src_desc);
bool check_reshape_match_flex(const ContiguousTensorDesc &);
bool check_reshape_match(const ContiguousTensorDesc &tensor_desc);
bool CanUseMemcpyForOtherFormat(const at::Tensor &tensor);

} // namespace native
} // namespace at_npu

#endif