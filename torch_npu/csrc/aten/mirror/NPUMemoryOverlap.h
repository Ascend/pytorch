#ifndef __PLUGIN_NATIVE_UTILS_NPU_MEMORY_OVERLAP__
#define __PLUGIN_NATIVE_UTILS_NPU_MEMORY_OVERLAP__

#include <ATen/ATen.h>

namespace at_npu {
namespace native {

// MemOverlap: Whether or not there is memory overlap
//
// NO: Absolutely no memory overlap
// YES: Absolutely yes memory overlap
// TOO_HARD: There might be memory overlap, but it was too expensive to compute.
//
// NB: Please update the python test for these if you renumber them.
enum class MemOverlap { NO, YES, TOO_HARD };
enum class MemOverlapStatus { FULL, PARTIAL, NO, TOO_HARD };

MemOverlap has_internal_overlap(const at::Tensor& t);
MemOverlap has_internal_overlap(at::TensorImpl* t);

void assert_no_internal_overlap(const at::Tensor& t);
void assert_no_internal_overlap(at::TensorImpl* t);

MemOverlapStatus get_overlap_status(const at::Tensor& a, const at::Tensor& b);
MemOverlapStatus get_overlap_status(const at::TensorImpl* a, const at::TensorImpl* b);

void assert_no_partial_overlap(const at::Tensor& a, const at::Tensor& b);
void assert_no_partial_overlap(at::TensorImpl* a, at::TensorImpl* b);

} // namespace native
} // namespace at_npu

#endif