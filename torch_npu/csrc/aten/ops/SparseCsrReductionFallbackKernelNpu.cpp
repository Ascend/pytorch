#include <ATen/DeviceGuard.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/_sparse_csr_prod.h>
#include <ATen/ops/_sparse_csr_sum.h>
#include <torch/library.h>
#include <torch_npu/csrc/framework/utils/CpuFallbackUtils.h>

#include <optional>

namespace {

at::Tensor sparse_csr_sum_cpu_fallback(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {
  at_npu::native::CheckCpuFallbackAllowed(
      "aten::_sparse_csr_sum.dim_dtype",
      at_npu::native::CpuFallbackKind::ExplicitKernel,
      "torch_npu currently provides only a CPU implementation for this "
      "Sparse CSR reduction");
  TORCH_NPU_WARN_ONCE(
      "CPU fallback is allowed because "
      "TORCH_NPU_FALLBACK_CPU_DISABLE=0. Since the NPU interface does not "
      "support operator 'aten::_sparse_csr_sum.dim_dtype', the operator will "
      "fall back to CPU.");
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  at::Tensor cpu_self = self.cpu();
  at::Tensor cpu_result = at::_sparse_csr_sum(cpu_self, dim, keepdim, dtype);
  return cpu_result.to(self.device());
}

at::Tensor sparse_csr_prod_cpu_fallback(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {
  at_npu::native::CheckCpuFallbackAllowed(
      "aten::_sparse_csr_prod.dim_dtype",
      at_npu::native::CpuFallbackKind::ExplicitKernel,
      "torch_npu currently provides only a CPU implementation for this "
      "Sparse CSR reduction");
  TORCH_NPU_WARN_ONCE(
      "CPU fallback is allowed because "
      "TORCH_NPU_FALLBACK_CPU_DISABLE=0. Since the NPU interface does not "
      "support operator 'aten::_sparse_csr_prod.dim_dtype', the operator will "
      "fall back to CPU.");
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  at::Tensor cpu_self = self.cpu();
  at::Tensor cpu_result = at::_sparse_csr_prod(cpu_self, dim, keepdim, dtype);
  return cpu_result.to(self.device());
}

TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m) {
  m.impl("_sparse_csr_sum.dim_dtype", TORCH_FN(sparse_csr_sum_cpu_fallback));
  m.impl("_sparse_csr_prod.dim_dtype", TORCH_FN(sparse_csr_prod_cpu_fallback));
}

} // namespace
