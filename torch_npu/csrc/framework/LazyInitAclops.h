#ifndef AT_NPU_ACOPLS_LAZYINITACLOPS_H_
#define AT_NPU_ACOPLS_LAZYINITACLOPS_H_

namespace at_npu {
namespace aclops {

void InitAclops();
void LazyInitAclops();
void InitializeJitCompilationMode();

}  // namespace aclops
}  // namespace at_npu

#endif  // AT_NPU_ACOPLS_LAZYINITACLOPS_H_