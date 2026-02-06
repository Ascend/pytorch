#ifndef AT_NPU_ACLOPS_LAZYINITACLOPS_H_
#define AT_NPU_ACLOPS_LAZYINITACLOPS_H_

#include <mutex>
#include <vector>
#include <string>
#include <utility>
#include "third_party/acl/inc/acl/acl_op_compiler.h"

namespace at_npu {
namespace aclops {
class LazyAclopSet {
public:
    static aclError LazyAclSetCompileopt(aclCompileOpt opt, const char *value);
    static void SetCompileopt();

private:
    static std::mutex lazy_set_mutex_;
    static bool acl_op_call_;
    static std::vector<std::pair<aclCompileOpt, std::string>> lazy_acl_opt_;
};

void InitAclops();
void LazyInitAclops();
void InitializeJitCompilationMode();

}  // namespace aclops
}  // namespace at_npu

#endif  // AT_NPU_ACLOPS_LAZYINITACLOPS_H_