#include "torch_npu/csrc/core/npu/NPUException.h"

c10::WarningHandler* getBaseHandler_()
{
    static c10::WarningHandler warning_handler_ = c10::WarningHandler();
    return &warning_handler_;
};

void warn_(const ::c10::Warning& warning)
{
  getBaseHandler_()->process(warning);
}

namespace c10_npu {

const char *c10_npu_get_error_message()
{
    return c10_npu::acl::AclGetErrMsg();
}

} // namespace c10_npu
