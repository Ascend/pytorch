#include <dlfcn.h>
#include "ops/ascend/composite/unify_linear.h"
#include "ops/op_register.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "common/logger.h"

namespace fxrt {
namespace ops {

UnifyLinear::UnifyLinear() : use_atb_linear_(false), atb_loaded_(false), atb_handle_(nullptr) {
  auto soc = fxrt::device::ascend::GetAscendSocVersion();
  if (soc != nullptr) {
    const std::string socName(soc);
    RT_VLOG(VL_OPS) << "Soc version: " << socName;
    if (socName.rfind("Ascend310", 0) == 0 || socName.rfind("ASCEND310", 0) == 0) {
      use_atb_linear_ = true;
    }
  }
  linear_op_ = CreateLinearOperator();
}

std::unique_ptr<Operator> UnifyLinear::CreateLinearOperator() {
  if (use_atb_linear_) {
    std::stringstream errMsg;
    if (lib_loader_.LoadDynamicLib("libops_ascend_atb.so", &errMsg)) {
      atb_handle_ = lib_loader_.GetHandle("libops_ascend_atb.so");

      if (atb_handle_ != nullptr) {
        typedef void* (*CreateAtbLinearFunc)();
        CreateAtbLinearFunc create_func = reinterpret_cast<CreateAtbLinearFunc>(dlsym(atb_handle_, "CreateAtbLinear"));

        if (create_func != nullptr) {
          void* atb_linear_ptr = create_func();
          if (atb_linear_ptr != nullptr) {
            atb_loaded_ = true;
            RT_VLOG(VL_OPS) << "Device is Ascend 310 series, successfully loaded AtbLinear operator.";
            return std::unique_ptr<Operator>(static_cast<Operator*>(atb_linear_ptr));
          }
        }
      }
    }

    RT_VLOG(VL_OPS) << "Failed to load AtbLinear for Ascend 310, falling back to AclnnLinear. Error: " << errMsg.str();
  }

  RT_VLOG(VL_OPS) << "Using AclnnLinear operator.";
  return std::make_unique<AclnnLinear>();
}

OpsErrorCode UnifyLinear::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  CHECK_IF_NULL(linear_op_);
  return linear_op_->CalcWorkspace(input, output, workspaceSize);
}

OpsErrorCode UnifyLinear::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  CHECK_IF_NULL(linear_op_);
  return linear_op_->Launch(input, workspace, workspaceSize, output, stream);
}

FXRT_REG_OP(linear, UnifyLinear, Ascend);

} // namespace ops
} // namespace fxrt
