#include <ATen/ATen.h>

#include "torch_npu/csrc/core/OverflowUtils.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "op_plugin/OpInterface.h"

namespace torch_npu {
namespace utils {
OverflowUtil::OverflowUtil() {}

OverflowUtil::~OverflowUtil() {}

void OverflowUtil::EnableOverflowNpu()
{
    auto result = c10_npu::NpuSysCtrl::GetInstance().OverflowSwitchEnable();
    return;
}

bool OverflowUtil::CheckOverflowNpu()
{
    auto options = at::TensorOptions(c10::DeviceType::PrivateUse1).dtype(at::kFloat);
    at::Tensor tmp = at::empty({ 8 }, options);
    auto floatStatus = op_plugin::npu_alloc_float_status(tmp);
    auto result = op_plugin::npu_get_float_status(floatStatus);
    if (result.cpu()[0].item().toInt() != 0) {
        return true;
    }
    return false;
}

void OverflowUtil::ClearOverflowNpu()
{
    auto options = at::TensorOptions(c10::DeviceType::PrivateUse1).dtype(at::kFloat);
    at::Tensor tmp = at::empty({ 8 }, options);
    auto floatStatus = op_plugin::npu_alloc_float_status(tmp);
    auto result = op_plugin::npu_clear_float_status(floatStatus);
    return;
}
}
}
