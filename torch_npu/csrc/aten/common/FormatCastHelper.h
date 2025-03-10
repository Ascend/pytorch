#ifndef __PLUGIN_NATIVE_NPU_COMMON_FORMAT_CAST_HELPER__
#define __PLUGIN_NATIVE_NPU_COMMON_FORMAT_CAST_HELPER__

#include <ATen/ATen.h>

#include "third_party/acl/inc/acl/acl_base.h"

namespace at_npu {
namespace native {

class FormatCastHelper {
public:
    static bool IsSameGroupType(const at::Tensor& src, const at::Tensor& dst);
    static void format_cast_as_base_format(const at::Tensor& src, aclFormat format);
    using FormatCastFunc = std::function<at::Tensor(at::Tensor&, const at::Tensor&)>;
    static bool format_cast_between_group(
        at::Tensor& dst, const at::Tensor& src, FormatCastFunc format_cast_inside_group);
    // this interface is similar to CastBackToOriFormat, but CastBackToOriFormat may have overload problem.
    static at::Tensor ApplyBaseFormatTensorBy(const at::Tensor& src);
    static at::Tensor& CovertSelfToBaseFormat(at::Tensor& src);
private:
    // help function of format_cast_between_group
    static void base_format_cast_nocheck(at::Tensor& dst, const at::Tensor& src);
}; // class FormatCastHelper

} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_COMMON_FORMAT_CAST_HELPER__
