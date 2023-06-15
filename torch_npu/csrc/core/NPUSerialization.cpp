#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/FormatCastHelper.h"
#include "torch_npu/csrc/core/NPUSerialization.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"

namespace torch_npu {

std::unordered_map<std::string, aclFormat> FORMAT_INFO = {
    {"NC1HWC0", ACL_FORMAT_NC1HWC0},
    {"ND", ACL_FORMAT_ND},
    {"NCHW", ACL_FORMAT_NCHW},
    {"NHWC", ACL_FORMAT_NHWC},
    {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
    {"FRACTAL_Z", ACL_FORMAT_FRACTAL_Z},
    {"NDHWC", ACL_FORMAT_NDHWC},
    {"NCDHW", ACL_FORMAT_NCDHW},
    {"NDC1HWC0", ACL_FORMAT_NDC1HWC0},
    {"FRACTAL_Z_3D", ACL_FRACTAL_Z_3D},
};

void npu_info_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& map) {
  std::string src_format_name = at_npu::native::FormatHelper::GetFormatName(t);
  map[src_format_name] = true;
}

void npu_info_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& map) {
  // Set the true stroage description
  if (t.is_contiguous()) {
    at_npu::native::StorageDescHelper::SetDesc(const_cast<at::Tensor&>(t), t.sizes(), t.strides());
  }
  auto iter_end = FORMAT_INFO.end();
  for (auto m : map) {
    // Filter out irrelevant key information.
    if (FORMAT_INFO.find(m.first) != iter_end){
      at_npu::native::NPUNativeFunctions::npu_format_cast_(const_cast<at::Tensor&>(t), FORMAT_INFO[m.first]);
      return;
    }
  }
}

}
