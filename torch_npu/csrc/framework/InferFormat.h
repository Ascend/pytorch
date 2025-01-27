#ifndef __PULGIN_NATIVE_UTILS_FORMAT_INFER__
#define __PULGIN_NATIVE_UTILS_FORMAT_INFER__

#include <ATen/ATen.h>

#include "torch_npu/csrc/framework/utils/NPUDefinition.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace at_npu {
namespace native {

// Format is the property of tensor storage. Format is the way to tell an
// operator how the result should be organized in memory and nothing more.
// Storage format collect the helper functions of npu's format. It tell the
// relationship between format and storage.
//
class InferFormat {
public:
    // Feature: The function is used to guess base format
    // The base formats are NCHW, NCDHW, ND, who is not padding.
    // The format transform between other formats should be based
    // on these base formats.(their should convert to base format first.)
    // This function will be called at new, reset, set and so on.
    static std::tuple<aclFormat, aclFormat> GuessFormatUnit(const c10::IntArrayRef& size, aclFormat format);
    // GuessBaseFormat is the base of the format assumption
    // this function is called when apply the new tensor
    static aclFormat GuessBaseFormat(const c10::IntArrayRef& size);
    // this function used to fix format when format and size is not match
    static aclFormat GuessStorageFormat(const c10::IntArrayRef& size, aclFormat format);
    // Features: guess the format of tensor after it called format_contiguous().
    // According to the law of continuity, the output format is same as input format,
    // this function is called to guess the input format, so it also the output format.
    // NOTE: The caller should make sure that the tensor is non-contigous
    static aclFormat GuessFormatWhenContiguous(const at::Tensor &tensor);
    // This api is used to infer storage size when called transdata
    // fix: ND->NZ when dim < 2
    // not effect the storage data.
    static FormatShape GuessStorageSizeWhenConvertFormat(const at::Tensor &tensor);
    // This api is used to judge if tensor is reasonable when size changes.
    // solution: tranform to base format to fix it.
    // fix: NCHW | 5HD -> NCDHW | NCDHW or ND | ND
    // unsqueeze/squeeze/select/flatten/view will change meta data, they will call
    // as_strided and view
    static bool IsDefiniteTensorWhenMetaDataChanges(const at::Tensor &tensor, const c10::IntArrayRef& size);
}; // class InferFormat

} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_UTILS_FORMAT_INFER__